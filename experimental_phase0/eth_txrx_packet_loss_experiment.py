#!/usr/bin/env python3
"""
eth0/eth1 TX/RX Rate + HARQ Experiment (Uplink & Downlink)
============================================================
Goal: Observe how tc packet loss on srsdu0 eth0 (egress) affects:
  - eth0/eth1 TX/RX byte & packet rates
  - HARQ retransmission counters (uplink-specific and downlink-specific)
  - Bitrate, MCS, CRC/BLER indicators
  - tc's own drop statistics

Design (4 phases, ~20 minutes total):
  Phase 1: UPLINK baseline   - 5 min, 0% loss, UE→5GC UDP uplink (iperf -c)
  Phase 2: UPLINK stress     - 5 min, 2% loss on du0 eth0, same uplink traffic
  Phase 3: DOWNLINK baseline - 5 min, 0% loss, 5GC→UE UDP downlink (iperf -R)
  Phase 4: DOWNLINK stress   - 5 min, 2% loss on du0 eth0, same downlink traffic

tc netem on eth0 = EGRESS only:
  - Uplink:  DU→CU F1U(UDP) packets dropped = uplink user data lost
  - Downlink: DU→UE ZMQ(TCP) packets dropped = TCP retransmits, PHY perfect

What HARQ counters to watch per direction:
  UPLINK HARQ (UE→DU):
    ul_nof_nok          = MAC-level HARQ NACKs (UE retransmissions)
    ul_nof_ok           = MAC-level HARQ ACKs
    nof_pusch_invalid_harqs = invalid PUSCH HARQ-ACKs
    avg_pusch_harq_delay    = average PUSCH HARQ feedback delay
    max_pusch_harq_delay    = max PUSCH HARQ feedback delay
    ul_mcs              = uplink MCS (drops if BLER increases)
    pusch_snr_db        = PUSCH SNR

  DOWNLINK HARQ (DU→UE):
    dl_nof_nok          = MAC-level HARQ NACKs (DU retransmissions)
    dl_nof_ok           = MAC-level HARQ ACKs
    nof_pucch_f0f1_invalid_harqs = invalid PUCCH Format0/1 HARQ
    nof_pucch_f2f3f4_invalid_harqs = invalid PUCCH Format2/3/4 HARQ
    avg_pucch_harq_delay    = average PUCCH HARQ feedback delay
    max_pucch_harq_delay    = max PUCCH HARQ feedback delay
    dl_mcs              = downlink MCS
    pucch_snr_db        = PUCCH SNR

Usage:
  python3 experimental_phase0/eth_txrx_packet_loss_experiment.py
"""

import asyncio
import aiohttp
from aiohttp import ClientTimeout
import subprocess
import time
import signal
import sys
import csv
import os
import logging
from datetime import datetime
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync

# =====================================================================
# CONFIGURATION
# =====================================================================
TARGET_CONTAINER = "srsdu0"
IPERF_SERVER_CONTAINER = "open5gs_5gc"
IPERF_TARGET_IP = "10.45.1.1"
TARGET_MBPS = 10.0

INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "605bc59413b7d5457d181ccf20f9fda15693f81b068d70396cc183081b264f3b"
INFLUXDB_ORG = "srs"
INFLUXDB_BUCKET = "srsran"

PHASE_DURATION = 300        # 5 minutes per phase
STRESS_LOSS_PCT = 2.0       # 2% packet loss
FETCH_INTERVAL = 1.0        # collect every 1 second

# ---- InfluxDB fields by direction ----
UPLINK_HARQ_FIELDS = [
    'ul_nof_nok', 'ul_nof_ok',
    'nof_pusch_invalid_harqs',
    'avg_pusch_harq_delay', 'max_pusch_harq_delay',
]
DOWNLINK_HARQ_FIELDS = [
    'dl_nof_nok', 'dl_nof_ok',
    'nof_pucch_f0f1_invalid_harqs', 'nof_pucch_f2f3f4_invalid_harqs',
    'avg_pucch_harq_delay', 'max_pucch_harq_delay',
]
BITRATE_FIELDS = ['ul_brate', 'dl_brate']
MCS_FIELDS = ['ul_mcs', 'dl_mcs']
SIGNAL_FIELDS = ['cqi', 'ri', 'bsr', 'pusch_snr_db', 'pucch_snr_db']

ALL_INFLUX_FIELDS = (UPLINK_HARQ_FIELDS + DOWNLINK_HARQ_FIELDS +
                     BITRATE_FIELDS + MCS_FIELDS + SIGNAL_FIELDS)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE_DIR, "experimental_phase0")
DATA_DIR = os.path.join(EXP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, "eth_txrx_harq_updown_experiment.csv")

# Logging
log_file = os.path.join(DATA_DIR, "eth_txrx_harq_updown.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# =====================================================================
# UTILITIES
# =====================================================================
active_iperf_procs = []
prev_eth_stats = {}
prev_time = 0.0


def cleanup_stress():
    """Remove any tc netem rules from target container."""
    subprocess.run(
        f"docker exec {TARGET_CONTAINER} tc qdisc del dev eth0 root",
        shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )


def apply_packet_loss(loss_pct):
    """Apply packet loss via tc netem on target container eth0 (egress)."""
    cmd = (f"docker exec {TARGET_CONTAINER} tc qdisc replace dev eth0 "
           f"root netem loss {loss_pct:.2f}%")
    result = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    if result.returncode != 0:
        logging.error(f"Failed to apply {loss_pct}% loss: {result.stderr.decode()}")
    else:
        logging.info(f"Applied {loss_pct:.1f}% packet loss on {TARGET_CONTAINER} eth0 (egress)")


def kill_all_iperf():
    """Kill iperf in server and all UE containers."""
    global active_iperf_procs
    for p in active_iperf_procs:
        try:
            p.terminate()
            p.wait(timeout=2)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    active_iperf_procs = []

    subprocess.run(f"docker exec {IPERF_SERVER_CONTAINER} pkill -9 iperf",
                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ue_ids = subprocess.check_output(
            "docker ps --filter 'name=srsue' --format '{{.ID}}'",
            shell=True, text=True
        ).strip().split('\n')
        for uid in ue_ids:
            if uid:
                subprocess.run(f"docker exec {uid} pkill -9 iperf", shell=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def cleanup_and_exit(signum=None, frame=None):
    logging.info("Cleaning up...")
    cleanup_stress()
    kill_all_iperf()
    logging.info("Cleanup done.")
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup_and_exit)


def start_iperf_uplink():
    """Start constant UDP UPLINK traffic: UE → 5GC (iperf client on UE)."""
    global active_iperf_procs
    kill_all_iperf()
    time.sleep(1)

    # Start iperf server on 5GC
    subprocess.run(
        f"docker exec -d {IPERF_SERVER_CONTAINER} iperf -s -u -B {IPERF_TARGET_IP}",
        shell=True
    )
    logging.info(f"iperf UDP server started on {IPERF_SERVER_CONTAINER} ({IPERF_TARGET_IP})")
    time.sleep(2)

    # Start iperf clients on UEs (uplink: UE → 5GC)
    try:
        ue_ids = subprocess.check_output(
            "docker ps --filter 'name=srsue' --format '{{.ID}}'",
            shell=True, text=True
        ).strip().split('\n')
    except Exception:
        ue_ids = []

    for uid in ue_ids:
        if not uid:
            continue
        cmd = ["docker", "exec", uid, "iperf", "-c", IPERF_TARGET_IP,
               "-u", "-b", f"{TARGET_MBPS}M", "-t", "86400"]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        active_iperf_procs.append(p)
        logging.info(f"UPLINK: Started {TARGET_MBPS}Mbps UDP UE {uid[:12]} → {IPERF_TARGET_IP}")

    logging.info(f"Total UPLINK streams: {len(active_iperf_procs)}")


def start_iperf_downlink():
    """Start constant UDP DOWNLINK traffic: 5GC → UE (iperf server on UE, client on 5GC)."""
    global active_iperf_procs
    kill_all_iperf()
    time.sleep(1)

    # Get UE TUN IPs (10.45.x.x)
    try:
        ue_ids = subprocess.check_output(
            "docker ps --filter 'name=srsue' --format '{{.ID}}'",
            shell=True, text=True
        ).strip().split('\n')
        ue_ids = [u for u in ue_ids if u]
    except Exception:
        ue_ids = []

    if not ue_ids:
        logging.error("No UE containers found!")
        return

    # Start iperf servers on all UEs
    ue_tun_ips = []
    for uid in ue_ids:
        # Start server
        subprocess.run(
            f"docker exec -d {uid} iperf -s -u",
            shell=True
        )
        # Get TUN IP
        try:
            ip = subprocess.check_output(
                f"docker exec {uid} ip -4 addr show tun_srsue 2>/dev/null | grep -oP '(?<=inet )\\S+' | cut -d/ -f1",
                shell=True, text=True
            ).strip()
            if ip:
                ue_tun_ips.append(ip)
                logging.info(f"DOWNLINK: UE {uid[:12]} server on {ip}")
        except Exception:
            pass
    time.sleep(2)

    # Start iperf clients on 5GC targeting each UE
    for ip in ue_tun_ips:
        cmd = ["docker", "exec", IPERF_SERVER_CONTAINER, "iperf",
               "-c", ip, "-u", "-b", f"{TARGET_MBPS}M", "-t", "86400"]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        active_iperf_procs.append(p)
        logging.info(f"DOWNLINK: Started {TARGET_MBPS}Mbps UDP {IPERF_SERVER_CONTAINER} → {ip}")

    logging.info(f"Total DOWNLINK streams: {len(active_iperf_procs)}")


# =====================================================================
# INTERFACE STATS FROM /proc/net/dev
# =====================================================================
def get_interface_stats():
    """
    Read /proc/net/dev inside srsdu0 for eth0 and eth1 TX/RX counters.
    """
    stats = {}
    try:
        output = subprocess.check_output(
            f"docker exec {TARGET_CONTAINER} cat /proc/net/dev",
            shell=True, text=True, stderr=subprocess.DEVNULL
        )
        for line in output.strip().split('\n'):
            line = line.strip()
            for iface in ['eth0', 'eth1']:
                if line.startswith(f'{iface}:'):
                    parts = line.split(':')[1].split()
                    # /proc/net/dev columns:
                    # RX: bytes packets errs drop fifo frame compressed multicast
                    # TX: bytes packets errs drop fifo colls carrier compressed
                    stats[f'{iface}_rx_bytes'] = int(parts[0])
                    stats[f'{iface}_rx_packets'] = int(parts[1])
                    stats[f'{iface}_rx_errs'] = int(parts[2])
                    stats[f'{iface}_rx_drop'] = int(parts[3])
                    stats[f'{iface}_tx_bytes'] = int(parts[8])
                    stats[f'{iface}_tx_packets'] = int(parts[9])
                    stats[f'{iface}_tx_errs'] = int(parts[10])
                    stats[f'{iface}_tx_drop'] = int(parts[11])
    except Exception as e:
        logging.error(f"Failed to read /proc/net/dev: {e}")
    return stats


def compute_rates(current, prev, elapsed):
    """Compute per-second rates from cumulative counters."""
    rates = {}
    if not prev or elapsed <= 0:
        for key in current:
            rates[f'{key}_persec'] = 0.0
        return rates
    for key, val in current.items():
        prev_val = prev.get(key, val)
        diff = max(0, val - prev_val)
        rates[f'{key}_persec'] = diff / elapsed
    return rates


# =====================================================================
# tc STATISTICS
# =====================================================================
def get_tc_stats():
    """Get tc qdisc stats: how many packets tc actually sent/dropped."""
    stats = {'tc_sent_pkts': 0, 'tc_dropped_pkts': 0, 'tc_sent_bytes': 0}
    try:
        output = subprocess.check_output(
            f"docker exec {TARGET_CONTAINER} tc -s qdisc show dev eth0",
            shell=True, text=True, stderr=subprocess.DEVNULL
        )
        for line in output.split('\n'):
            line = line.strip()
            if 'Sent' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == 'Sent':
                        stats['tc_sent_bytes'] = int(parts[i+1])
                        stats['tc_sent_pkts'] = int(parts[i+3])
                    if p.startswith('(dropped'):
                        stats['tc_dropped_pkts'] = int(parts[i+1].rstrip(','))
    except Exception:
        pass
    return stats


# =====================================================================
# INFLUXDB METRICS
# =====================================================================
async def fetch_influx_metrics():
    """Fetch all HARQ + bitrate + MCS + signal metrics from InfluxDB."""
    data = {}
    try:
        async with InfluxDBClientAsync(
            url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG
        ) as client:
            query_api = client.query_api()
            flux_query = (
                f'from(bucket: "{INFLUXDB_BUCKET}") |> range(start: -30s) '
                f'|> filter(fn: (r) => r["_measurement"] == "ue_info") '
                f'|> last() '
                f'|> pivot(rowKey:["_time"], columnKey: ["_field"], '
                f'valueColumn: "_value")'
            )
            tables = await query_api.query(flux_query)
            for table in tables:
                for record in table.records:
                    pci = record.values.get("pci")
                    rnti = record.values.get("rnti")
                    for field, value in record.values.items():
                        if field in ['_start', '_stop', '_time',
                                     '_measurement', 'rnti', 'pci',
                                     'testbed', 'result', 'table']:
                            continue
                        if field in ALL_INFLUX_FIELDS:
                            key = f"PCI{pci}_RNTI{rnti}_{field}"
                            try:
                                data[key] = (float(value)
                                             if value is not None
                                             and value != 'n/a'
                                             else 0.0)
                            except (ValueError, TypeError):
                                data[key] = 0.0
    except Exception as e:
        logging.error(f"InfluxDB fetch error: {e}")
    return data


# =====================================================================
# MAIN EXPERIMENT
# =====================================================================
async def run_experiment():
    global prev_eth_stats, prev_time

    logging.info("=" * 70)
    logging.info("ETH0/ETH1 TX/RX + HARQ EXPERIMENT (UPLINK & DOWNLINK)")
    logging.info("=" * 70)
    logging.info(f"Target:     {TARGET_CONTAINER}")
    logging.info(f"Phase dur:  {PHASE_DURATION}s ({PHASE_DURATION//60} min)")
    logging.info(f"Loss:       {STRESS_LOSS_PCT}%")
    logging.info(f"Output:     {OUTPUT_FILE}")
    logging.info("")
    logging.info("PHASES:")
    logging.info("  1. UL baseline  (0% loss, UE→5GC iperf UDP)")
    logging.info("  2. UL stress    (2% loss on du0 eth0 egress)")
    logging.info("  3. DL baseline  (0% loss, 5GC→UE iperf UDP)")
    logging.info("  4. DL stress    (2% loss on du0 eth0 egress)")
    logging.info("")
    logging.info("tc on eth0 EGRESS affects:")
    logging.info("  UL: DU→CU F1U(UDP) = uplink data LOST (no retx)")
    logging.info("  DL: DU→UE ZMQ(TCP) = IQ samples survive (TCP retx)")
    logging.info("=" * 70)

    # ---- Discover InfluxDB fields ----
    logging.info("Starting initial traffic to discover InfluxDB schema...")
    start_iperf_uplink()
    await asyncio.sleep(15)
    init_data = await fetch_influx_metrics()
    influx_headers = sorted(init_data.keys())
    logging.info(f"Found {len(influx_headers)} InfluxDB fields")
    kill_all_iperf()
    await asyncio.sleep(3)

    # ---- CSV headers ----
    iface_raw = [
        'eth0_rx_bytes', 'eth0_tx_bytes',
        'eth0_rx_packets', 'eth0_tx_packets',
        'eth0_rx_errs', 'eth0_tx_errs',
        'eth0_rx_drop', 'eth0_tx_drop',
        'eth1_rx_bytes', 'eth1_tx_bytes',
        'eth1_rx_packets', 'eth1_tx_packets',
        'eth1_rx_errs', 'eth1_tx_errs',
        'eth1_rx_drop', 'eth1_tx_drop',
    ]
    iface_rate = [f'{h}_persec' for h in iface_raw]
    tc_hdrs = ['tc_sent_pkts', 'tc_dropped_pkts', 'tc_sent_bytes']

    headers = (
        ["timestamp", "phase", "direction", "packet_loss_pct", "delta_sec"]
        + iface_raw + iface_rate + tc_hdrs + influx_headers
    )

    with open(OUTPUT_FILE, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=headers).writeheader()

    row_count = 0
    prev_eth_stats = get_interface_stats()
    prev_time = time.time()

    async def collect_row(phase, direction, loss_pct):
        nonlocal row_count, prev_eth_stats, prev_time
        now = time.time()
        delta = now - prev_time
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Collect all data
        cur_stats = get_interface_stats()
        tc_stats = get_tc_stats()
        influx_data = await fetch_influx_metrics()
        rates = compute_rates(cur_stats, prev_eth_stats, delta)

        row = {
            "timestamp": ts,
            "phase": phase,
            "direction": direction,
            "packet_loss_pct": f"{loss_pct:.2f}",
            "delta_sec": f"{delta:.3f}",
        }
        for h in iface_raw:
            row[h] = cur_stats.get(h, 0)
        for h in iface_rate:
            row[h] = f"{rates.get(h, 0.0):.1f}"
        for h in tc_hdrs:
            row[h] = tc_stats.get(h, 0)
        for h in influx_headers:
            row[h] = influx_data.get(h, 0.0)

        with open(OUTPUT_FILE, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=headers, extrasaction='ignore').writerow(row)

        prev_eth_stats = cur_stats
        prev_time = now
        row_count += 1
        wait = max(0, FETCH_INTERVAL - (time.time() - now))
        await asyncio.sleep(wait)

    async def run_phase(phase_name, direction, loss_pct, duration):
        nonlocal row_count
        phase_start_rows = row_count
        end_time = time.time() + duration
        while time.time() < end_time:
            await collect_row(phase_name, direction, loss_pct)
            if (row_count - phase_start_rows) % 60 == 0 and (row_count - phase_start_rows) > 0:
                remaining = int(end_time - time.time())
                logging.info(
                    f"  [{phase_name}] rows={row_count - phase_start_rows}, "
                    f"remaining={remaining}s"
                )
        logging.info(
            f"  {phase_name} complete: {row_count - phase_start_rows} rows"
        )

    # ==================================================================
    # PHASE 1: UPLINK BASELINE
    # ==================================================================
    logging.info(f"\n{'='*60}")
    logging.info("PHASE 1: UPLINK BASELINE (0% loss)")
    logging.info(f"{'='*60}")
    cleanup_stress()
    start_iperf_uplink()
    await asyncio.sleep(5)
    await run_phase("ul_baseline", "uplink", 0.0, PHASE_DURATION)

    # ==================================================================
    # PHASE 2: UPLINK STRESS
    # ==================================================================
    logging.info(f"\n{'='*60}")
    logging.info(f"PHASE 2: UPLINK STRESS ({STRESS_LOSS_PCT}% loss on eth0 egress)")
    logging.info(f"{'='*60}")
    apply_packet_loss(STRESS_LOSS_PCT)
    await run_phase("ul_stress", "uplink", STRESS_LOSS_PCT, PHASE_DURATION)
    cleanup_stress()
    kill_all_iperf()
    await asyncio.sleep(5)

    # ==================================================================
    # PHASE 3: DOWNLINK BASELINE
    # ==================================================================
    logging.info(f"\n{'='*60}")
    logging.info("PHASE 3: DOWNLINK BASELINE (0% loss)")
    logging.info(f"{'='*60}")
    start_iperf_downlink()
    await asyncio.sleep(5)
    await run_phase("dl_baseline", "downlink", 0.0, PHASE_DURATION)

    # ==================================================================
    # PHASE 4: DOWNLINK STRESS
    # ==================================================================
    logging.info(f"\n{'='*60}")
    logging.info(f"PHASE 4: DOWNLINK STRESS ({STRESS_LOSS_PCT}% loss on eth0 egress)")
    logging.info(f"{'='*60}")
    apply_packet_loss(STRESS_LOSS_PCT)
    await run_phase("dl_stress", "downlink", STRESS_LOSS_PCT, PHASE_DURATION)

    # ==================================================================
    # DONE
    # ==================================================================
    cleanup_stress()
    kill_all_iperf()

    logging.info(f"\n{'='*70}")
    logging.info("EXPERIMENT COMPLETE")
    logging.info(f"{'='*70}")
    logging.info(f"Total rows: {row_count}")
    logging.info(f"Output:     {OUTPUT_FILE}")
    logging.info("")
    logging.info("ANALYSIS GUIDE:")
    logging.info("")
    logging.info("UPLINK (UE → DU → CU):")
    logging.info("  tc drops F1U(UDP) packets on DU→CU path (egress)")
    logging.info("  Watch: eth0_tx_bytes_persec  -- should show drops being sent")
    logging.info("  Watch: tc_dropped_pkts       -- confirms tc is dropping")
    logging.info("  Watch: ul_brate              -- should DECREASE (F1U loss)")
    logging.info("  Watch: ul_nof_nok            -- HARQ NACKs (expect 0 in ZMQ)")
    logging.info("  Watch: nof_pusch_invalid_harqs -- PUSCH HARQ failures")
    logging.info("  Watch: eth1_tx_bytes_persec  -- metrics plane, should be STABLE")
    logging.info("")
    logging.info("DOWNLINK (CU → DU → UE):")
    logging.info("  tc drops ZMQ(TCP) packets on DU→UE path (egress)")
    logging.info("  BUT TCP retransmits → PHY always gets perfect IQ")
    logging.info("  Watch: eth0_tx_bytes_persec  -- may INCREASE (TCP retransmissions)")
    logging.info("  Watch: dl_brate              -- may stay stable (TCP saves it)")
    logging.info("  Watch: dl_nof_nok            -- HARQ NACKs (expect 0 in ZMQ)")
    logging.info("  Watch: nof_pucch_f0f1_invalid_harqs -- PUCCH HARQ failures")
    logging.info("  Watch: eth1_tx_bytes_persec  -- metrics plane, should be STABLE")
    logging.info("")
    logging.info("KEY CONCLUSION:")
    logging.info("  If eth0 tx drops + ul_brate drops BUT ul_nof_nok = 0:")
    logging.info("  → F1U(UDP) loss causes RLC retransmissions (above MAC)")
    logging.info("  → But srsRAN ue_info only exposes MAC-level HARQ counters")
    logging.info("  → RLC retx counters are NOT in ue_info measurement")
    logging.info("  → Need to check 'rlc_info' or 'gnb_info' measurements")
    logging.info("    for fields like: tx_retx_pdus, rx_retx_pdus, etc.")


if __name__ == "__main__":
    cleanup_stress()
    try:
        asyncio.run(run_experiment())
    except KeyboardInterrupt:
        cleanup_and_exit()
    finally:
        cleanup_and_exit()
