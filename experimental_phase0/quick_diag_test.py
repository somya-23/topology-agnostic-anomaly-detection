#!/usr/bin/env python3
"""
Quick Diagnostic Test: ul_brate root cause analysis
=====================================================
2 min baseline + 2 min 2% packet loss on du0 eth0
Collects: ul_brate, bsr, ul_mcs, ul_nof_ok, ul_nof_nok, pusch_snr_db,
          eth0 tx/rx rates, tc drop stats
"""

import asyncio
import subprocess
import time
import signal
import sys
import csv
import os
import logging
from datetime import datetime
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync

# Config
TARGET_CONTAINER = "srsdu0"
IPERF_SERVER_CONTAINER = "open5gs_5gc"
IPERF_TARGET_IP = "10.45.1.1"
TARGET_MBPS = 10.0

INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "605bc59413b7d5457d181ccf20f9fda15693f81b068d70396cc183081b264f3b"
INFLUXDB_ORG = "srs"
INFLUXDB_BUCKET = "srsran"

BASELINE_DURATION = 120   # 2 min
STRESS_DURATION = 120     # 2 min
STRESS_LOSS_PCT = 2.0
FETCH_INTERVAL = 1.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "experimental_phase0", "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, "quick_diag_test.csv")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, "quick_diag_test.log")),
        logging.StreamHandler()
    ]
)

# All fields we want from InfluxDB
INFLUX_FIELDS = [
    'ul_brate', 'dl_brate', 'bsr',
    'ul_mcs', 'dl_mcs',
    'ul_nof_ok', 'ul_nof_nok',
    'dl_nof_ok', 'dl_nof_nok',
    'pusch_snr_db', 'pucch_snr_db',
    'cqi', 'ri',
    'nof_pusch_invalid_harqs',
    'nof_pucch_f0f1_invalid_harqs',
]

active_procs = []
prev_stats = {}
prev_time = 0.0


def cleanup_stress():
    subprocess.run(f"docker exec {TARGET_CONTAINER} tc qdisc del dev eth0 root",
                   shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


def apply_loss(pct):
    r = subprocess.run(
        f"docker exec {TARGET_CONTAINER} tc qdisc replace dev eth0 root netem loss {pct:.2f}%",
        shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE
    )
    if r.returncode == 0:
        logging.info(f"Applied {pct:.1f}% packet loss on {TARGET_CONTAINER} eth0")
    else:
        logging.error(f"tc failed: {r.stderr.decode()}")


def kill_iperf():
    global active_procs
    for p in active_procs:
        try:
            p.terminate(); p.wait(timeout=2)
        except:
            try: p.kill()
            except: pass
    active_procs = []
    subprocess.run(f"docker exec {IPERF_SERVER_CONTAINER} pkill -9 iperf",
                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ues = subprocess.check_output(
            "docker ps --filter 'name=srsue' --format '{{.ID}}'", shell=True, text=True
        ).strip().split('\n')
        for u in ues:
            if u:
                subprocess.run(f"docker exec {u} pkill -9 iperf", shell=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass


def cleanup_and_exit(signum=None, frame=None):
    logging.info("Cleaning up...")
    cleanup_stress()
    kill_iperf()
    logging.info("Done.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)


def start_uplink():
    global active_procs
    kill_iperf()
    time.sleep(1)
    subprocess.run(f"docker exec -d {IPERF_SERVER_CONTAINER} iperf -s -u -B {IPERF_TARGET_IP}", shell=True)
    time.sleep(2)
    try:
        ues = subprocess.check_output(
            "docker ps --filter 'name=srsue' --format '{{.ID}}'", shell=True, text=True
        ).strip().split('\n')
    except:
        ues = []
    for u in ues:
        if not u: continue
        p = subprocess.Popen(
            ["docker", "exec", u, "iperf", "-c", IPERF_TARGET_IP, "-u", "-b", f"{TARGET_MBPS}M", "-t", "86400"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        active_procs.append(p)
        logging.info(f"Started {TARGET_MBPS}Mbps UL on UE {u[:12]}")


def get_eth_stats():
    stats = {}
    try:
        out = subprocess.check_output(
            f"docker exec {TARGET_CONTAINER} cat /proc/net/dev", shell=True, text=True, stderr=subprocess.DEVNULL
        )
        for line in out.strip().split('\n'):
            line = line.strip()
            if line.startswith('eth0:'):
                p = line.split(':')[1].split()
                stats['eth0_rx_bytes'] = int(p[0])
                stats['eth0_rx_pkts'] = int(p[1])
                stats['eth0_tx_bytes'] = int(p[8])
                stats['eth0_tx_pkts'] = int(p[9])
    except Exception as e:
        logging.error(f"proc/net/dev error: {e}")
    return stats


def get_tc_drops():
    try:
        out = subprocess.check_output(
            f"docker exec {TARGET_CONTAINER} tc -s qdisc show dev eth0",
            shell=True, text=True, stderr=subprocess.DEVNULL
        )
        for line in out.split('\n'):
            if 'dropped' in line:
                parts = line.strip().split()
                for i, p in enumerate(parts):
                    if p.startswith('(dropped'):
                        return int(parts[i+1].rstrip(','))
    except:
        pass
    return 0


async def fetch_influx():
    data = {}
    try:
        async with InfluxDBClientAsync(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
            q = (f'from(bucket: "{INFLUXDB_BUCKET}") |> range(start: -30s) '
                 f'|> filter(fn: (r) => r["_measurement"] == "ue_info") '
                 f'|> last() '
                 f'|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")')
            tables = await client.query_api().query(q)
            for table in tables:
                for rec in table.records:
                    for field, val in rec.values.items():
                        if field in INFLUX_FIELDS:
                            try:
                                data[field] = float(val) if val is not None and val != 'n/a' else 0.0
                            except:
                                data[field] = 0.0
    except Exception as e:
        logging.error(f"InfluxDB error: {e}")
    return data


async def run():
    global prev_stats, prev_time

    logging.info("=" * 60)
    logging.info("QUICK DIAGNOSTIC: ul_brate root cause")
    logging.info(f"Baseline: {BASELINE_DURATION}s | Stress: {STRESS_DURATION}s @ {STRESS_LOSS_PCT}% loss")
    logging.info(f"Output: {OUTPUT_FILE}")
    logging.info("=" * 60)

    # Start traffic
    start_uplink()
    logging.info("Waiting 10s for traffic to stabilize...")
    await asyncio.sleep(10)

    # CSV setup
    eth_cols = ['eth0_rx_bytes', 'eth0_rx_pkts', 'eth0_tx_bytes', 'eth0_tx_pkts',
                'eth0_rx_Bps', 'eth0_tx_Bps', 'eth0_rx_pps', 'eth0_tx_pps']
    headers = ['timestamp', 'phase', 'loss_pct', 'tc_total_drops'] + eth_cols + INFLUX_FIELDS

    with open(OUTPUT_FILE, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=headers).writeheader()

    state = {
        'prev_stats': get_eth_stats(),
        'prev_time': time.time(),
        'row_count': 0,
    }

    async def collect(phase, loss_pct):
        now = time.time()
        dt = now - state['prev_time']
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        cur = get_eth_stats()
        tc_drops = get_tc_drops()
        influx = await fetch_influx()

        ps = state['prev_stats']
        rx_Bps = (cur.get('eth0_rx_bytes', 0) - ps.get('eth0_rx_bytes', 0)) / dt if dt > 0 else 0
        tx_Bps = (cur.get('eth0_tx_bytes', 0) - ps.get('eth0_tx_bytes', 0)) / dt if dt > 0 else 0
        rx_pps = (cur.get('eth0_rx_pkts', 0) - ps.get('eth0_rx_pkts', 0)) / dt if dt > 0 else 0
        tx_pps = (cur.get('eth0_tx_pkts', 0) - ps.get('eth0_tx_pkts', 0)) / dt if dt > 0 else 0

        row = {
            'timestamp': ts, 'phase': phase, 'loss_pct': f"{loss_pct:.2f}",
            'tc_total_drops': tc_drops,
            'eth0_rx_bytes': cur.get('eth0_rx_bytes', 0),
            'eth0_rx_pkts': cur.get('eth0_rx_pkts', 0),
            'eth0_tx_bytes': cur.get('eth0_tx_bytes', 0),
            'eth0_tx_pkts': cur.get('eth0_tx_pkts', 0),
            'eth0_rx_Bps': f"{rx_Bps:.0f}",
            'eth0_tx_Bps': f"{tx_Bps:.0f}",
            'eth0_rx_pps': f"{rx_pps:.0f}",
            'eth0_tx_pps': f"{tx_pps:.0f}",
        }
        for fld in INFLUX_FIELDS:
            row[fld] = influx.get(fld, 0.0)

        with open(OUTPUT_FILE, 'a', newline='') as fh:
            csv.DictWriter(fh, fieldnames=headers, extrasaction='ignore').writerow(row)

        state['prev_stats'] = cur
        state['prev_time'] = now
        state['row_count'] += 1
        rc = state['row_count']

        if rc % 30 == 0:
            ul = influx.get('ul_brate', 0)
            bsr = influx.get('bsr', 0)
            mcs = influx.get('ul_mcs', 0)
            nok = influx.get('ul_nof_nok', 0)
            logging.info(
                f"  [{phase}] row={rc} | ul_brate={ul:.0f} bsr={bsr:.0f} "
                f"ul_mcs={mcs:.0f} ul_nof_nok={nok:.0f} | "
                f"eth0_tx={tx_Bps/1e6:.2f}MB/s tc_drops={tc_drops}"
            )

        wait = max(0, FETCH_INTERVAL - (time.time() - now))
        await asyncio.sleep(wait)

    # PHASE 1: BASELINE
    logging.info(f"\n--- PHASE 1: BASELINE (0% loss, {BASELINE_DURATION}s) ---")
    cleanup_stress()
    end = time.time() + BASELINE_DURATION
    while time.time() < end:
        await collect("baseline", 0.0)
    logging.info(f"Baseline done. {state['row_count']} rows.")

    # PHASE 2: STRESS
    logging.info(f"\n--- PHASE 2: STRESS ({STRESS_LOSS_PCT}% loss, {STRESS_DURATION}s) ---")
    apply_loss(STRESS_LOSS_PCT)
    stress_start = state['row_count']
    end = time.time() + STRESS_DURATION
    while time.time() < end:
        await collect("stress", STRESS_LOSS_PCT)
    logging.info(f"Stress done. {state['row_count'] - stress_start} stress rows.")

    # Cleanup
    cleanup_stress()
    kill_iperf()

    logging.info(f"\n{'='*60}")
    logging.info(f"DONE. {state['row_count']} total rows → {OUTPUT_FILE}")
    logging.info("=" * 60)


if __name__ == "__main__":
    cleanup_stress()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        cleanup_and_exit()
    finally:
        cleanup_and_exit()
