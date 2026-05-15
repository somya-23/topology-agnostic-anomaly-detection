#!/usr/bin/env python3
"""
HARQ vs Packet-Loss Experiment
===============================
Goal: Prove that packet loss on srsdu0 → HARQ retransmissions ↑ → bitrate ↓

Design:
  1. Start constant 10 Mbps UDP uplink via iperf (same as const_traffic.py)
  2. Collect 3 metric groups every 1s from InfluxDB + Prometheus:
     - Memory usage of srsdu0
     - UL/DL bitrate of the UE on DU0
     - HARQ retransmission counts & delays
  3. Phase 1 (baseline): 10 minutes, no packet loss
  4. Phase 2 (ramp):     Step packet loss on srsdu0 from 0.1% → 3.0%
                          Each step lasts 2 minutes, increment 0.1%
  5. Save everything to CSV for plotting

Usage:
  python3 harq_packet_loss_experiment.py
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

PROMETHEUS_URL = "http://localhost:9090"
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "605bc59413b7d5457d181ccf20f9fda15693f81b068d70396cc183081b264f3b"
INFLUXDB_ORG = "srs"
INFLUXDB_BUCKET = "srsran"

BASELINE_DURATION = 600     # 10 minutes baseline
STEP_DURATION = 120         # 2 minutes per packet-loss step
LOSS_START = 0.1            # start at 0.1%
LOSS_END = 2.0              # end at 3.0%
LOSS_INCREMENT = 0.1        # step size
FETCH_INTERVAL = 1.0        # collect every 1 second

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE_DIR, "experimental_phase0")
DATA_DIR = os.path.join(EXP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, "harq_packet_loss_experiment_m_plane.csv")

# Logging
log_file = os.path.join(DATA_DIR, "harq_experiment.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# =====================================================================
# INFLUXDB METRICS - HARQ + Bitrate fields for DU0's UE
# =====================================================================
# These are the InfluxDB fields we care about (from ue_info measurement).
# We collect ALL fields dynamically via pivot, then filter to these groups.
HARQ_FIELDS = [
    'nof_pucch_f0f1_invalid_harqs',
    'nof_pucch_f2f3f4_invalid_harqs',
    'nof_pusch_invalid_harqs',
    'avg_pucch_harq_delay',
    'avg_pusch_harq_delay',
    'max_pucch_harq_delay',
    'max_pusch_harq_delay',
]
BITRATE_FIELDS = [
    'ul_brate',
    'dl_brate',
]
# We collect all fields but tag the ones we care about for easy plotting

# =====================================================================
# UTILITIES
# =====================================================================
active_iperf_procs = []

def get_container_full_id(name: str) -> str:
    try:
        return subprocess.check_output(
            f"docker inspect --format '{{{{.Id}}}}' {name}",
            shell=True, stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        return ""

def cleanup_stress():
    """Remove any tc netem rules from target container."""
    subprocess.run(
        f"docker exec {TARGET_CONTAINER} tc qdisc del dev eth0 root",
        shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )

def apply_packet_loss(loss_pct: float):
    """Apply packet loss via tc netem on target container."""
    cmd = f"docker exec {TARGET_CONTAINER} tc qdisc replace dev eth0 root netem loss {loss_pct:.2f}%"
    result = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    if result.returncode != 0:
        logging.error(f"Failed to apply {loss_pct}% loss: {result.stderr.decode()}")
    else:
        logging.info(f"Applied {loss_pct:.1f}% packet loss on {TARGET_CONTAINER}")

def cleanup_and_exit(signum=None, frame=None):
    logging.info("Cleaning up...")
    cleanup_stress()
    # Kill iperf processes
    for p in active_iperf_procs:
        try:
            p.terminate()
            p.wait(timeout=1)
        except:
            pass
    # Kill iperf inside containers
    for cname in [IPERF_SERVER_CONTAINER]:
        subprocess.run(f"docker exec {cname} pkill -9 iperf", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Kill iperf in all UE containers
    try:
        ue_ids = subprocess.check_output(
            "docker ps --filter 'name=srsue' --format '{{.ID}}'",
            shell=True, text=True
        ).strip().split('\n')
        for uid in ue_ids:
            if uid:
                subprocess.run(f"docker exec {uid} pkill -9 iperf", shell=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass
    logging.info("Cleanup done.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)

# =====================================================================
# TRAFFIC GENERATION (constant iperf UDP)
# =====================================================================
def start_iperf_traffic():
    """Start constant UDP uplink traffic from all UEs, same as const_traffic.py."""
    # Start server
    subprocess.run(
        f"docker exec {IPERF_SERVER_CONTAINER} pkill -9 iperf",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(1)
    subprocess.run(
        f"docker exec -d {IPERF_SERVER_CONTAINER} iperf -s -u -B {IPERF_TARGET_IP}",
        shell=True
    )
    logging.info(f"iperf server started on {IPERF_SERVER_CONTAINER}")
    time.sleep(2)

    # Start clients on all UEs
    try:
        ue_ids = subprocess.check_output(
            "docker ps --filter 'name=srsue' --format '{{.ID}}'",
            shell=True, text=True
        ).strip().split('\n')
    except:
        ue_ids = []

    for uid in ue_ids:
        if not uid:
            continue
        cmd = ["docker", "exec", uid, "iperf", "-c", IPERF_TARGET_IP,
               "-u", "-b", f"{TARGET_MBPS}M", "-t", "86400"]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        active_iperf_procs.append(p)
        logging.info(f"Started {TARGET_MBPS} Mbps UDP uplink on UE {uid[:12]}")

    logging.info(f"Total UE streams: {len(active_iperf_procs)}")

# =====================================================================
# METRIC COLLECTION
# =====================================================================
async def fetch_influx_harq_bitrate() -> dict:
    """Fetch HARQ + bitrate metrics from InfluxDB for UE on DU0."""
    data = {}
    try:
        async with InfluxDBClientAsync(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
            query_api = client.query_api()
            # Get all fields via pivot - same approach as scrapper.py
            flux_query = (
                f'from(bucket: "{INFLUXDB_BUCKET}") |> range(start: -30s) '
                f'|> filter(fn: (r) => r["_measurement"] == "ue_info") '
                f'|> last() '
                f'|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
            )
            tables = await query_api.query(flux_query)
            for table in tables:
                for record in table.records:
                    pci = record.values.get("pci")
                    rnti = record.values.get("rnti")
                    for field, value in record.values.items():
                        if field in ['_start', '_stop', '_time', '_measurement', 'rnti', 'pci', 'testbed', 'result', 'table']:
                            continue
                        # Only keep HARQ and bitrate fields to keep CSV manageable
                        if field in HARQ_FIELDS or field in BITRATE_FIELDS:
                            key = f"PCI-{pci}_RNTI-{rnti}_{field}"
                            try:
                                data[key] = float(value) if value is not None and value != 'n/a' else 0.0
                            except:
                                data[key] = 0.0
    except Exception as e:
        logging.error(f"InfluxDB fetch error: {e}")
    return data

async def fetch_du0_memory() -> float:
    """Fetch srsdu0 memory usage from Prometheus (cAdvisor)."""
    cid = get_container_full_id(TARGET_CONTAINER)
    if not cid:
        # Fallback to name-based query
        query = f'container_memory_usage_bytes{{name="{TARGET_CONTAINER}"}}-container_memory_cache{{name="{TARGET_CONTAINER}"}}'
    else:
        query = f'container_memory_usage_bytes{{id=~".*{cid}.*"}}-container_memory_cache{{id=~".*{cid}.*"}}'

    try:
        timeout = ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}) as resp:
                result = await resp.json()
                if result.get('status') == 'success' and result.get('data', {}).get('result'):
                    return float(result['data']['result'][0]['value'][1])
    except Exception as e:
        logging.error(f"Prometheus fetch error: {e}")
    return 0.0

# =====================================================================
# MAIN EXPERIMENT LOOP
# =====================================================================
async def run_experiment():
    logging.info("=" * 60)
    logging.info("HARQ vs PACKET-LOSS EXPERIMENT")
    logging.info("=" * 60)
    logging.info(f"Target container:  {TARGET_CONTAINER}")
    logging.info(f"Baseline duration: {BASELINE_DURATION}s ({BASELINE_DURATION//60} min)")
    logging.info(f"Loss range:        {LOSS_START}% → {LOSS_END}% (step {LOSS_INCREMENT}%)")
    logging.info(f"Step duration:     {STEP_DURATION}s ({STEP_DURATION//60} min)")
    logging.info(f"Output:            {OUTPUT_FILE}")
    logging.info("=" * 60)

    # Start constant traffic
    start_iperf_traffic()
    logging.info("Waiting 10s for traffic to stabilize...")
    await asyncio.sleep(10)

    # Discover CSV headers from first fetch
    logging.info("Discovering InfluxDB schema...")
    init_data = await fetch_influx_harq_bitrate()
    influx_headers = sorted(init_data.keys())
    logging.info(f"Found {len(influx_headers)} InfluxDB fields: {influx_headers}")

    headers = ["Timestamp", "phase", "packet_loss_pct", "srsdu0_memory_bytes"] + influx_headers

    # Write CSV header
    with open(OUTPUT_FILE, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=headers).writeheader()

    row_count = 0

    async def collect_one_row(phase: str, loss_pct: float):
        nonlocal row_count
        start = time.time()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Parallel fetch
        influx_task = fetch_influx_harq_bitrate()
        mem_task = fetch_du0_memory()
        influx_data, mem_bytes = await asyncio.gather(influx_task, mem_task)

        row = {
            "Timestamp": ts,
            "phase": phase,
            "packet_loss_pct": f"{loss_pct:.2f}",
            "srsdu0_memory_bytes": mem_bytes,
        }
        for h in influx_headers:
            row[h] = influx_data.get(h, 0.0)

        with open(OUTPUT_FILE, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=headers, extrasaction='ignore').writerow(row)

        row_count += 1
        elapsed = time.time() - start
        await asyncio.sleep(max(0, FETCH_INTERVAL - elapsed))

    # ---- PHASE 1: BASELINE (no packet loss) ----
    logging.info(f"\n{'='*60}")
    logging.info(f"PHASE 1: BASELINE — collecting for {BASELINE_DURATION//60} minutes")
    logging.info(f"{'='*60}")
    cleanup_stress()  # ensure clean state

    baseline_end = time.time() + BASELINE_DURATION
    while time.time() < baseline_end:
        await collect_one_row("baseline", 0.0)
        if row_count % 60 == 0:
            logging.info(f"  Baseline rows: {row_count}, remaining: {int(baseline_end - time.time())}s")

    logging.info(f"Baseline complete. {row_count} rows collected.")

    # ---- PHASE 2: RAMP PACKET LOSS ----
    logging.info(f"\n{'='*60}")
    logging.info(f"PHASE 2: RAMPING PACKET LOSS {LOSS_START}% → {LOSS_END}%")
    logging.info(f"{'='*60}")

    loss_pct = LOSS_START
    while loss_pct <= LOSS_END + 0.001:  # float tolerance
        apply_packet_loss(loss_pct)
        step_end = time.time() + STEP_DURATION
        step_rows = 0

        while time.time() < step_end:
            await collect_one_row("stress", loss_pct)
            step_rows += 1
            if step_rows % 30 == 0:
                logging.info(f"  Loss={loss_pct:.1f}% | step rows={step_rows} | total={row_count}")

        logging.info(f"Step {loss_pct:.1f}% complete — {step_rows} rows")
        loss_pct = round(loss_pct + LOSS_INCREMENT, 1)

    # ---- CLEANUP ----
    logging.info(f"\n{'='*60}")
    logging.info("EXPERIMENT COMPLETE")
    logging.info(f"{'='*60}")
    cleanup_stress()
    logging.info(f"Total rows: {row_count}")
    logging.info(f"Output: {OUTPUT_FILE}")
    logging.info("Next: plot packet_loss_pct vs HARQ counts vs bitrate to confirm correlation")

if __name__ == "__main__":
    cleanup_stress()
    try:
        asyncio.run(run_experiment())
    except KeyboardInterrupt:
        cleanup_and_exit()
    finally:
        cleanup_and_exit()
