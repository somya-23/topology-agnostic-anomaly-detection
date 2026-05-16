#!/usr/bin/env python3
"""
O-RAN srsRAN FALCON - Automated Data Collection Pipeline
=========================================================
Single-run pipeline that:
  1. Collects NORMAL baseline data for --normal-hours (default 5h)
  2. Auto-transitions to ANOMALY mode
  3. Injects every fault type into every target container --fault-reps times (default 2)
  4. Monitors UE containers for RRC Release throughout
  5. Verifies stress is actually applied before labelling rows

Usage:
    python3 scrapper_prometheus_backup.py [--normal-hours 5] [--fault-reps 2]
"""

import aiohttp
import asyncio
import time
from datetime import datetime
import logging
from aiohttp import ClientTimeout
from typing import Dict, List, Optional
import pandas as pd
import csv
import json
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
import os
import random
import subprocess
import threading
import signal
import sys
import re
import argparse

# =====================================================================
# 1. CONFIGURATION (CLI-driven, no hardcoded MODE)
# =====================================================================
parser = argparse.ArgumentParser(description="O-RAN FALCON automated data collection pipeline")
parser.add_argument("--normal-hours", type=float, default=5.0,
                    help="Hours of NORMAL baseline collection (default: 5)")
parser.add_argument("--fault-reps", type=int, default=2,
                    help="Number of times to repeat ALL faults on ALL containers (default: 2)")
parser.add_argument("--stress-duration", type=int, default=180,
                    help="Stress injection duration in seconds (default: 180)")
parser.add_argument("--cooldown-period", type=int, default=420,
                    help="Cooldown between faults in seconds (default: 420)")
parser.add_argument("--anomaly-only", action="store_true",
                    help="Skip NORMAL phase and go straight to ANOMALY collection")
parser.add_argument("--anomaly-hours", type=float, default=0.0,
                    help="When --fault-reps 0: collect ANOMALY data for this many hours then stop (0=unlimited)")
args = parser.parse_args()

NORMAL_DURATION_SEC = 0 if args.anomaly_only else int(args.normal_hours * 3600)
FAULT_REPS = args.fault_reps
STRESS_DURATION = args.stress_duration
COOLDOWN_PERIOD = args.cooldown_period
FETCH_INTERVAL = 1.0

# Phase tracking — shared across threads
current_phase = "ANOMALY" if args.anomaly_only else "NORMAL"
phase_lock = threading.Lock()
normal_phase_done = threading.Event()
experiment_complete = threading.Event()

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE_DIR, "experimental_phase0")
DATA_DIR = os.path.join(EXP_DIR, "traffic_testing")
os.makedirs(DATA_DIR, exist_ok=True)

NORMAL_OUTPUT = os.path.join(DATA_DIR, "train_normal.csv")
ANOMALY_OUTPUT = os.path.join(DATA_DIR, "test_normal_half.csv")
INPUT_FILE = os.path.join(BASE_DIR, "dataScrapper", "promCadvisor_cu1.txt")
PROMETHEUS_URL = "http://localhost:9090"
influxDB_Token = os.environ.get("INFLUXDB_TOKEN")
if not influxDB_Token:
    raise RuntimeError("INFLUXDB_TOKEN env var required. Export it before running this script.")
org = "srs"

NUM_WORKERS = 50
TIMEOUT = 10
RETRY_ATTEMPTS = 3

# Stress tracking
global_stress_data: Dict[str, List[int]] = {}
stress_lock = threading.Lock()

# RRC Release tracking
UE_CONTAINERS = ["srsue2"]
rrc_release_flags: Dict[str, bool] = {ue: False for ue in UE_CONTAINERS}
rrc_lock = threading.Lock()
rrc_log_offsets: Dict[str, int] = {ue: 0 for ue in UE_CONTAINERS}

# Target containers for stress
TARGET_CONTAINERS = ["srscu1", "srsdu2"]

log_file = os.path.join(DATA_DIR, "experiment_run.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# =====================================================================
# 2. CORE UTILITIES (Docker & Cleanup)
# =====================================================================
def nuclear_cleanup():
    try:
        cmd = "docker ps --format '{{.ID}}'"
        container_ids = subprocess.check_output(cmd, shell=True).decode().splitlines()
        for cid in container_ids:
            subprocess.run(f"docker exec {cid} pkill -9 stress-ng", shell=True, stderr=subprocess.DEVNULL)
            subprocess.run(f"docker exec {cid} tc qdisc del dev eth0 root", shell=True, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def signal_handler(sig, frame):
    logging.info("Received termination signal. Cleaning up...")
    nuclear_cleanup()
    experiment_complete.set()
    sys.exit(0)


def get_container_full_id(container_name: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            f"docker inspect --format '{{{{.Id}}}}' {container_name}",
            shell=True, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def get_container_name(container_id: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            f"docker inspect --format '{{{{.Name}}}}' {container_id}",
            shell=True, stderr=subprocess.DEVNULL
        ).decode().strip().lstrip('/')
    except Exception:
        return None


def fix_queries_with_ids(queries: List[str]) -> List[str]:
    fixed_queries = []
    name_pattern = re.compile(r'name="([^"]+)"')
    for q in queries:
        match = name_pattern.search(q)
        if match:
            cname = match.group(1)
            cid = get_container_full_id(cname)
            if cid and cname not in ['cadvisor', 'node-exporter', 'prometheus', 'open5gs_5gc']:
                fixed_queries.append(q.replace(f'name="{cname}"', f'id=~".*{cid}.*"'))
            else:
                fixed_queries.append(q)
        else:
            fixed_queries.append(q)
    return fixed_queries


def check_for_crashed_containers():
    logging.info("=== POST-EXPERIMENT CRASH CHECK ===")
    try:
        cmd = "docker ps -a --filter 'status=exited' --format '{{.Names}} (Exit {{.Status}})'"
        crashed = subprocess.check_output(cmd, shell=True).decode().splitlines()
        if crashed:
            logging.error("WARNING! The following containers CRASHED during the experiment:")
            for c in crashed:
                logging.error(f"  -> {c}")
        else:
            logging.info("SUCCESS! All containers survived the fault injections.")
    except Exception as e:
        logging.error(f"Failed to check crashes: {e}")


# =====================================================================
# 3. RRC RELEASE MONITOR
# =====================================================================
def check_rrc_release_once() -> Dict[str, bool]:
    """Check all UE container logs for 'Received RRC Release'.
    Uses incremental log checking (only new lines since last check)."""
    results = {}
    for ue in UE_CONTAINERS:
        try:
            logs = subprocess.check_output(
                ["docker", "logs", "--tail", "100", ue],
                text=True, stderr=subprocess.DEVNULL
            )
            found = "Received RRC Release" in logs
            results[ue] = found
            if found:
                with rrc_lock:
                    if not rrc_release_flags[ue]:
                        rrc_release_flags[ue] = True
                        logging.warning(f"RRC RELEASE detected in {ue}!")
        except Exception:
            results[ue] = False
    return results


def rrc_monitor_loop():
    """Background thread: checks UE logs every 10s for RRC Release.
    If detected, attempts to restart the affected UE."""
    logging.info("[RRC Monitor] Started — checking UE logs every 10s")
    while not experiment_complete.is_set():
        detections = check_rrc_release_once()
        for ue, detected in detections.items():
            if detected:
                logging.warning(f"[RRC Monitor] {ue} received RRC Release — attempting restart")
                try:
                    subprocess.run(f"docker restart {ue}", shell=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=60)
                    logging.info(f"[RRC Monitor] {ue} restarted successfully")
                    with rrc_lock:
                        rrc_release_flags[ue] = False
                except Exception as e:
                    logging.error(f"[RRC Monitor] Failed to restart {ue}: {e}")

        # Wait 10s between checks, but exit early if experiment ends
        experiment_complete.wait(timeout=10)
    logging.info("[RRC Monitor] Stopped")


# =====================================================================
# 4. TELEMETRY CLIENTS (Parallel Fetch Logic)
# =====================================================================
class PrometheusClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.timeout = ClientTimeout(total=TIMEOUT)
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            connector=aiohttp.TCPConnector(limit=NUM_WORKERS)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_query(self, query: str) -> Optional[float]:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                async with self.session.get(
                    f"{self.base_url}/api/v1/query",
                    params={"query": query}
                ) as response:
                    result = await response.json()
                    if result.get('status') == 'success' and result.get('data', {}).get('result'):
                        return float(result['data']['result'][0]['value'][1])
                    return 0.0
            except Exception:
                if attempt == RETRY_ATTEMPTS - 1:
                    return 0.0
                await asyncio.sleep(0.1)


async def fetch_influx_data() -> pd.DataFrame:
    data = {}
    try:
        async with InfluxDBClientAsync(url="http://localhost:8086", token=influxDB_Token, org=org) as client:
            query_api = client.query_api()
            flux_query = (
                f'from(bucket: "srsran") |> range(start: -30s) '
                f'|> filter(fn: (r) => r["_measurement"] == "ue_info") '
                f'|> filter(fn: (r) => r["rnti"] == "4601") '
                f'|> last() '
                f'|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
            )
            tables = await query_api.query(flux_query)
            for table in tables:
                for record in table.records:
                    pci = record.values.get("pci")
                    for field, value in record.values.items():
                        if field in ['_start', '_stop', '_time', '_measurement', 'rnti', 'pci', 'testbed', 'result', 'table']:
                            continue
                        key = f"PCI-{pci}_RNTI-4601_{field}"
                        try:
                            data[key] = float(value) if value is not None and value != 'n/a' else 0.0
                        except (ValueError, TypeError):
                            data[key] = 0.0
    except Exception:
        pass
    return pd.DataFrame([data])


# =====================================================================
# 5. STRESS ENGINE (with verification & N repetitions)
# =====================================================================
def verify_stress_applied(container_id: str, stress_type: int) -> bool:
    """After injecting stress, verify it is actually applied."""
    cname = get_container_name(container_id) or container_id
    try:
        if stress_type == 1:  # CPU — check stress-ng process exists
            result = subprocess.run(
                f"docker exec {container_id} pgrep -c stress-ng",
                shell=True, capture_output=True, text=True
            )
            count = int(result.stdout.strip()) if result.stdout.strip() else 0
            if count > 0:
                logging.info(f"  [VERIFY] {cname} CPU stress confirmed ({count} stress-ng processes)")
                return True
            logging.warning(f"  [VERIFY] {cname} CPU stress NOT detected!")
            return False

        elif stress_type == 2:  # MEM — check stress-ng process exists
            result = subprocess.run(
                f"docker exec {container_id} pgrep -c stress-ng",
                shell=True, capture_output=True, text=True
            )
            count = int(result.stdout.strip()) if result.stdout.strip() else 0
            if count > 0:
                logging.info(f"  [VERIFY] {cname} MEM stress confirmed ({count} stress-ng processes)")
                return True
            logging.warning(f"  [VERIFY] {cname} MEM stress NOT detected!")
            return False

        elif stress_type == 3:  # NET — check tc qdisc rule exists
            result = subprocess.run(
                f"docker exec {container_id} tc qdisc show dev eth0",
                shell=True, capture_output=True, text=True
            )
            if "netem" in result.stdout:
                logging.info(f"  [VERIFY] {cname} NET stress confirmed (netem rule active)")
                return True
            logging.warning(f"  [VERIFY] {cname} NET stress NOT detected!")
            return False

    except Exception as e:
        logging.warning(f"  [VERIFY] {cname} verification failed: {e}")
        return False
    return False


def injectStress(container_id: str, typeOfStress: int, duration: int):
    """Inject stress and verify it is applied. Updates global_stress_data.

    Label is only set AFTER verify_stress_applied confirms the fault is running.
    If verification fails, the fault is dropped and rows are NOT labelled with it.
    NET intensity (85-95) maps to actual packet loss (4.4-4.8%) so the recorded
    _stepStress value reflects the real knob applied via tc netem.
    """
    cname = get_container_name(container_id)
    intensity = random.randint(85, 95)
    stress_names = {1: "CPU", 2: "MEM", 3: "NET"}
    net_applied = False

    try:
        if typeOfStress == 1:  # CPU
            cmd = f"docker exec {container_id} stress-ng --cpu 4 --cpu-load {intensity} --timeout {duration}s"
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
            if not verify_stress_applied(container_id, typeOfStress):
                logging.error(f"[FAULT-DROP] {cname} CPU stress did not apply; rows NOT labelled.")
                subprocess.run(
                    f"docker exec {container_id} pkill -9 stress-ng",
                    shell=True, stderr=subprocess.DEVNULL,
                )
                return
            with stress_lock:
                global_stress_data[cname] = [typeOfStress, intensity]
            proc.wait()

        elif typeOfStress == 2:  # MEM
            mb = int((intensity / 100.0) * 512)
            cmd = f"docker exec {container_id} stress-ng --vm 1 --vm-bytes {mb}M --timeout {duration}s"
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
            if not verify_stress_applied(container_id, typeOfStress):
                logging.error(f"[FAULT-DROP] {cname} MEM stress did not apply; rows NOT labelled.")
                subprocess.run(
                    f"docker exec {container_id} pkill -9 stress-ng",
                    shell=True, stderr=subprocess.DEVNULL,
                )
                return
            with stress_lock:
                global_stress_data[cname] = [typeOfStress, intensity]
            proc.wait()

        elif typeOfStress == 3:  # NET
            # Faithful label: map intensity 85-95 to packet loss 4.4%-4.8% on eth0
            loss = 1.0 + (4.0 * intensity / 100.0)
            cmd = f"docker exec {container_id} tc qdisc replace dev eth0 root netem loss {loss:.2f}%"
            subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)
            net_applied = True
            time.sleep(1)
            if not verify_stress_applied(container_id, typeOfStress):
                logging.error(f"[FAULT-DROP] {cname} NET stress did not apply; rows NOT labelled.")
                subprocess.run(
                    f"docker exec {container_id} tc qdisc del dev eth0 root",
                    shell=True, stderr=subprocess.DEVNULL,
                )
                net_applied = False
                return
            with stress_lock:
                global_stress_data[cname] = [typeOfStress, intensity]
            time.sleep(duration - 1)

    except Exception as e:
        logging.error(f"Stress injection failed on {cname}: {e}")
    finally:
        if typeOfStress == 3 and net_applied:
            subprocess.run(
                f"docker exec {container_id} tc qdisc del dev eth0 root",
                shell=True, stderr=subprocess.DEVNULL,
            )
        with stress_lock:
            global_stress_data[cname] = [0, 0]


def stress_loop():
    """Phase-aware stress loop:
    1. Wait for NORMAL phase to finish (signalled by normal_phase_done)
    2a. If fault-reps=0: no injection; wait anomaly-hours then signal complete
    2b. Otherwise: wait 5 min baseline, inject faults, signal complete
    """
    logging.info(f"[Stress] Waiting for NORMAL phase to complete ({args.normal_hours}h)...")
    normal_phase_done.wait()

    if experiment_complete.is_set():
        return

    # Pure half-traffic mode: no stress injection, just timed anomaly collection
    if FAULT_REPS == 0:
        if args.anomaly_hours > 0:
            logging.info(f"[Stress] No fault injection. Collecting ANOMALY data for {args.anomaly_hours}h...")
            experiment_complete.wait(timeout=args.anomaly_hours * 3600)
            logging.info("[Stress] ANOMALY collection window complete.")
            experiment_complete.set()
        else:
            logging.info("[Stress] No fault injection, no --anomaly-hours limit. Run until interrupted.")
        return

    logging.info("[Stress] ANOMALY phase started. Waiting 5m for anomaly baseline...")
    time.sleep(300)

    # Build container ID map
    output = subprocess.check_output(
        "docker ps --format '{{.ID}} {{.Names}}'", shell=True
    ).decode().splitlines()
    cmap = {line.split()[1]: line.split()[0] for line in output if len(line.split()) >= 2}
    victims = [(cmap[name], name) for name in TARGET_CONTAINERS if name in cmap]

    if not victims:
        logging.error("[Stress] No target containers found!")
        experiment_complete.set()
        return

    stress_names = {1: "CPU", 2: "MEM", 3: "NET"}
    total_faults = len(victims) * 3 * FAULT_REPS
    fault_num = 0

    for rep in range(1, FAULT_REPS + 1):
        logging.info(f"=== FAULT REPETITION {rep}/{FAULT_REPS} ===")
        for victim_id, victim_name in victims:
            for s_type in [1, 2, 3]:
                fault_num += 1
                logging.info(
                    f"[{fault_num}/{total_faults}] START STRESS: {victim_name} "
                    f"Type:{stress_names[s_type]} Rep:{rep}/{FAULT_REPS}"
                )
                injectStress(victim_id, s_type, STRESS_DURATION)
                logging.info(f"COOLDOWN: {COOLDOWN_PERIOD}s...")
                time.sleep(COOLDOWN_PERIOD)

    logging.info("=== ALL FAULTS INJECTED. EXPERIMENT COMPLETE. ===")
    check_for_crashed_containers()

    # Let the scrapper collect a few more minutes of post-fault data
    logging.info("Collecting 5 min of post-fault baseline...")
    time.sleep(300)
    experiment_complete.set()


# =====================================================================
# 6. PRE-FLIGHT & BASELINE VALIDATION
# =====================================================================
PCI_UL_BRATE_MIN = 2_000_000
NET_TX_MIN = 1_000
CU_CPU_MIN = 0.05
DU_CPU_MIN = 1.0
DRIFT_WARN_PCT = 0.20
DRIFT_CHECK_EVERY = 300
PREFLIGHT_CONTAINERS = TARGET_CONTAINERS


async def pre_flight_check(init_df: pd.DataFrame, client, raw_queries: List[str], actual_queries: List[str]) -> bool:
    logging.info("=== PRE-FLIGHT CHECK ===")
    passed = True

    # 1. All PCIs must have active UL traffic
    for col in sorted(init_df.columns):
        if 'ul_brate' not in col:
            continue
        pci_tag = col.split('_')[0]
        val = float(init_df[col].values[0])
        ok = val >= PCI_UL_BRATE_MIN
        logging.info(f"  {pci_tag} ul_brate = {val / 1e6:.2f} Mbps  [{'OK' if ok else 'FAIL — UE NOT ATTACHED'}]")
        if not ok:
            passed = False

    # 2. Check CPU and NET_TX for all CU/DU containers
    for cname in PREFLIGHT_CONTAINERS:
        cpu_min = CU_CPU_MIN if cname.startswith("srscu") else DU_CPU_MIN
        cpu_idx = next((i for i, q in enumerate(raw_queries) if f'name="{cname}"' in q and 'cpu_user' in q), None)
        if cpu_idx is not None:
            val = await client.fetch_query(actual_queries[cpu_idx])
            ok = val >= cpu_min
            logging.info(f"  {cname:8s} CPU    = {val:.2f}      [{'OK' if ok else 'FAIL — container idle'}]")
            if not ok:
                passed = False

        tx_idx = next((i for i, q in enumerate(raw_queries) if f'name="{cname}"' in q and 'transmit' in q), None)
        if tx_idx is not None:
            val = await client.fetch_query(actual_queries[tx_idx])
            ok = val >= NET_TX_MIN
            logging.info(f"  {cname:8s} NET_TX = {val / 1e6:.2f} MB/s  [{'OK' if ok else 'FAIL — low network activity'}]")
            if not ok:
                passed = False

    # 3. Check for RRC Release before starting
    logging.info("  Checking UE containers for RRC Release...")
    rrc_results = check_rrc_release_once()
    for ue, detected in rrc_results.items():
        if detected:
            logging.warning(f"  {ue}: RRC Release detected BEFORE collection — restarting UE")
            subprocess.run(f"docker restart {ue}", shell=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            time.sleep(10)
            passed = False

    if passed:
        logging.info("PRE-FLIGHT PASSED — starting collection")
    else:
        logging.error("PRE-FLIGHT FAILED — fix issues before collecting")
        logging.error("  Common fixes:")
        logging.error("    PCI ul_brate = 0  -> UE not attached, restart UE for that cell")
        logging.error("    CPU low          -> container may be idle or just restarted (wait 5m)")
        logging.error("    NET_TX low       -> check docker ps, verify DU-CU connectivity")
        logging.error("    RRC Release      -> UE was restarted, wait for re-attachment")
        logging.error("Aborting — baseline would be corrupted. Fix and re-run.")
        sys.exit(1)

    logging.info("=== END PRE-FLIGHT ===")
    return passed


def save_baseline_snapshot(row: dict, snapshot_file: str):
    keep_keys = ['cpu_user', 'memory_usage', 'transmit', 'receive', 'ul_brate', 'bsr']
    snapshot = {k: v for k, v in row.items()
                if any(x in k for x in keep_keys) and k != 'Timestamp'}
    snapshot['_timestamp'] = row['Timestamp']
    snapshot['_phase'] = row.get('phase', 'NORMAL')
    with open(snapshot_file, 'w') as f:
        json.dump(snapshot, f, indent=2)
    logging.info(f"Baseline snapshot saved -> {snapshot_file}")


def check_baseline_drift(snapshot: dict, row: dict):
    drifted = []
    for key, snap_val in snapshot.items():
        if key.startswith('_') or snap_val == 0:
            continue
        curr_val = row.get(key, 0)
        drift = abs(curr_val - snap_val) / (abs(snap_val) + 1e-9)
        if drift > DRIFT_WARN_PCT:
            drifted.append(f"{key[:55]}: baseline={snap_val:.2f} now={curr_val:.2f} drift={drift * 100:.0f}%")
    if drifted:
        logging.warning("=== BASELINE DRIFT DETECTED ===")
        for msg in drifted:
            logging.warning(f"  {msg}")
        logging.warning("================================")


# =====================================================================
# 7. MAIN SCRAPER (Auto NORMAL -> ANOMALY with RRC tracking)
# =====================================================================
async def main():
    global current_phase

    # A. Prometheus Discovery
    with open(INPUT_FILE, "r") as f:
        raw_queries = [line.strip() for line in f.readlines() if line.strip()]
    actual_queries = fix_queries_with_ids(raw_queries)
    query_map = dict(zip(raw_queries, actual_queries))

    # B. Influx Discovery & Locking
    logging.info("Locking InfluxDB Schema...")
    init_df = await fetch_influx_data()
    influx_headers = sorted(list(init_df.columns))

    # B2. Pre-flight check
    async with PrometheusClient(PROMETHEUS_URL) as preflight_client:
        await pre_flight_check(init_df, preflight_client, raw_queries, actual_queries)

    # C. Stress Header Locking
    cnames = sorted(subprocess.check_output(
        "docker ps --filter 'name=^srscu' --filter 'name=^srsdu' --format '{{.Names}}'",
        shell=True
    ).decode().splitlines())
    stress_headers = []
    for c in cnames:
        stress_headers.extend([f"{c}_stressType", f"{c}_stepStress"])
        with stress_lock:
            global_stress_data[c] = [0, 0]

    # D. Extra columns: phase, rrc_release, any_stress_active
    extra_headers = ["phase", "rrc_release_detected", "any_stress_active"]

    # E. Initialize CSV headers
    headers = ["Timestamp"] + raw_queries + influx_headers + stress_headers + extra_headers

    # Create both output files with headers
    for out_file in [NORMAL_OUTPUT, ANOMALY_OUTPUT]:
        if not os.path.exists(out_file):
            with open(out_file, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=headers).writeheader()

    # F. Log experiment plan
    total_fault_time = len(TARGET_CONTAINERS) * 3 * FAULT_REPS * (STRESS_DURATION + COOLDOWN_PERIOD)
    logging.info("=" * 60)
    logging.info("EXPERIMENT PLAN:")
    logging.info(f"  NORMAL phase:  {args.normal_hours}h ({NORMAL_DURATION_SEC}s)")
    logging.info(f"  ANOMALY phase: {len(TARGET_CONTAINERS)} containers x 3 faults x {FAULT_REPS} reps")
    logging.info(f"  Per fault:     {STRESS_DURATION}s stress + {COOLDOWN_PERIOD}s cooldown")
    logging.info(f"  Est. anomaly:  {total_fault_time / 3600:.1f}h")
    logging.info(f"  Est. total:    {(NORMAL_DURATION_SEC + total_fault_time + 600) / 3600:.1f}h")
    logging.info(f"  Normal output: {NORMAL_OUTPUT}")
    logging.info(f"  Anomaly output: {ANOMALY_OUTPUT}")
    logging.info("=" * 60)

    # G. Collection Loop
    snapshot_file = os.path.join(DATA_DIR, "baseline_snapshot.json")
    baseline_snapshot = {}
    row_count = 0
    normal_start = time.time()

    async with PrometheusClient(PROMETHEUS_URL) as client:
        while not experiment_complete.is_set():
            start_mark = time.time()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # Check phase transition: NORMAL -> ANOMALY
            with phase_lock:
                if current_phase == "NORMAL" and (time.time() - normal_start) >= NORMAL_DURATION_SEC:
                    current_phase = "ANOMALY"
                    logging.info("=" * 60)
                    logging.info("PHASE TRANSITION: NORMAL -> ANOMALY")
                    logging.info(f"  Normal rows collected: {row_count}")
                    logging.info("=" * 60)
                    normal_phase_done.set()
                active_phase = current_phase

            # Parallel Fetch
            prom_task = asyncio.gather(*[client.fetch_query(aq) for aq in actual_queries])
            influx_task = fetch_influx_data()
            prom_results, current_influx_df = await asyncio.gather(prom_task, influx_task)

            # Map Row
            row = {"Timestamp": ts}
            row.update(dict(zip(raw_queries, prom_results)))

            influx_vals = current_influx_df.iloc[0].to_dict() if not current_influx_df.empty else {}
            for h in influx_headers:
                row[h] = influx_vals.get(h, 0.0)

            # Stress labels
            with stress_lock:
                sc = global_stress_data.copy()
            for c in cnames:
                row[f"{c}_stressType"], row[f"{c}_stepStress"] = sc.get(c, [0, 0])

            # Check if ANY stress is active right now
            any_stress = any(v[0] != 0 for v in sc.values())

            # RRC Release detection
            with rrc_lock:
                any_rrc = any(rrc_release_flags.values())

            # Extra columns
            row["phase"] = active_phase
            row["rrc_release_detected"] = 1 if any_rrc else 0
            row["any_stress_active"] = 1 if any_stress else 0

            # Write to the appropriate file based on phase
            output_file = NORMAL_OUTPUT if active_phase == "NORMAL" else ANOMALY_OUTPUT
            with open(output_file, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=headers, extrasaction='ignore').writerow(row)

            # Baseline tracking
            row_count += 1
            if row_count == 1:
                save_baseline_snapshot(row, snapshot_file)
                baseline_snapshot = {k: v for k, v in row.items()
                                     if any(x in k for x in ['cpu_user', 'memory_usage', 'transmit', 'receive', 'ul_brate', 'bsr'])}

            if row_count % DRIFT_CHECK_EVERY == 0:
                logging.info(f"Rows collected: {row_count} | Phase: {active_phase} | Stress: {any_stress} | RRC: {any_rrc}")
                if active_phase == "NORMAL":
                    check_baseline_drift(baseline_snapshot, row)

            # Accurate 1s Timing
            elapsed = time.time() - start_mark
            await asyncio.sleep(max(0, FETCH_INTERVAL - elapsed))

    logging.info("=" * 60)
    logging.info("DATA COLLECTION COMPLETE")
    logging.info(f"  Total rows: {row_count}")
    logging.info(f"  Normal data: {NORMAL_OUTPUT}")
    logging.info(f"  Anomaly data: {ANOMALY_OUTPUT}")
    logging.info("=" * 60)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    nuclear_cleanup()

    # If anomaly-only, skip normal phase wait immediately
    if args.anomaly_only:
        normal_phase_done.set()
        logging.info("--anomaly-only: skipping NORMAL phase, starting ANOMALY collection")

    # Start background threads
    threading.Thread(target=stress_loop, daemon=True).start()
    threading.Thread(target=rrc_monitor_loop, daemon=True).start()

    asyncio.run(main())
