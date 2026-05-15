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
import restart

# =====================================================================
# 1. CONFIGURATION & TOGGLE
# =====================================================================
MODE = "NORMAL"   # <-- Set to "NORMAL" for Baseline, "ANOMALY" for Faults

# Timing Configuration
STRESS_DURATION = 180  # 3 minutes
COOLDOWN_PERIOD = 420  # 7 minutes
FETCH_INTERVAL = 1.0   # Target 1 second collection

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE_DIR, "experimental_phase0")
DATA_DIR = os.path.join(EXP_DIR, "data_m_plane_separate")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, "train_normal_m_plane.csv" if MODE == "NORMAL" else "test_anomaly_m_plane.csv")
INPUT_FILE = os.path.join(BASE_DIR, "dataScrapper", "promCadvisor.txt")
PROMETHEUS_URL = "http://localhost:9090"
influxDB_Token = "605bc59413b7d5457d181ccf20f9fda15693f81b068d70396cc183081b264f3b"
org = "srs"

NUM_WORKERS = 50
TIMEOUT = 10
RETRY_ATTEMPTS = 3

global_stress_data: Dict[str, List[int]] = {}
stress_lock = threading.Lock()

# This creates a permanent log file AND prints to the terminal
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
    except: pass

def signal_handler(sig, frame):
    nuclear_cleanup()
    sys.exit(0)

def get_container_full_id(container_name: str) -> Optional[str]:
    try: return subprocess.check_output(f"docker inspect --format '{{{{.Id}}}}' {container_name}", shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except: return None

def get_container_name(container_id: str) -> Optional[str]:
    try: return subprocess.check_output(f"docker inspect --format '{{{{.Name}}}}' {container_id}", shell=True, stderr=subprocess.DEVNULL).decode().strip().lstrip('/')
    except: return None

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
            else: fixed_queries.append(q)
        else: fixed_queries.append(q)
    return fixed_queries

def check_for_crashed_containers():
    logging.info("=== POST-EXPERIMENT CRASH CHECK ===")
    try:
        cmd = "docker ps -a --filter 'status=exited' --format '{{.Names}} (Exit {{.Status}})'"
        crashed = subprocess.check_output(cmd, shell=True).decode().splitlines()
        if crashed:
            logging.error(f"WARNING! The following containers CRASHED during the experiment:")
            for c in crashed:
                logging.error(f"  -> {c}")
        else:
            logging.info("SUCCESS! All containers survived the fault injections.")
    except Exception as e:
        logging.error(f"Failed to check crashes: {e}")

# =====================================================================
# 3. TELEMETRY CLIENTS (Parallel Fetch Logic)
# =====================================================================
class PrometheusClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.timeout = ClientTimeout(total=TIMEOUT)
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout, connector=aiohttp.TCPConnector(limit=NUM_WORKERS))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session: await self.session.close()

    async def fetch_query(self, query: str) -> Optional[float]:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                async with self.session.get(f"{self.base_url}/api/v1/query", params={"query": query}) as response:
                    result = await response.json()
                    if result.get('status') == 'success' and result.get('data', {}).get('result'):
                        return float(result['data']['result'][0]['value'][1])
                    return 0.0
            except:
                if attempt == RETRY_ATTEMPTS - 1: return 0.0
                await asyncio.sleep(0.1)

async def fetch_influx_data() -> pd.DataFrame:
    data = {}
    try:
        async with InfluxDBClientAsync(url="http://localhost:8086", token=influxDB_Token, org=org) as client:
            query_api = client.query_api()
            # Dynamic discovery of all 30+ fields using pivot
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
                        try: data[key] = float(value) if value is not None and value != 'n/a' else 0.0
                        except: data[key] = 0.0
    except Exception: pass
    return pd.DataFrame([data])

# =====================================================================
# 4. STRESS ENGINE (3 min on / 7 min off)
# =====================================================================
def injectStress(container_id, typeOfStress, duration):
    cname = get_container_name(container_id)
    intensity = random.randint(85, 95)
    
    try:
        if typeOfStress == 1: # CPU
            cmd = f"docker exec {container_id} stress-ng --cpu 4 --cpu-load {intensity} --timeout {duration}s"
            with stress_lock: global_stress_data[cname] = [typeOfStress, intensity]
            subprocess.run(cmd, shell=True)
            
        elif typeOfStress == 2: # MEM
            mb = int((intensity / 100.0) * 512) # ~435-486 MB, safe for 3GB containers
            cmd = f"docker exec {container_id} stress-ng --vm 1 --vm-bytes {mb}M --timeout {duration}s"
            with stress_lock: global_stress_data[cname] = [typeOfStress, intensity]
            subprocess.run(cmd, shell=True)
            
        elif typeOfStress == 3: # NET
            loss = 2.0
            cmd = f"docker exec {container_id} tc qdisc replace dev eth0 root netem loss {loss:.2f}%"
            with stress_lock: global_stress_data[cname] = [typeOfStress, intensity]
            
            # Apply the rule (returns instantly)
            subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL) 
            
            # Force Python to wait for the network fault duration!
            time.sleep(duration) 
        
    finally:
        if typeOfStress == 3: 
            # Delete the rule after the sleep is over
            subprocess.run(f"docker exec {container_id} tc qdisc del dev eth0 root", shell=True, stderr=subprocess.DEVNULL)
            
        with stress_lock: global_stress_data[cname] = [0, 0]

def stress_loop():
    if MODE == "NORMAL": return
    logging.info("Waiting 5m for baseline...")
    time.sleep(300)
    
    target_containers = ["srscu0", "srsdu0", "srsdu1", "srscu1", "srsdu2", "srscu2","srsdu3","srsdu4", "srsdu5"]
    output = subprocess.check_output("docker ps --format '{{.ID}} {{.Names}}'", shell=True).decode().splitlines()
    cmap = {line.split()[1]: line.split()[0] for line in output if len(line.split()) >= 2}
    victims = [cmap[name] for name in target_containers if name in cmap]

    for victim in victims:
        for s_type in [1, 2, 3]:
            logging.info(f"START STRESS: {get_container_name(victim)} Type:{s_type}")
            injectStress(victim, s_type, STRESS_DURATION)
            logging.info(f"COOLDOWN: 7 minutes...")
            time.sleep(COOLDOWN_PERIOD)
            
    # Auto-check for crashes when the loop finishes!
    logging.info("All faults injected. Experiment Complete.")
    check_for_crashed_containers()

# =====================================================================
# 5. PRE-FLIGHT & BASELINE VALIDATION
# =====================================================================
# Thresholds derived from test_anomaly_final.csv normal rows
PCI_UL_BRATE_MIN    = 5_000_000   # 5 Mbps — confirms UE is attached to that cell
NET_TX_MIN          = 1_000_000   # 1 MB/s — confirms container has active network traffic
CU_CPU_MIN          = 0.05        # CUs handle control plane only, CPU usage is naturally low (~0.1-0.2%)
DU_CPU_MIN          = 1.0         # DUs handle user plane, CPU usage is higher (~2.5-3%)
DRIFT_WARN_PCT      = 0.20        # warn if any key metric drifts >20% from snapshot
DRIFT_CHECK_EVERY   = 300         # check drift every 300 rows (~5 minutes)

# All containers to check in pre-flight
PREFLIGHT_CONTAINERS = ["srscu0", "srscu1", "srscu2", "srsdu0", "srsdu1", "srsdu2", "srsdu3", "srsdu4", "srsdu5"]

async def pre_flight_check(init_df: pd.DataFrame, client, raw_queries: List[str], actual_queries: List[str]) -> bool:
    """
    Runs once at startup before any CSV rows are written.
    Checks all UEs are attached and all DUs are active.
    In NORMAL mode: aborts the process if anything fails.
    In ANOMALY mode: logs a warning but continues (faults may intentionally kill cells).
    """
    logging.info("=== PRE-FLIGHT CHECK ===")
    passed = True

    # 1. All 6 PCIs must have active UL traffic (confirms each UE is attached)
    for col in sorted(init_df.columns):
        if 'ul_brate' not in col:
            continue
        pci_tag = col.split('_')[0]  # e.g. "PCI-1"
        val = float(init_df[col].values[0])
        ok  = val >= PCI_UL_BRATE_MIN
        logging.info(f"  {pci_tag} ul_brate = {val/1e6:.2f} Mbps  [{'OK' if ok else 'FAIL — UE NOT ATTACHED'}]")
        if not ok:
            passed = False

    # 2. Check CPU and NET_TX for ALL CU/DU containers
    for cname in PREFLIGHT_CONTAINERS:
        # CPU check (CUs use much less CPU than DUs — different thresholds)
        cpu_min = CU_CPU_MIN if cname.startswith("srscu") else DU_CPU_MIN
        cpu_idx = next((i for i, q in enumerate(raw_queries) if f'name="{cname}"' in q and 'cpu_user' in q), None)
        if cpu_idx is not None:
            val = await client.fetch_query(actual_queries[cpu_idx])
            ok = val >= cpu_min
            logging.info(f"  {cname:8s} CPU    = {val:.2f}      [{'OK' if ok else 'FAIL — container idle'}]")
            if not ok:
                passed = False

        # NET_TX check
        tx_idx = next((i for i, q in enumerate(raw_queries) if f'name="{cname}"' in q and 'transmit' in q), None)
        if tx_idx is not None:
            val = await client.fetch_query(actual_queries[tx_idx])
            ok = val >= NET_TX_MIN
            logging.info(f"  {cname:8s} NET_TX = {val/1e6:.2f} MB/s  [{'OK' if ok else 'FAIL — low network activity'}]")
            if not ok:
                passed = False

    if passed:
        logging.info("PRE-FLIGHT PASSED — starting collection")
    else:
        logging.error("PRE-FLIGHT FAILED — fix issues before collecting")
        logging.error("  Common fixes:")
        logging.error("    PCI ul_brate = 0  → UE not attached, restart UE for that cell")
        logging.error("    CPU low          → container may be idle or just restarted (wait 5m)")
        logging.error("    NET_TX low       → check docker ps, verify DU-CU connectivity")
        if MODE == "NORMAL":
            logging.error("Aborting NORMAL collection — baseline would be corrupted.")
            sys.exit(1)
        else:
            logging.warning("Continuing ANOMALY collection despite warnings (stress may affect cells).")

    logging.info("=== END PRE-FLIGHT ===")
    return passed


def save_baseline_snapshot(row: dict, snapshot_file: str):
    """
    Saves the first collected row's key metric values to a JSON file.
    On the next run, this file can be compared to detect baseline drift
    before collection starts.
    """
    keep_keys = ['cpu_user', 'memory_usage', 'transmit', 'receive', 'ul_brate', 'bsr']
    snapshot = {k: v for k, v in row.items()
                if any(x in k for x in keep_keys) and k != 'Timestamp'}
    snapshot['_timestamp'] = row['Timestamp']
    snapshot['_mode']      = MODE
    with open(snapshot_file, 'w') as f:
        json.dump(snapshot, f, indent=2)
    logging.info(f"Baseline snapshot saved → {snapshot_file}")


def check_baseline_drift(snapshot: dict, row: dict):
    """
    Called every DRIFT_CHECK_EVERY rows during collection.
    Logs a warning for any key metric that has drifted >20% from the opening snapshot.
    Does NOT abort — just alerts so you can investigate.
    """
    drifted = []
    for key, snap_val in snapshot.items():
        if key.startswith('_') or snap_val == 0:
            continue
        curr_val = row.get(key, 0)
        drift = abs(curr_val - snap_val) / (abs(snap_val) + 1e-9)
        if drift > DRIFT_WARN_PCT:
            drifted.append(f"{key[:55]}: baseline={snap_val:.2f} now={curr_val:.2f} drift={drift*100:.0f}%")
    if drifted:
        logging.warning("=== BASELINE DRIFT DETECTED ===")
        for msg in drifted:
            logging.warning(f"  {msg}")
        logging.warning("================================")


# =====================================================================
# 6. MAIN SCRAPER (Header Locking & Parallel Fetch)
# =====================================================================
async def main():
    # A. Prometheus Discovery
    with open(INPUT_FILE, "r") as f:
        raw_queries = [line.strip() for line in f.readlines() if line.strip()]
    actual_queries = fix_queries_with_ids(raw_queries)
    query_map = dict(zip(raw_queries, actual_queries))

    # B. Influx Discovery & Locking
    logging.info("Locking InfluxDB Schema...")
    init_df = await fetch_influx_data()
    influx_headers = sorted(list(init_df.columns))

    # B2. Pre-flight check — aborts in NORMAL mode if any UE/DU is missing
    async with PrometheusClient(PROMETHEUS_URL) as preflight_client:
        await pre_flight_check(init_df, preflight_client, raw_queries, actual_queries)

    # C. Stress Header Locking
    cnames = sorted(subprocess.check_output("docker ps --filter 'name=^srscu' --filter 'name=^srsdu' --format '{{.Names}}'", shell=True).decode().splitlines())
    stress_headers = []
    for c in cnames:
        stress_headers.extend([f"{c}_stressType", f"{c}_stepStress"])
        with stress_lock: global_stress_data[c] = [0, 0]

    # D. Initialize CSV
    headers = ["Timestamp"] + raw_queries + influx_headers + stress_headers
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()

    # E. Collection Loop
    snapshot_file = os.path.join(DATA_DIR, f"baseline_snapshot_{MODE.lower()}.json")
    baseline_snapshot = {}
    row_count = 0

    async with PrometheusClient(PROMETHEUS_URL) as client:
        while True:
            start_mark = time.time()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # Parallel Fetch
            prom_task = asyncio.gather(*[client.fetch_query(aq) for aq in actual_queries])
            influx_task = fetch_influx_data()
            prom_results, current_influx_df = await asyncio.gather(prom_task, influx_task)

            # Map Row (Locked Structure)
            row = {"Timestamp": ts}
            row.update(dict(zip(raw_queries, prom_results)))

            influx_vals = current_influx_df.iloc[0].to_dict() if not current_influx_df.empty else {}
            for h in influx_headers: row[h] = influx_vals.get(h, 0.0)

            with stress_lock: sc = global_stress_data.copy()
            for c in cnames:
                row[f"{c}_stressType"], row[f"{c}_stepStress"] = sc.get(c, [0, 0])

            with open(OUTPUT_FILE, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=headers, extrasaction='ignore').writerow(row)

            # Save snapshot from first row — permanent record of baseline state at collection start
            row_count += 1
            if row_count == 1:
                save_baseline_snapshot(row, snapshot_file)
                baseline_snapshot = {k: v for k, v in row.items()
                                     if any(x in k for x in ['cpu_user', 'memory_usage', 'transmit', 'receive', 'ul_brate', 'bsr'])}

            # Periodic drift check — warns if current values diverge from opening snapshot
            if row_count % DRIFT_CHECK_EVERY == 0:
                logging.info(f"Rows collected: {row_count}")
                check_baseline_drift(baseline_snapshot, row)

            # Accurate 1s Timing
            elapsed = time.time() - start_mark
            await asyncio.sleep(max(0, FETCH_INTERVAL - elapsed))

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    nuclear_cleanup()
    threading.Thread(target=stress_loop, daemon=True).start()
    asyncio.run(main())