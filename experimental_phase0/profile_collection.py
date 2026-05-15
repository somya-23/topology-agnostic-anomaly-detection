import aiohttp
import asyncio
import time
from datetime import datetime
import logging
from aiohttp import ClientTimeout
from typing import Dict, List, Optional
import pandas as pd
import csv
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
import os
import random
import subprocess
import threading
import signal
import sys
import re

# =====================================================================
# 1. CONFIGURATION: TARGETED RECOVERY PROFILING
# =====================================================================
MODE = "ANOMALY"   
OUTPUT_FILE = "recovery_profiling.csv" # Saved in current directory for easy access
INPUT_FILE = "../dataScrapper/promCadvisor.txt" # Adjust path if necessary
PROMETHEUS_URL = "http://localhost:9090"
influxDB_Token = "605bc59413b7d5457d181ccf20f9fda15693f81b068d70396cc183081b264f3b"
org = "srs"

FETCH_INTERVAL = 1.0   
STRESS_DURATION = 180  # 3 minutes

NUM_WORKERS = 50
TIMEOUT = 10
RETRY_ATTEMPTS = 3

global_stress_data: Dict[str, List[int]] = {}
stress_lock = threading.Lock()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# =====================================================================
# 2. CORE UTILITIES
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
    logging.info("Shutting down and cleaning up containers...")
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

# =====================================================================
# 3. TELEMETRY CLIENTS
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
                        if field in ['_start', '_stop', '_time', '_measurement', 'rnti', 'pci', 'testbed', 'result', 'table']: continue
                        key = f"PCI-{pci}_RNTI-4601_{field}"
                        try: data[key] = float(value) if value is not None and value != 'n/a' else 0.0
                        except: data[key] = 0.0
    except Exception: pass
    return pd.DataFrame([data])

def injectStress(container_id, typeOfStress, duration):
    cname = get_container_name(container_id)
    intensity = 95 # Hardcoded high intensity for clear profiling
    
    try:
        cmd = f"docker exec {container_id} stress-ng --cpu 4 --cpu-load {intensity} --timeout {duration}s"
        with stress_lock: global_stress_data[cname] = [typeOfStress, intensity]
        subprocess.run(cmd, shell=True)
    finally:
        with stress_lock: global_stress_data[cname] = [0, 0]

# =====================================================================
# 4. TARGETED PROFILING LOOP (10m -> 3m -> 20m)
# =====================================================================
def stress_loop():
    logging.info("PHASE 1: Starting 10-minute (600s) normal baseline collection...")
    time.sleep(600)
    
    # Find srscu0
    output = subprocess.check_output("docker ps --format '{{.ID}} {{.Names}}'", shell=True).decode().splitlines()
    cmap = {line.split()[1]: line.split()[0] for line in output if len(line.split()) >= 2}
    victim = cmap.get("srscu0")

    if not victim:
        logging.error("Could not find container srscu0! Exiting stress loop.")
        os.kill(os.getpid(), signal.SIGINT)
        return

    logging.info("PHASE 2: Injecting CPU Stress into srscu0 for 3 minutes (180s)...")
    injectStress(victim, 1, STRESS_DURATION)
    
    logging.info("PHASE 3: Stress ended. Starting 20-minute (1200s) cooling tracking...")
    time.sleep(1200)
    
    logging.info("EXPERIMENT COMPLETE! Auto-stopping the scraper...")
    os.kill(os.getpid(), signal.SIGINT) # Automatically stops the script

# =====================================================================
# 5. MAIN SCRAPER
# =====================================================================
async def main():
    with open(INPUT_FILE, "r") as f:
        raw_queries = [line.strip() for line in f.readlines() if line.strip()]
    actual_queries = fix_queries_with_ids(raw_queries)

    print("Locking InfluxDB Schema...")
    init_df = await fetch_influx_data()
    influx_headers = sorted(list(init_df.columns))

    cnames = sorted(subprocess.check_output("docker ps --filter 'name=^srscu' --filter 'name=^srsdu' --format '{{.Names}}'", shell=True).decode().splitlines())
    stress_headers = []
    for c in cnames:
        stress_headers.extend([f"{c}_stressType", f"{c}_stepStress"])
        with stress_lock: global_stress_data[c] = [0, 0]

    headers = ["Timestamp"] + raw_queries + influx_headers + stress_headers
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()

    async with PrometheusClient(PROMETHEUS_URL) as client:
        while True:
            start_mark = time.time()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            prom_task = asyncio.gather(*[client.fetch_query(aq) for aq in actual_queries])
            prom_results, current_influx_df = await asyncio.gather(prom_task, fetch_influx_data())

            row = {"Timestamp": ts}
            row.update(dict(zip(raw_queries, prom_results)))
            
            influx_vals = current_influx_df.iloc[0].to_dict() if not current_influx_df.empty else {}
            for h in influx_headers: row[h] = influx_vals.get(h, 0.0)

            with stress_lock: sc = global_stress_data.copy()
            for c in cnames:
                row[f"{c}_stressType"], row[f"{c}_stepStress"] = sc.get(c, [0, 0])

            with open(OUTPUT_FILE, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=headers, extrasaction='ignore').writerow(row)

            elapsed = time.time() - start_mark
            await asyncio.sleep(max(0, FETCH_INTERVAL - elapsed))

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    nuclear_cleanup()
    threading.Thread(target=stress_loop, daemon=True).start()
    asyncio.run(main())