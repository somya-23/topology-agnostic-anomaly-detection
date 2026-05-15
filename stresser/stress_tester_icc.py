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
import numpy as np
import signal
import sys
import re

# Add this after logging.basicConfig
file_handler = logging.FileHandler('experiment_log.txt')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

# ================= Global Variables =================
global_stress_data: Dict[str, List[int]] = {}
stress_lock = threading.Lock()
stress_pids: List[int] = []

# ================= Configuration =================
influxDB_Token = os.environ.get("INFLUXDB_TOKEN")
if not influxDB_Token:
    raise RuntimeError("INFLUXDB_TOKEN env var required. Export it before running this script.")
org = "srs"
bucket = "srsran"

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMETHEUS_URL = "http://localhost:9090"
INPUT_FILE = os.path.join(BASE_DIR, "dataScrapper", "promCadvisor.txt")

# *** CHANGE THIS FILENAME MANUALLY BEFORE RUNNING ***
# Use "correct_normal.csv" for Normal Data
# Use "correct_test.csv" for Anomaly Data
OUTPUT_FILE = os.path.join(BASE_DIR, "dataScrapper", "train_normal_th3.csv")

NUM_WORKERS = 40
TIMEOUT = 10
RETRY_ATTEMPTS = 3
FETCH_INTERVAL = 1

# ================= Setup Logging =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================= CLEANUP FUNCTIONS (CRITICAL) =================

def nuclear_cleanup():
    """
    Forcefully cleans up ALL stress processes and network rules 
    from ALL Docker containers.
    """
    print("\n[CLEANUP] Executing Nuclear Cleanup Sequence...")
    
    # 1. Kill Local Python Subprocesses
    if 'stress_pids' in globals() and stress_pids:
        for pid in stress_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        stress_pids.clear()

    # 2. Get all running container IDs
    try:
        cmd = "docker ps --format '{{.ID}}'"
        container_ids = subprocess.check_output(cmd, shell=True).decode().splitlines()
    except:
        return

    # 3. Clean inside containers
    for cid in container_ids:
        # Kill stress-ng
        subprocess.run(f"docker exec {cid} pkill -9 stress-ng", shell=True, stderr=subprocess.DEVNULL)
        # Delete TC Network Rules
        subprocess.run(f"docker exec {cid} tc qdisc del dev eth0 root", shell=True, stderr=subprocess.DEVNULL)

    print("[CLEANUP] Environment is clean.\n")

def signal_handler(sig, frame):
    nuclear_cleanup()
    sys.exit(0)

# ================= Helper Functions =================

def get_container_full_id(container_name: str) -> Optional[str]:
    try:
        cmd = f"docker inspect --format '{{{{.Id}}}}' {container_name}"
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except:
        return None

def get_container_name(container_id: str) -> Optional[str]:
    try:
        cmd = f"docker inspect --format '{{{{.Name}}}}' {container_id}"
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip().lstrip('/')
    except:
        return None

def fix_queries_with_ids(queries: List[str]) -> List[str]:
    print("--- Resolving Container IDs for Queries ---")
    fixed_queries = []
    name_pattern = re.compile(r'name="([^"]+)"')
    for q in queries:
        match = name_pattern.search(q)
        if match:
            cname = match.group(1)
            if cname in ['cadvisor', 'node-exporter', 'prometheus', 'open5gs_5gc']:
                 pass
            cid = get_container_full_id(cname)
            if cid:
                new_q = q.replace(f'name="{cname}"', f'id=~".*{cid}.*"')
                fixed_queries.append(new_q)
            else:
                fixed_queries.append(q)
        else:
            fixed_queries.append(q)
    return fixed_queries

# ================= Prometheus Client =================

class PrometheusClient:
    def __init__(self, base_url: str, timeout: int = TIMEOUT):
        self.base_url = base_url
        self.timeout = ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            connector=aiohttp.TCPConnector(limit=NUM_WORKERS * 2, ttl_dns_cache=300)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_query(self, query: str, retries: int = RETRY_ATTEMPTS) -> Optional[float]:
        for attempt in range(retries):
            try:
                async with self.session.get(
                    f"{self.base_url}/api/v1/query",
                    params={"query": query},
                    raise_for_status=True
                ) as response:
                    result = await response.json()
                    if result.get('status') == 'success' and result.get('data', {}).get('result'):
                        return float(result['data']['result'][0]['value'][1])
                    return None
            except Exception:
                if attempt == retries - 1:
                    return None
                await asyncio.sleep(0.5)

async def process_batch(client: PrometheusClient, queries: List[str]) -> Dict[str, Optional[float]]:
    tasks = [client.fetch_query(query) for query in queries]
    results = await asyncio.gather(*tasks)
    return dict(zip(queries, results))

# ================= InfluxDB =================

async def fetch_influx_data() -> pd.DataFrame:
    data = {}
    try:
        async with InfluxDBClientAsync(url="http://localhost:8086", token=influxDB_Token, org=org) as client:
            query_api = client.query_api()
            fields = ['bsr', 'cqi', 'dl_brate', 'dl_bs', 'dl_mcs', 'dl_nof_nok', 'dl_nof_ok', 'pucch_snr_db', 'pucch_ta_ns', 'pusch_snr_db', 'pusch_ta_ns', 'ri', 'srs_ta_ns', 'ta_ns', 'ul_brate', 'ul_mcs', 'ul_nof_nok', 'ul_nof_ok']
            for field in fields:
                for PCI in range(1, 7):
                    dummy_query = f'''
                    from(bucket: "srsran")
                    |> range(start:-30s)
                    |> filter(fn: (r) => r["_measurement"] == "ue_info")
                    |> filter(fn: (r) => r["_field"] == "{field}")
                    |> filter(fn: (r) => r["pci"] == "{PCI}")
                    |> filter(fn: (r) => r["rnti"] == "4601")
                    '''
                    dummy_records = await query_api.query_stream(dummy_query)
                    values = []
                    async for record in dummy_records:
                        if(record['_value']!='n/a' and record['_value']!=None):
                            values.append(float(record['_value']))
                        else:
                            values.append(0)
                    key = f"PCI-{PCI}_RNTI-4601_{field}"
                    if(len(values)!=0):
                        data[key]= sum(values)/len(values)
                    else:
                        data[key]= 0
    except Exception as e:
        print(f"InfluxDB Error: {e}")
    return pd.DataFrame([data])

# ================= Stress Functions =================

# ================= CALIBRATED STRESS INJECTION =================

def injectStress(container_id, typeOfStress, duration, perc_start, perc_end, pattern="STATIC"):
    cname = get_container_name(container_id)
    print(f"\n--- Injecting {pattern} Stress on {cname} (Type {typeOfStress}) ---")
    
    # Cleanup before start
    subprocess.run(f"docker exec {container_id} pkill -f stress-ng", shell=True, stderr=subprocess.DEVNULL)
    subprocess.run(f"docker exec {container_id} tc qdisc del dev eth0 root", shell=True, stderr=subprocess.DEVNULL)

    try:
        target_intensity = random.randint(perc_start, perc_end)

        # --- TYPE 1: CPU (Fixed at 85-90%) ---
        if typeOfStress == 1:
            workers = 4
            load = random.randint(85, 90)  # Corrected range
            print(f"🔥 CPU Stress: {workers} Workers @ {load}% Load")
            cmd = f"docker exec {container_id} stress-ng --cpu {workers} --cpu-load {load} --timeout {duration}s"
            
            proc = subprocess.Popen(cmd, shell=True)
            with stress_lock:
                global_stress_data[cname] = [1, target_intensity]
                stress_pids.append(proc.pid)
            proc.wait()

        # --- TYPE 2: MEMORY ---
        elif typeOfStress == 2:
            # Intensity 60-90 maps to ~1.8GB - ~2.8GB
            megabytes = int((target_intensity / 100.0) * 3100) 
            bytes_to_stress = f"{megabytes}M"
            print(f"🔥 Memory Stress: Injecting {bytes_to_stress}")
            cmd = f"docker exec {container_id} stress-ng --vm 1 --vm-bytes {bytes_to_stress} --timeout {duration}s"
            
            proc = subprocess.Popen(cmd, shell=True)
            with stress_lock:
                global_stress_data[cname] = [2, target_intensity]
                stress_pids.append(proc.pid)
            proc.wait()

        # --- TYPE 3: NETWORK (3% to 6% Packet Loss) ---
        elif typeOfStress == 3:
            loss_val = 3.0 + (3.0 * (target_intensity / 100.0)) 
            print(f"🔥 Network Stress: {loss_val:.2f}% Packet Loss")
            
            # Use 'replace' instead of 'add' to avoid "File exists" errors
            cmd = f"docker exec {container_id} tc qdisc replace dev eth0 root netem loss {loss_val:.2f}%"
            # If 'replace' fails because no rule exists, 'add' it
            if subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL).returncode != 0:
                subprocess.run(f"docker exec {container_id} tc qdisc add dev eth0 root netem loss {loss_val:.2f}%", shell=True)
            
            with stress_lock:
                global_stress_data[cname] = [3, target_intensity]
            time.sleep(duration)
            subprocess.run(f"docker exec {container_id} tc qdisc del dev eth0 root", shell=True, stderr=subprocess.DEVNULL)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        with stress_lock:
             global_stress_data[cname] = [0, 0]
# ================= PRODUCTION STRESS LOOP (With Traffic Logic) =================




# ================= TRAFFIC-AWARE CONTROL LOOP =================

# TOGGLE THIS MANUALLY:
# MODE = "NORMAL"   # For clean training data
MODE = "ANOMALY"  # For testing data with attacks

def stress_loop():
    if MODE == "NORMAL": return
    time.sleep(10)
    
    cmd = "docker ps --filter 'name=^srscu' --filter 'name=^srsdu' --format '{{.ID}}'"
    container_ids = subprocess.check_output(cmd, shell=True).decode().splitlines()

    while True:
        stress_type = random.choice([1, 2, 3]) 
        duration = random.randint(45, 90)
        candidates = [cid for cid in container_ids if "srs" in get_container_name(cid)]
        
        if candidates:
            # WEIGHTS: du1 gets 1, everything else gets 5 (5x less likely)
            weights = [1 if "du1" in get_container_name(cid) else 5 for cid in candidates]
            victim = random.choices(candidates, weights=weights, k=1)[0]
            cname = get_container_name(victim)

            print(f"!!! RANDOM FAULT -> Injecting Type {stress_type} on {cname} !!!")
            injectStress(victim, stress_type, duration, 60, 90) # Synchronous call is fine here
            
            # Cooldown to maintain 10% anomaly density
            sleep_time = int(duration * random.uniform(8.0, 10.0))
            print(f"Cooling down for {sleep_time}s...")
            time.sleep(sleep_time)
             
# ================= Main =================

async def main():
    try:
        with open(INPUT_FILE, "r") as f:
            raw_queries = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        logging.error(f"Input file {INPUT_FILE} not found")
        return

    actual_queries = fix_queries_with_ids(raw_queries)
    query_map = dict(zip(raw_queries, actual_queries))
    headers = ["Timestamp"] + raw_queries
    
    # Add Influx headers
    influx_fields = ['bsr', 'cqi', 'dl_brate', 'dl_bs', 'dl_mcs', 'dl_nof_nok', 'dl_nof_ok', 'pucch_snr_db', 'pucch_ta_ns', 'pusch_snr_db', 'pusch_ta_ns', 'ri', 'srs_ta_ns', 'ta_ns', 'ul_brate', 'ul_mcs', 'ul_nof_nok', 'ul_nof_ok']
    for field in influx_fields:
        for PCI in range(1, 7):
            headers.append(f"PCI-{PCI}_RNTI-4601_{field}")

    # Add Stress headers
    try:
        cmd = "docker ps --filter 'name=^srscu' --filter 'name=^srsdu' --format '{{.Names}}'"
        cnames = subprocess.check_output(cmd, shell=True).decode().splitlines()
        for cname in cnames:
            headers.append(f"{cname}_stressType")
            headers.append(f"{cname}_stepStress")
            with stress_lock:
                if cname not in global_stress_data:
                    global_stress_data[cname] = [0, 0]
    except:
        pass

    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    async with PrometheusClient(PROMETHEUS_URL) as client:
        while True:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Fetch Data
            batch_results = await process_batch(client, actual_queries)
            formatted_results = {q: batch_results.get(aq, '') for q, aq in query_map.items()}
            influxDF = await fetch_influx_data()
            influx_data = influxDF.iloc[-1].to_dict() if not influxDF.empty else {}

            # Stress Data
            with stress_lock:
                stress_copy = global_stress_data.copy()
            stress_data_formatted = {}
            for cname, (stype, sval) in stress_copy.items():
                stress_data_formatted[f"{cname}_stressType"] = stype
                stress_data_formatted[f"{cname}_stepStress"] = sval

            combined_data = {
                "Timestamp": timestamp,
                **formatted_results,
                **influx_data,
                **stress_data_formatted
            }

            with open(OUTPUT_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
                writer.writerow(combined_data)

            elapsed = time.time() - start_time
            await asyncio.sleep(max(0, FETCH_INTERVAL - elapsed))

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    # 1. Clean on Startup
    nuclear_cleanup()
    
    try:
        stress_thread = threading.Thread(target=stress_loop, daemon=True)
        stress_thread.start()
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping script...")
    finally:
        # 2. Clean on Exit
        nuclear_cleanup()