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
import json

# ================= Global Variables for Container Stress =================
global_stress_data: Dict[str, List[int]] = {}  # {container_name: [stress_type, step_stress]}
stress_lock = threading.Lock()  # For thread-safe access to global_stress_data
stress_pids: List[int] = []  # To track PIDs of launched stress processes

# ================= Configuration for Prometheus/InfluxDB =================

# Your token, organization, and bucket information
influxDB_Token = "605bc59413b7d5457d181ccf20f9fda15693f81b068d70396cc183081b264f3b"
org = "srs"
bucket = "srsran"

traffic_file = "trafficGenerator/traffic_distribution.txt"

PROMETHEUS_URL = "http://localhost:9090"
INPUT_FILE = "dataScrapper/allPromQuery.txt"
OUTPUT_FILE = "dataScrapper/prometheus_combined.csv"
NUM_WORKERS = 20
BATCH_SIZE = 20
TIMEOUT = 10
RETRY_ATTEMPTS = 3
FETCH_INTERVAL = 1  # Seconds between fetch cycles

# ================= Setup Logging =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


result = subprocess.run("pwd", capture_output=True, text=True)
pwd = result.stdout.strip()
print(f'Current working directory is {pwd}')

# ================= Prometheus/InfluxDB Data Scraper =================

class PrometheusClient:
    def __init__(self, base_url: str, timeout: int = TIMEOUT):
        self.base_url = base_url
        self.timeout = ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            connector=aiohttp.TCPConnector(
                limit=NUM_WORKERS * 2,
                ttl_dns_cache=300,
                force_close=False
            )
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
            except Exception as e:
                if attempt == retries - 1:
                    logging.error(f"Failed to fetch query '{query}': {str(e)}")
                    return None
                await asyncio.sleep(0.5 * (attempt + 1))


async def process_batch(client: PrometheusClient, queries: List[str]) -> Dict[str, Optional[float]]:
    tasks = [client.fetch_query(query) for query in queries]
    results = await asyncio.gather(*tasks)
    return dict(zip(queries, results))

async def fetch_influx_data() -> pd.DataFrame:
    dataFrame = {}
    uniqueFields = []
    async with InfluxDBClientAsync(url="http://175.40.1.5:8086", token=influxDB_Token, org=org) as client:
        # Stream of FluxRecords
        query_api = client.query_api()
        
        # Flux query to get all records for the last 2 seconds
        query = '''
        from(bucket: "srsran")
          |> range(start: -60s)
          |> filter(fn: (r) => r["_measurement"] == "ue_info")
          |> filter(fn: (r) => r["testbed"] == "default")
        '''
        
        
        


   
        
        fields = ['bsr', 'cqi', 'dl_brate', 'dl_bs', 'dl_mcs', 'dl_nof_nok', 'dl_nof_ok', 'pucch_snr_db', 'pucch_ta_ns', 'pusch_snr_db', 'pusch_ta_ns', 'ri', 'srs_ta_ns', 'ta_ns', 'ul_brate', 'ul_mcs', 'ul_nof_nok', 'ul_nof_ok']
        data = {}
        for field in fields:
            for PCI in range(1, 5):
                pci=PCI
                rnti = 4601

                dummy_query = f'''
                from(bucket: "srsran")
                |> range(start:-30s)
                |> filter(fn: (r) => r["_measurement"] == "ue_info")
                |> filter(fn: (r) => r["_field"] == "{field}")
                |> filter(fn: (r) => r["pci"] == "{pci}")
                |> filter(fn: (r) => r["rnti"] == "{rnti}")
                |> filter(fn: (r) => r["testbed"] == "default")
                '''

                dummy_records = await query_api.query_stream(dummy_query)
                values = []
                async for record in dummy_records:
                    if(record['_value']!='n/a' and record['_value']!=None):
                        values.append(float(record['_value']))
                    else:
                        values.append(0)

                    key = f"PCI-{record['pci']}_RNTI-{record['rnti']}_{record['_field']}"
                    if(len(values)!=0):
                        data[key]= sum(values)/len(values)
                    else:
                        data[key]= 0

        # print(data)        
        # Active_UEs_query = "from(bucket: \"srsran\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")\n  |> filter(fn: (r) => r[\"testbed\"] == \"default\")\n  |> filter(fn: (r) => r[\"_field\"] == \"dl_brate\")\n  |> map(fn: (r) => ({ r with ue_id: r[\"pci\"]+\".\"+r[\"rnti\"]}))\n  |> window(every: 2s)\n  |> group(columns: [\"_stop\"])\n  |> unique(column: \"ue_id\")\n  |> count(column: \"ue_id\")\n  |> map(fn: (r) => ({ r with _value: r[\"ue_id\"] }))\n  |> drop(columns: [\"ue_id\"])\n  |> group()\n"
        # Current_Total_Downlink_Bitrate_query = "from(bucket: \"srsran\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")\n  |> filter(fn: (r) => r[\"testbed\"] == \"default\")\n  |> filter(fn: (r) => r[\"_field\"] == \"dl_brate\")\n  |> window(every: 1s)\n  |> group(columns: [\"_stop\"])\n  |> sum(column: \"_value\")\n  |> group()\n  |> movingAverage(n: 2)\n  "
        # Maximum_Total_Downlink_Bitrate_query = "from(bucket: \"srsran\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")\n  |> filter(fn: (r) => r[\"testbed\"] == \"default\")\n  |> filter(fn: (r) => r[\"_field\"] == \"dl_brate\")\n  |> window(every: 1s)\n  |> group(columns: [\"_stop\"])\n  |> sum(column: \"_value\")\n  |> group()\n  |> movingAverage(n: 2)\n "
        # Num_Cells_with_Active_UEs_query = "from(bucket: \"srsran\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")\n  |> filter(fn: (r) => r[\"testbed\"] == \"default\")\n  |> filter(fn: (r) => r[\"_field\"] == \"dl_brate\")\n  |> window(every: 2s)\n  |> group(columns: [\"_stop\"])\n  |> unique(column: \"pci\")\n  |> count(column: \"pci\")\n  |> map(fn: (r) => ({ r with _value: r[\"pci\"] }))\n  |> drop(columns: [\"pci\"])\n  |> group()\n"
        # Downlink_Bitrate_query = "from(bucket: \"srsran\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")\n  |> filter(fn: (r) => r[\"testbed\"] == \"default\")\n  |> filter(fn: (r) => r[\"_field\"] == \"dl_brate\")\n"
        # Downlink_MCS_query = "from(bucket: \"srsran\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")  \n  |> filter(fn: (r) => r[\"testbed\"] == \"default\")\n  |> filter(fn: (r) => r[\"_field\"] == \"dl_mcs\")"
        # Uplink_Bitrate_query = "from(bucket: \"srsran\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")\n  |> filter(fn: (r) => r[\"testbed\"] == \"default\")\n  |> filter(fn: (r) => r[\"_field\"] == \"ul_brate\")\n"
        # Uplink_MCS_query = "from(bucket: \"srsran\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")  \n  |> filter(fn: (r) => r[\"testbed\"] == \"default\")\n  |> filter(fn: (r) => r[\"_field\"] == \"ul_mcs\")"
        # Uplink_SNR_query = "from(bucket: \"srsran\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")  \n  |> filter(fn: (r) => r[\"testbed\"] == \"default\")\n  |> filter(fn: (r) => r[\"_field\"] == \"pusch_snr_db\")"
        # CQI_query = "from(bucket: \"srsran\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")  \n  |> filter(fn: (r) => r[\"testbed\"] == \"default\")\n  |> filter(fn: (r) => r[\"_field\"] == \"cqi\")"

        # queries = [Active_UEs_query, Current_Total_Downlink_Bitrate_query, Maximum_Total_Downlink_Bitrate_query, Num_Cells_with_Active_UEs_query,
        #              Downlink_Bitrate_query, Downlink_MCS_query, Uplink_Bitrate_query, Uplink_MCS_query, Uplink_SNR_query, CQI_query]





        # # Execute the query
        # records = await query_api.query_stream(query)

        # # Process records and populate dataFrame
        # async for record in records:
        #     if(record['_field'] not in uniqueFields):
        #         uniqueFields.append(record['_field'])
        #     key = f"PCI-{record['pci']}_RNTI-{record['rnti']}_{record['_field']}"
        #     if key in dataFrame and (record['_value']!='n/a' and record['_value']!=None):
        #         dataFrame[key].append(float(record['_value']))
        #     elif(key in dataFrame):
        #         dataFrame[key].append(0)
        #     else:
        #         dataFrame[key] = [0]
        # print(uniqueFields)
        # # Calculate the averages and create a pandas DataFrame
        # data = [{key: sum(values) / len(values)} for key, values in dataFrame.items()]
        df = pd.DataFrame([data])        
        # print(df)
        return df

# ================= Container Stress Functions =================

def get_container_name(container_id: str) -> Optional[str]:
    """
    Retrieve the container name from its ID using docker inspect.
    """
    command = f"docker inspect {container_id}"
    try:
        output = subprocess.check_output(command, shell=True).decode()
        container_info = json.loads(output)
        container_name = container_info[0]['Name'].lstrip('/')
        return container_name
    except subprocess.CalledProcessError as e:
        print(f"Error while fetching container name: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error while parsing JSON output: {e}")
        return None

def cleanup_stress():
    """Stop all stress processes and cleanup inside all containers."""
    global stress_pids
    print("\nCleaning up stress processes...")
    with stress_lock:
        # Terminate host-side docker exec processes
        for pid in stress_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Terminated process with PID {pid}")
            except OSError as e:
                print(f"Error terminating process with PID {pid}: {e}")
        stress_pids.clear()

        # Kill any remaining stress-ng processes inside containers
        for cname in global_stress_data.keys():
            cleanup_cmd = f"docker exec {cname} pkill -f stress-ng"
            subprocess.run(cleanup_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sys.exit(0)



def injectStress(container_id: str, typeOfStress: int, d: int, percentageOfStress: int, percentageOfStressEnd: int):
    """
    Inject stress into a Docker container based on the specified type.
    """
    global stress_pids

    cname = get_container_name(container_id)  # Get container name before acquiring lock
    if not cname:
        print(f"Could not retrieve container name for ID: {container_id}")
        return
    
    # Cleanup any existing stress-ng processes inside the container
    cleanup_cmd = f"docker exec {container_id} pkill -f stress-ng"
    subprocess.run(cleanup_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # with stress_lock:
    #     global_stress_data[cname] = [typeOfStress, percentageOfStress]

    try:
        if typeOfStress == 0:  # No stress
            time.sleep(d)

        elif typeOfStress == 1:  # CPU stress
            cpu_cores = 2
            remaining_time = d
            current_stress = percentageOfStress

            while remaining_time > 0:
                step_duration = min(random.randint(2, 5), remaining_time)
                remaining_time -= step_duration
                step_stress = random.randint(current_stress, percentageOfStressEnd)

                stress_cmd = (f"docker exec {container_id} stress-ng --cpu {cpu_cores} "
                              f"--cpu-load {step_stress} --timeout {step_duration}s")
                proc = subprocess.Popen(stress_cmd, shell=True)
                
                with stress_lock:
                    global_stress_data[cname] = [typeOfStress, step_stress]
                    stress_pids.append(proc.pid)

                proc.wait()
                current_stress = step_stress

        elif typeOfStress == 2:  # Memory stress
            remaining_time = d
            current_stress = percentageOfStress

            while remaining_time > 0:
                step_duration = min(random.randint(2, 5), remaining_time)
                remaining_time -= step_duration
                step_stress = random.randint(current_stress, percentageOfStressEnd)

                stress_cmd = (f"docker exec {container_id} stress-ng --vm 1 "
                              f"--vm-bytes {step_stress}% --timeout {step_duration}s")
                proc = subprocess.Popen(stress_cmd, shell=True)

                with stress_lock:
                    global_stress_data[cname] = [typeOfStress, step_stress]
                    stress_pids.append(proc.pid)

                proc.wait()
                current_stress = step_stress

        elif typeOfStress == 3:  # Packet loss stress
            cleanup_cmd = f"docker exec {container_id} tc qdisc del dev eth0 root netem"
            os.system(cleanup_cmd)

            remaining_time = d
            current_stress = percentageOfStress

            while remaining_time > 0:
                step_duration = min(random.randint(2, 5), remaining_time)
                remaining_time -= step_duration
                step_stress = random.randint(current_stress, percentageOfStressEnd)

                tc_cmd = f"docker exec {container_id} tc qdisc add dev eth0 root netem loss {step_stress}%"
                subprocess.run(tc_cmd, shell=True)

                with stress_lock:
                    global_stress_data[cname] = [typeOfStress, step_stress]

                time.sleep(step_duration)
                os.system(cleanup_cmd)

        # Ensure stress data resets
        with stress_lock:
            global_stress_data[cname] = [0, 0]

        print(f"Stress completed for {cname}. Resetting stress data.")

    except Exception as e:
        print(f"Error applying stress to {cname}: {e}")


def cleanup_stress():
    """Stop all stress processes and cleanup."""
    global stress_pids
    print("\nCleaning up stress processes...")
    with stress_lock:
        for pid in stress_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Terminated process with PID {pid}")
            except OSError as e:
                print(f"Error terminating process with PID {pid}: {e}")
        stress_pids.clear()
    sys.exit(0)


def signal_handler(sig, frame):
    """Handle interrupt signal to perform cleanup."""
    cleanup_stress()


def stress_loop():
    """
    Continuously apply stress on a set of containers.
    This function runs in a separate thread.
    """
    cmd = "docker ps --filter 'name=^srscu' --filter 'name=^srsdu' --format '{{.ID}}'"
    try:
        container_ids = subprocess.check_output(cmd, shell=True).decode().splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Error fetching container IDs: {e}")
        return

    # Initialize global stress data using container names
    for cid in container_ids:
        cname = get_container_name(cid)
        if cname:
            with stress_lock:
                global_stress_data[cname] = [0, 0]

    MAX_STRESS_CONTAINERS = 1  # Change this to your desired number

    while True:
        duration = random.randint(120, 300)
        type_of_stress = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
        
        threads = []
        stress_candidates = []
        read_list = []

        # Read the traffic distribution once
        with open(f'{pwd}/trafficGenerator/traffic_distribution.txt', 'r') as file:
            for line in file:
                read_list.append(line.strip())

        for cid in container_ids:
            cname = get_container_name(cid)
            if not cname:
                continue

            if ((float(read_list[1]) - float(read_list[0]) > 0) or len(stress_candidates) == 0):
                is_stress = np.random.choice([0, 1], p=[0.2, 0.8])
            else:
                is_stress = 0

            if is_stress:
                stress_candidates.append(cid)

        # Select at most MAX_STRESS_CONTAINERS containers to stress
        selected_cids = random.sample(stress_candidates, min(MAX_STRESS_CONTAINERS, len(stress_candidates)))

        for cid in selected_cids:
            if type_of_stress == 1:
                perc_start = np.random.randint(40, 91)
                perc_end = np.random.randint(perc_start, 101)
            elif type_of_stress == 2:
                perc_start = np.random.randint(25, 36)
                perc_end = np.random.randint(perc_start, 61)
            else:
                perc_start = np.random.randint(1, 4)
                perc_end = np.random.randint(perc_start, 5)

            t = threading.Thread(
                target=injectStress,
                args=(cid, type_of_stress, duration, perc_start, perc_end)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        print("Stress cycle completed. Moving to next iteration.")


# ================= Asynchronous Main Loop =================

async def main():
    # Read Prometheus queries from file
    try:
        with open(INPUT_FILE, "r") as f:
            queries = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        logging.error(f"Input file {INPUT_FILE} not found")
        return

    if not queries:
        logging.error("No queries found in input file")
        return

    async with PrometheusClient(PROMETHEUS_URL) as client:
        while True:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            all_results = {}

            # Process Prometheus queries in batches
            for i in range(0, len(queries), BATCH_SIZE):
                batch = queries[i:i + BATCH_SIZE]
                batch_results = await process_batch(client, batch)
                all_results.update(batch_results)

            # Fetch InfluxDB data (one row only)
            influxDF = await fetch_influx_data()
            if not influxDF.empty:
                influx_data = influxDF.iloc[-1].to_dict()
            else:
                influx_data = {}

            # Instead of loading from file, grab the latest container stress data from the global variable.
            with stress_lock:
                stress_copy = global_stress_data.copy()
            # Format stress data to use keys like "<container>_stressType" and "<container>_stepStress"
            stress_data_formatted = {}
            for cname, (stress_type, step_stress) in stress_copy.items():
                stress_data_formatted[f"{cname}_stressType"] = stress_type
                stress_data_formatted[f"{cname}_stepStress"] = step_stress

            # Combine all data into one dictionary.
            prometheus_data = {query: all_results.get(query, '') for query in queries}
            combined_data = {
                "Timestamp": timestamp,
                **prometheus_data,
                **influx_data,
                **stress_data_formatted
            }

            # print("\nInflucData = ", influx_data)
            # Write the combined data to CSV (appending a row each cycle)
            with open(OUTPUT_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=combined_data.keys())
                if f.tell() == 0:  # Write header only if file is empty
                    writer.writeheader()
                writer.writerow(combined_data)

            elapsed = time.time() - start_time
            # Wait until next cycle (ensuring we respect the fetch interval)
            await asyncio.sleep(max(0, FETCH_INTERVAL - elapsed))

# ================= Program Entry Point =================

if __name__ == "__main__":
    # Register the signal handler in the main thread
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start the container stress loop in a background thread.
    stress_thread = threading.Thread(target=stress_loop, daemon=True)
    stress_thread.start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutting down gracefully...")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
