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

# Your token, organization, and bucket information
influxDB_Token = "605bc59413b7d5457d181ccf20f9fda15693f81b068d70396cc183081b264f3b"
org = "srs"
bucket = "srsran"

# Configuration
PROMETHEUS_URL = "http://localhost:9090"
INPUT_FILE = "dataScrapper/allPromQuery.txt"
OUTPUT_FILE = "dataScrapper/prometheus_combined.csv"
EXTRA_DATA_FILE = "stresser/file_data.csv"
NUM_WORKERS = 20
BATCH_SIZE = 20
TIMEOUT = 10
RETRY_ATTEMPTS = 3
FETCH_INTERVAL = 1  # Seconds between fetch cycles

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


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
    async with InfluxDBClientAsync(url="http://175.40.1.5:8086", token=influxDB_Token, org=org) as client:
        query_api = client.query_api()
        query = '''
        from(bucket: "srsran")
          |> range(start: -10s)
          |> filter(fn: (r) => r["_measurement"] == "ue_info")
          |> filter(fn: (r) => r["testbed"] == "default")
        '''
        records = await query_api.query_stream(query)
        data = []
        async for record in records:
            try:
                average = float(record['_value'])
            except ValueError:
                average = 0.0  # Set to 0 if there's an error converting to float
            
            data.append({
                "key": f"{record['pci']}{record['rnti']}{record['_field']}",
                "average": average
            })
        return pd.DataFrame(data)


def load_extra_data(file_path: str) -> Dict:
    try:
        df = pd.read_csv(file_path)
        # logging.info(f"Successfully loaded extra data from {file_path}")
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception as e:
        logging.error(f"Error reading extra data file {file_path}: {str(e)}")
        return {}


async def main():
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
                        
            # Process Prometheus queries
            for i in range(0, len(queries), BATCH_SIZE):
                batch = queries[i:i + BATCH_SIZE]
                batch_results = await process_batch(client, batch)
                all_results.update(batch_results)
            
            # Fetch InfluxDB data (one row only)
            influxDF = await fetch_influx_data()
            if not influxDF.empty:
                influx_data = influxDF.iloc[0].to_dict()  # Convert the single row to a dictionary
            else:
                influx_data = {}

            # Load the latest extra data
            extra_data = load_extra_data(EXTRA_DATA_FILE)

            # Combine all data
            prometheus_data = {query: all_results.get(query, '') for query in queries}
            combined_data = {
                "Timestamp": timestamp,
                **prometheus_data,
                **influx_data,
                **extra_data
            }

            # Write the combined data to CSV
            with open(OUTPUT_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=combined_data.keys())
                if f.tell() == 0:  # Write header only if the file is empty
                    writer.writeheader()
                writer.writerow(combined_data)

            elapsed = time.time() - start_time
            # logging.info(f"Processed {len(queries)} queries in {elapsed:.2f}s")
            await asyncio.sleep(max(0, FETCH_INTERVAL - elapsed))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutting down gracefully...")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
