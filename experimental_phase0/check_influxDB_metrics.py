from influxdb_client import InfluxDBClient

token = "605bc59413b7d5457d181ccf20f9fda15693f81b068d70396cc183081b264f3b"
org = "srs"
url = "http://localhost:8086"

client = InfluxDBClient(url=url, token=token, org=org)

# Query to find all unique field names in the 'ue_info' measurement
query = 'import "influxdata/influxdb/schema" schema.measurementFieldKeys(bucket: "srsran", measurement: "ue_info")'

result = client.query_api().query(query)

db_fields = [record.get_value() for table in result for record in table.records]
script_fields = ['bsr', 'cqi', 'dl_brate', 'dl_bs', 'dl_mcs', 'dl_nof_nok', 'dl_nof_ok', 'pucch_snr_db', 'pucch_ta_ns', 'pusch_snr_db', 'pusch_ta_ns', 'ri', 'srs_ta_ns', 'ta_ns', 'ul_brate', 'ul_mcs', 'ul_nof_nok', 'ul_nof_ok']

print("--- INFRASTRUCTURE AUDIT ---")
print(f"Fields in Database: {len(db_fields)}")
print(f"Fields in Scraper:  {len(script_fields)}")

missing = set(db_fields) - set(script_fields)
if missing:
    print(f"\n[!] YOU ARE MISSING THESE METRICS:")
    for m in missing:
        print(f" -> {m}")
else:
    print("\n[V] Your scraper is collecting all available fields.")