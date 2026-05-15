"""Fetch live metrics from InfluxDB (radio KPIs) and Prometheus/cAdvisor (infra KPIs).

Infra metrics are returned under nice names: {container}_cpu_pct, etc.
Radio metrics are returned as: PCI-{n}_RNTI-4601_{field}.
"""
import asyncio
import logging
import os
import subprocess

import aiohttp
from aiohttp import ClientTimeout
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync

INFLUXDB_URL    = "http://localhost:8086"
INFLUXDB_TOKEN  = os.environ.get("INFLUXDB_TOKEN", "")
INFLUXDB_ORG    = "srs"
INFLUXDB_BUCKET = "srsran"

PROMETHEUS_URL  = "http://localhost:9090"
CADVISOR        = "cadvisor:8080"
TIMEOUT_S       = 10

CONTAINERS = [
    "srscu0", "srscu1", "srscu2",
    "srsdu0", "srsdu1", "srsdu2", "srsdu3", "srsdu4", "srsdu5",
]
PCIS = list(range(1, 7))


def _container_id(name: str) -> str:
    try:
        return subprocess.check_output(
            f"docker inspect --format '{{{{.Id}}}}' {name}",
            shell=True, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return name


def _build_queries(container_ids: dict[str, str]) -> dict[str, str]:
    """Return {nice_name: prometheus_query} for all containers."""
    queries: dict[str, str] = {}
    for c in CONTAINERS:
        cid = container_ids.get(c, c)
        # cadvisor uses full container ID in the `id` label
        id_sel = f'id=~".*{cid[:12]}.*"' if len(cid) > 20 else f'name="{c}"'
        inst   = f'instance="{CADVISOR}"'

        queries[f"{c}_cpu_pct"] = (
            f'sum(irate(container_cpu_user_seconds_total{{{id_sel},{inst}}}[30s]) * 100)'
            f' / sum(machine_cpu_cores{{{inst}}})'
        )
        queries[f"{c}_memory_bytes"] = (
            f'sum(container_memory_usage_bytes{{{id_sel},{inst}}})'
            f' - sum(container_memory_cache{{{id_sel},{inst}}})'
        )
        queries[f"{c}_eth0_tx_bytes"] = (
            f'sum(irate(container_network_transmit_bytes_total'
            f'{{{id_sel},{inst},interface="eth0"}}[30s]))'
        )
        queries[f"{c}_eth0_rx_bytes"] = (
            f'sum(irate(container_network_receive_bytes_total'
            f'{{{id_sel},{inst},interface="eth0"}}[30s]))'
        )
    return queries


async def _prom_fetch_all(queries: dict[str, str]) -> dict[str, float]:
    timeout = ClientTimeout(total=TIMEOUT_S)
    results: dict[str, float] = {}

    async def fetch_one(session: aiohttp.ClientSession, name: str, query: str) -> tuple[str, float]:
        try:
            async with session.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": query},
            ) as resp:
                data = await resp.json()
                if data.get("status") == "success" and data["data"]["result"]:
                    return name, float(data["data"]["result"][0]["value"][1])
        except Exception:
            pass
        return name, 0.0

    async with aiohttp.ClientSession(timeout=timeout) as session:
        pairs = await asyncio.gather(
            *[fetch_one(session, n, q) for n, q in queries.items()]
        )
    return dict(pairs)


async def _influx_fetch(window_seconds: int) -> dict[str, float]:
    data: dict[str, float] = {}
    try:
        async with InfluxDBClientAsync(
            url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG
        ) as client:
            flux = (
                f'from(bucket: "{INFLUXDB_BUCKET}")'
                f' |> range(start: -{window_seconds}s)'
                f' |> filter(fn: (r) => r["_measurement"] == "ue_info")'
                f' |> filter(fn: (r) => r["rnti"] == "4601")'
                f' |> last()'
                f' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
            )
            tables = await client.query_api().query(flux)
            for table in tables:
                for record in table.records:
                    pci = record.values.get("pci")
                    skip = {"_start", "_stop", "_time", "_measurement",
                            "rnti", "pci", "testbed", "result", "table"}
                    for field, value in record.values.items():
                        if field in skip:
                            continue
                        key = f"PCI-{pci}_RNTI-4601_{field}"
                        try:
                            data[key] = float(value) if value is not None else 0.0
                        except (ValueError, TypeError):
                            data[key] = 0.0
    except Exception as e:
        logging.warning(f"[LiveFetcher] InfluxDB error: {e}")

    # Derived harq_error_rate
    for pci in PCIS:
        ok  = data.get(f"PCI-{pci}_RNTI-4601_dl_nof_ok",  0.0)
        nok = data.get(f"PCI-{pci}_RNTI-4601_dl_nof_nok", 0.0)
        total = ok + nok
        data[f"PCI-{pci}_RNTI-4601_harq_error_rate"] = nok / total if total > 0 else 0.0

    return data


class LiveFetcher:
    def __init__(self):
        logging.info("[LiveFetcher] Resolving container IDs...")
        self._container_ids = {c: _container_id(c) for c in CONTAINERS}
        self._prom_queries  = _build_queries(self._container_ids)
        logging.info(f"[LiveFetcher] Built {len(self._prom_queries)} Prometheus queries")

    def fetch(self, window_seconds: int = 30) -> dict[str, float]:
        """Synchronously fetch all live metrics and return merged dict."""
        queries = self._prom_queries

        async def _run() -> dict[str, float]:
            prom, influx = await asyncio.gather(
                _prom_fetch_all(queries),
                _influx_fetch(window_seconds),
            )
            return {**influx, **prom}

        return asyncio.run(_run())
