"""Load train_normal_th3.csv and compute per-KPI baseline statistics.

Infra columns in the CSV are raw Prometheus query strings — matched by substring.
Radio columns are nice-named: PCI-{n}_RNTI-4601_{field}.
"""
import logging
import pandas as pd
from typing import Optional

CONTAINERS = [
    "srscu0", "srscu1", "srscu2",
    "srsdu0", "srsdu1", "srsdu2", "srsdu3", "srsdu4", "srsdu5",
]
PCIS = list(range(1, 7))
RADIO_FIELDS = ["ul_brate", "dl_brate", "bsr", "cqi", "pucch_snr_db", "pusch_snr_db",
                "dl_nof_ok", "dl_nof_nok"]


def _find_col(columns: list[str], required: list[str], excluded: list[str] = None) -> Optional[str]:
    excluded = excluded or []
    for c in columns:
        if all(p in c for p in required) and not any(e in c for e in excluded):
            return c
    return None


class BaselineStats:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path, low_memory=False)
        if "phase" in df.columns:
            df = df[df["phase"] == "NORMAL"]
        logging.info(f"[Baseline] Normal rows: {len(df)}")

        self.stats: dict[str, dict] = {}
        cols = list(df.columns)

        # Radio KPIs (InfluxDB — column names already nice)
        for pci in PCIS:
            for field in RADIO_FIELDS:
                col = f"PCI-{pci}_RNTI-4601_{field}"
                if col in cols:
                    self._ingest(df[col], col)

        # Derived harq_error_rate per PCI
        for pci in PCIS:
            ok_col  = f"PCI-{pci}_RNTI-4601_dl_nof_ok"
            nok_col = f"PCI-{pci}_RNTI-4601_dl_nof_nok"
            key     = f"PCI-{pci}_RNTI-4601_harq_error_rate"
            if ok_col in cols and nok_col in cols:
                ok  = pd.to_numeric(df[ok_col],  errors="coerce").fillna(0)
                nok = pd.to_numeric(df[nok_col], errors="coerce").fillna(0)
                total = ok + nok
                series = (nok / total.replace(0, float("nan"))).fillna(0)
                self._ingest(series, key)

        # Infrastructure KPIs (Prometheus — columns are raw query strings)
        for c in CONTAINERS:
            mapping = {
                f"{c}_cpu_pct": _find_col(cols,
                    [f'name="{c}"', "cpu_user_seconds_total", "machine_cpu_cores"]),
                f"{c}_memory_bytes": _find_col(cols,
                    [f'name="{c}"', "container_memory_usage_bytes", "container_memory_cache"],
                    excluded=["machine_memory_bytes"]),
                f"{c}_eth0_tx_bytes": _find_col(cols,
                    [f'name="{c}"', "network_transmit_bytes_total", 'interface="eth0"']),
                f"{c}_eth0_rx_bytes": _find_col(cols,
                    [f'name="{c}"', "network_receive_bytes_total", 'interface="eth0"']),
            }
            for nice, csv_col in mapping.items():
                if csv_col:
                    self._ingest(df[csv_col], nice)
                else:
                    logging.warning(f"[Baseline] No CSV column found for {nice}")

        logging.info(f"[Baseline] Loaded {len(self.stats)} KPI baselines")

    def _ingest(self, series: pd.Series, key: str):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) < 2:
            return
        self.stats[key] = {
            "mean": float(s.mean()),
            "std":  float(s.std()),
            "p5":   float(s.quantile(0.05)),
            "p95":  float(s.quantile(0.95)),
            "n":    int(len(s)),
        }

    def has(self, key: str) -> bool:
        return key in self.stats

    def z_score(self, key: str, value: float) -> float:
        s = self.stats.get(key)
        if not s or s["std"] == 0:
            return 0.0
        return (value - s["mean"]) / s["std"]

    def pct_deviation(self, key: str, value: float) -> float:
        s = self.stats.get(key)
        if not s or s["mean"] == 0:
            return 0.0
        return (value - s["mean"]) / s["mean"] * 100.0
