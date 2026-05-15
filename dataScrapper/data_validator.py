#!/usr/bin/env python3
"""
O-RAN srsRAN FALCON - CSV Data Validator
=========================================
Audits a CSV produced by scrapper_prometheus_backup.py for ML-training readiness.

Runs 13 deterministic checks and emits PASS/WARN/FAIL per check, plus a JSON
report saved next to the input CSV as validation_<basename>.json.

Usage:
    python3 data_validator.py \
        --csv experimental_phase0/data_fs_plane/train_normal_th3.csv \
        --expected-mode NORMAL \
        --expected-duration-sec 23400 \
        --fetch-interval 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("data_validator")


# Sentinel substrings used to recognise specific column kinds inside raw PromQL strings.
PROM_CPU_SUBSTR = "cpu_user"
PROM_MEM_SUBSTR = "memory_usage"
PROM_FS_SUBSTR = "fs_usage_bytes"
PROM_NET_RX_SUBSTR = "container_network_receive_bytes_total"
PROM_NET_TX_SUBSTR = "container_network_transmit_bytes_total"
ETH0_TAG = 'interface="eth0"'
ETH1_TAG = 'interface="eth1"'

EXPECTED_CONTAINERS = (
    "srscu0", "srscu1", "srscu2",
    "srsdu0", "srsdu1", "srsdu2", "srsdu3", "srsdu4", "srsdu5",
)
INFLUX_FIELDS = (
    "bsr", "cqi", "dl_brate", "dl_bs", "dl_mcs", "dl_nof_nok", "dl_nof_ok",
    "pucch_snr_db", "pucch_ta_ns", "pusch_snr_db", "pusch_ta_ns",
    "ri", "srs_ta_ns", "ta_ns", "ul_brate", "ul_mcs", "ul_nof_nok", "ul_nof_ok",
)
PCI_RANGE = range(1, 7)
TRAFFIC_BIN_SECONDS = 15 * 60  # 15-min staircase steps in test_traffic.py


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str
    metric: Any = None


def _container_columns(df: pd.DataFrame, container: str, substr: str) -> list[str]:
    pattern = f'name="{container}"'
    return [
        c for c in df.columns
        if pattern in c and substr in c
    ]


def _eth_columns(df: pd.DataFrame, eth_tag: str) -> list[str]:
    return [c for c in df.columns if eth_tag in c]


def _stress_type_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_stressType")]


def _stress_step_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_stepStress")]


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_row_count(df: pd.DataFrame, expected_rows: int) -> CheckResult:
    actual = len(df)
    ratio = actual / expected_rows if expected_rows else 0.0
    if ratio >= 0.95:
        status = "PASS"
    elif ratio >= 0.80:
        status = "WARN"
    else:
        status = "FAIL"
    return CheckResult(
        name="row_count",
        status=status,
        detail=f"actual={actual} expected≈{expected_rows} ({ratio:.1%})",
        metric={"actual": actual, "expected": expected_rows, "ratio": round(ratio, 4)},
    )


def check_timestamp_monotonic(df: pd.DataFrame) -> CheckResult:
    if "Timestamp" not in df.columns:
        return CheckResult("timestamp_monotonic", "FAIL", "missing Timestamp column")
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    if ts.isna().any():
        return CheckResult(
            "timestamp_monotonic", "FAIL",
            f"{int(ts.isna().sum())} unparseable timestamps",
            metric={"unparseable": int(ts.isna().sum())},
        )
    diffs = ts.diff().dropna().dt.total_seconds()
    backward = int((diffs <= 0).sum())
    status = "PASS" if backward == 0 else "FAIL"
    return CheckResult(
        name="timestamp_monotonic",
        status=status,
        detail=f"{backward} non-increasing transitions",
        metric={"non_increasing": backward},
    )


def check_timestamp_spacing(df: pd.DataFrame, target_dt: float) -> CheckResult:
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    diffs = ts.diff().dropna().dt.total_seconds().to_numpy()
    if diffs.size == 0:
        return CheckResult("timestamp_spacing", "FAIL", "no diffs")
    median = float(np.median(diffs))
    max_gap = float(np.max(diffs))
    in_band = (median >= target_dt * 0.75) and (median <= target_dt * 1.25)
    gap_ok = max_gap < target_dt * 3.0
    if in_band and gap_ok:
        status = "PASS"
    elif in_band or gap_ok:
        status = "WARN"
    else:
        status = "FAIL"
    return CheckResult(
        name="timestamp_spacing",
        status=status,
        detail=f"median={median:.3f}s target={target_dt}s max_gap={max_gap:.2f}s",
        metric={"median_dt": median, "max_gap": max_gap, "target": target_dt},
    )


def check_header_schema(df: pd.DataFrame) -> CheckResult:
    cols = set(df.columns)
    missing: list[str] = []

    if "Timestamp" not in cols:
        missing.append("Timestamp")
    for c in EXPECTED_CONTAINERS:
        for suffix in ("_stressType", "_stepStress"):
            if f"{c}{suffix}" not in cols:
                missing.append(f"{c}{suffix}")

    has_eth0 = any(ETH0_TAG in c for c in cols)
    has_eth1 = any(ETH1_TAG in c for c in cols)
    if not has_eth0:
        missing.append("any eth0 network column")
    if not has_eth1:
        missing.append("any eth1 network column")

    for field in INFLUX_FIELDS:
        for pci in PCI_RANGE:
            key = f"PCI-{pci}_RNTI-4601_{field}"
            if key not in cols:
                missing.append(key)
                break

    if not missing:
        return CheckResult("header_schema", "PASS", "all expected columns present")
    return CheckResult(
        name="header_schema",
        status="FAIL",
        detail=f"missing {len(missing)} expected columns (first 10: {missing[:10]})",
        metric={"missing": missing[:50]},
    )


def check_normal_label_purity(df: pd.DataFrame) -> CheckResult:
    stype_cols = _stress_type_columns(df)
    sstep_cols = _stress_step_columns(df)
    nonzero = 0
    for c in stype_cols + sstep_cols:
        nonzero += int((_coerce_numeric(df[c]).fillna(0) != 0).sum())
    status = "PASS" if nonzero == 0 else "FAIL"
    return CheckResult(
        name="normal_label_purity",
        status=status,
        detail=f"{nonzero} nonzero stress cells across all rows",
        metric={"nonzero_cells": nonzero},
    )


def check_anomaly_label_coverage(df: pd.DataFrame) -> CheckResult:
    seen: set[tuple[str, int]] = set()
    expected: set[tuple[str, int]] = set()
    for c in EXPECTED_CONTAINERS:
        for t in (1, 2, 3):
            expected.add((c, t))
    for c in EXPECTED_CONTAINERS:
        col = f"{c}_stressType"
        if col not in df.columns:
            continue
        values = _coerce_numeric(df[col]).dropna().astype(int).unique().tolist()
        for v in values:
            if v in (1, 2, 3):
                seen.add((c, v))
    missing = sorted(expected - seen)
    if not missing:
        return CheckResult("anomaly_label_coverage", "PASS",
                           "all 27 (container, stress_type) pairs present")
    if len(missing) <= 3:
        return CheckResult(
            name="anomaly_label_coverage",
            status="WARN",
            detail=f"missing pairs: {missing}",
            metric={"missing": [list(p) for p in missing]},
        )
    return CheckResult(
        name="anomaly_label_coverage",
        status="FAIL",
        detail=f"{len(missing)} pairs missing (first 5: {missing[:5]})",
        metric={"missing": [list(p) for p in missing]},
    )


def check_anomaly_density(df: pd.DataFrame) -> CheckResult:
    if "any_stress_active" not in df.columns:
        flags = pd.Series(0, index=df.index)
        for c in EXPECTED_CONTAINERS:
            col = f"{c}_stressType"
            if col in df.columns:
                flags = flags | (_coerce_numeric(df[col]).fillna(0) != 0).astype(int)
    else:
        flags = _coerce_numeric(df["any_stress_active"]).fillna(0).astype(int)
    density = float(flags.mean()) if len(flags) else 0.0
    if 0.20 <= density <= 0.40:
        status = "PASS"
    elif 0.10 <= density < 0.20 or 0.40 < density <= 0.50:
        status = "WARN"
    else:
        status = "FAIL"
    return CheckResult(
        name="anomaly_density",
        status=status,
        detail=f"density={density:.1%} (target 30% = 3min stress / 10min episode)",
        metric={"density": round(density, 4)},
    )


def check_anomaly_value_range(df: pd.DataFrame) -> CheckResult:
    """When _stressType != 0, _stepStress must be in [85, 95] (current intensity range)."""
    bad = 0
    samples: list[str] = []
    for c in EXPECTED_CONTAINERS:
        tcol, scol = f"{c}_stressType", f"{c}_stepStress"
        if tcol not in df.columns or scol not in df.columns:
            continue
        active = _coerce_numeric(df[tcol]).fillna(0) != 0
        step = _coerce_numeric(df[scol]).fillna(-1)
        out_of_range = active & ~((step >= 85) & (step <= 95))
        bad += int(out_of_range.sum())
        if out_of_range.any() and len(samples) < 3:
            samples.append(f"{c}:{int(step[out_of_range].iloc[0])}")
    status = "PASS" if bad == 0 else ("WARN" if bad < 10 else "FAIL")
    return CheckResult(
        name="anomaly_value_range",
        status=status,
        detail=f"{bad} rows where active stress has intensity outside [85,95] {samples}",
        metric={"out_of_range_rows": bad},
    )


def check_prometheus_liveness(df: pd.DataFrame) -> CheckResult:
    """For each container, at least one CPU/MEM/FS column must be ≥80% numeric & nonempty."""
    weak: list[str] = []
    for c in EXPECTED_CONTAINERS:
        ok = False
        for substr in (PROM_CPU_SUBSTR, PROM_MEM_SUBSTR, PROM_FS_SUBSTR):
            cols = _container_columns(df, c, substr)
            if not cols:
                continue
            series = _coerce_numeric(df[cols[0]])
            ratio = series.notna().mean() if len(series) else 0.0
            if ratio >= 0.80:
                ok = True
                break
        if not ok:
            weak.append(c)
    if not weak:
        return CheckResult("prometheus_liveness", "PASS",
                           "all containers have ≥80% numeric core metrics")
    status = "WARN" if len(weak) <= 2 else "FAIL"
    return CheckResult(
        name="prometheus_liveness",
        status=status,
        detail=f"weak telemetry for: {weak}",
        metric={"weak_containers": weak},
    )


def check_eth1_nonzero(df: pd.DataFrame) -> CheckResult:
    cols = _eth_columns(df, ETH1_TAG)
    if not cols:
        return CheckResult("eth1_nonzero", "FAIL",
                           "no eth1 columns present — promCadvisor.txt not regenerated?")
    means = []
    for c in cols:
        vals = _coerce_numeric(df[c]).fillna(0)
        means.append(float(vals.mean()))
    nonzero_share = float(np.mean([m > 0 for m in means]))
    if nonzero_share >= 0.5:
        status = "PASS"
    elif nonzero_share > 0:
        status = "WARN"
    else:
        status = "FAIL"
    return CheckResult(
        name="eth1_nonzero",
        status=status,
        detail=f"{nonzero_share:.0%} of eth1 columns have nonzero mean ({len(cols)} total)",
        metric={"eth1_columns": len(cols), "nonzero_share": round(nonzero_share, 4)},
    )


def check_influx_liveness(df: pd.DataFrame) -> CheckResult:
    sentinels = ["PCI-1_RNTI-4601_dl_brate",
                 "PCI-1_RNTI-4601_ul_brate",
                 "PCI-1_RNTI-4601_cqi"]
    results = {}
    weak: list[str] = []
    for s in sentinels:
        if s not in df.columns:
            weak.append(f"{s} (missing)")
            continue
        ratio = float((_coerce_numeric(df[s]).fillna(0) != 0).mean())
        results[s] = round(ratio, 4)
        if ratio < 0.80:
            weak.append(f"{s} ({ratio:.1%})")
    if not weak:
        return CheckResult("influx_liveness", "PASS",
                           f"all sentinels ≥80% nonzero: {results}")
    status = "FAIL" if len(weak) == len(sentinels) else "WARN"
    return CheckResult(
        name="influx_liveness",
        status=status,
        detail=f"weak: {weak}",
        metric=results,
    )


def check_traffic_cycle_coverage(df: pd.DataFrame) -> CheckResult:
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    if ts.isna().all():
        return CheckResult("traffic_cycle_coverage", "FAIL", "no parseable timestamps")
    secs = (ts - ts.iloc[0]).dt.total_seconds()
    bins = (secs // TRAFFIC_BIN_SECONDS).astype("Int64").nunique(dropna=True)
    distinct = int(bins)
    if distinct >= 24:
        status = "PASS"
    elif distinct >= 12:
        status = "WARN"
    else:
        status = "FAIL"
    return CheckResult(
        name="traffic_cycle_coverage",
        status=status,
        detail=f"{distinct} distinct 15-min bins (cycle has 25 steps)",
        metric={"distinct_bins": distinct},
    )


def check_visible_stress_impact(df: pd.DataFrame) -> CheckResult:
    """For each (container, stress_type) with rows labelled, mean(target-metric)
    during stress must differ from cooldown baseline by ≥ 2x in the right direction.

    CPU type=1 -> *cpu_user*    expect >= 2x baseline
    MEM type=2 -> *memory_usage* expect >= 1.5x baseline
    NET type=3 -> *eth0*receive* expect <= 0.66x baseline (packet loss drops RX)
    """
    target_map = {
        1: (PROM_CPU_SUBSTR, "above", 2.0),
        2: (PROM_MEM_SUBSTR, "above", 1.5),
        3: (PROM_NET_RX_SUBSTR + ".*" + ETH0_TAG, "below", 0.66),
    }
    weak: list[str] = []
    passed = 0
    total = 0

    for c in EXPECTED_CONTAINERS:
        tcol = f"{c}_stressType"
        if tcol not in df.columns:
            continue
        types = _coerce_numeric(df[tcol]).fillna(0).astype(int)
        for stress_type, (substr, direction, factor) in target_map.items():
            if stress_type == 3:
                substr_pat = re.compile(
                    re.escape(PROM_NET_RX_SUBSTR) + ".*" + re.escape(ETH0_TAG)
                )
                cols = [col for col in df.columns
                        if f'name="{c}"' in col and substr_pat.search(col)]
            else:
                cols = _container_columns(df, c, substr)
            if not cols:
                continue
            mask_active = types == stress_type
            mask_baseline = types == 0
            if mask_active.sum() < 5 or mask_baseline.sum() < 30:
                continue
            total += 1
            metric = _coerce_numeric(df[cols[0]]).fillna(0)
            active_mean = float(metric[mask_active].mean())
            baseline_mean = float(metric[mask_baseline].mean())
            if baseline_mean == 0 and active_mean == 0:
                weak.append(f"{c}:type{stress_type} (both means zero)")
                continue
            ratio = active_mean / baseline_mean if baseline_mean else float("inf")
            ok = (direction == "above" and ratio >= factor) or \
                 (direction == "below" and ratio <= factor)
            if ok:
                passed += 1
            else:
                weak.append(f"{c}:type{stress_type} ratio={ratio:.2f} (need {direction} {factor})")

    if total == 0:
        return CheckResult("visible_stress_impact", "WARN",
                           "no labelled stress windows found to evaluate")
    share = passed / total
    if share >= 0.80:
        status = "PASS"
    elif share >= 0.50:
        status = "WARN"
    else:
        status = "FAIL"
    return CheckResult(
        name="visible_stress_impact",
        status=status,
        detail=f"{passed}/{total} pairs show expected metric impact",
        metric={"passed": passed, "total": total, "weak": weak[:8]},
    )


def check_drift_sanity(df: pd.DataFrame) -> CheckResult:
    candidates = [c for c in df.columns if "dl_brate" in c]
    if not candidates:
        return CheckResult("drift_sanity", "WARN", "no dl_brate column found")
    series = _coerce_numeric(df[candidates[0]]).fillna(0)
    std = float(series.std())
    if std > 1.0:
        return CheckResult("drift_sanity", "PASS",
                           f"dl_brate std={std:.2f} (>1)",
                           metric={"dl_brate_std": std})
    return CheckResult("drift_sanity", "FAIL",
                       f"dl_brate std={std:.2f} (≤1) — InfluxDB likely returning constants",
                       metric={"dl_brate_std": std})


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_validation(
    csv_path: str,
    mode: str,
    expected_duration_sec: float,
    fetch_interval: float,
) -> tuple[list[CheckResult], dict[str, int]]:
    log.info(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    log.info(f"Loaded {len(df)} rows x {len(df.columns)} cols")

    expected_rows = int(expected_duration_sec / fetch_interval)
    checks: list[CheckResult] = []

    checks.append(check_row_count(df, expected_rows))
    checks.append(check_timestamp_monotonic(df))
    checks.append(check_timestamp_spacing(df, fetch_interval))
    checks.append(check_header_schema(df))
    checks.append(check_prometheus_liveness(df))
    checks.append(check_eth1_nonzero(df))
    checks.append(check_influx_liveness(df))
    checks.append(check_drift_sanity(df))

    if mode == "NORMAL":
        checks.append(check_normal_label_purity(df))
        checks.append(check_traffic_cycle_coverage(df))
    else:
        checks.append(check_anomaly_label_coverage(df))
        checks.append(check_anomaly_density(df))
        checks.append(check_anomaly_value_range(df))
        checks.append(check_visible_stress_impact(df))

    summary = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for c in checks:
        summary[c.status] = summary.get(c.status, 0) + 1
    return checks, summary


def write_report(csv_path: str, mode: str, checks: list[CheckResult],
                 summary: dict[str, int]) -> str:
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    base = os.path.splitext(os.path.basename(csv_path))[0]
    report_path = os.path.join(out_dir, f"validation_{base}.json")
    payload = {
        "csv": os.path.abspath(csv_path),
        "mode": mode,
        "summary": summary,
        "checks": [asdict(c) for c in checks],
    }
    with open(report_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return report_path


def print_summary(checks: list[CheckResult], summary: dict[str, int]) -> None:
    width = max(len(c.name) for c in checks) + 2
    print()
    print("=" * 72)
    print(f"{'CHECK':<{width}} {'STATUS':<6}  DETAIL")
    print("-" * 72)
    for c in checks:
        print(f"{c.name:<{width}} {c.status:<6}  {c.detail}")
    print("-" * 72)
    print(f"PASS={summary.get('PASS', 0)}  WARN={summary.get('WARN', 0)}  FAIL={summary.get('FAIL', 0)}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate O-RAN FALCON collected CSV")
    p.add_argument("--csv", required=True, help="Path to the CSV produced by scrapper")
    p.add_argument("--expected-mode", required=True, choices=["NORMAL", "ANOMALY"],
                   help="Which phase this CSV represents")
    p.add_argument("--expected-duration-sec", type=float, required=True,
                   help="Expected collection wall-clock duration in seconds")
    p.add_argument("--fetch-interval", type=float, default=1.0,
                   help="Scrapper FETCH_INTERVAL (default 1.0)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.csv):
        log.error(f"CSV not found: {args.csv}")
        return 2

    checks, summary = run_validation(
        csv_path=args.csv,
        mode=args.expected_mode,
        expected_duration_sec=args.expected_duration_sec,
        fetch_interval=args.fetch_interval,
    )
    report_path = write_report(args.csv, args.expected_mode, checks, summary)
    print_summary(checks, summary)
    log.info(f"Report written to {report_path}")
    return 0 if summary.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
