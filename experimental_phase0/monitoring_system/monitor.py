#!/usr/bin/env python3
"""O-RAN live stress impact monitor.

Usage:
    cd experimental_phase0
    export INFLUXDB_TOKEN=<token>
    python3 monitoring_system/monitor.py \
        --baseline-csv data_fs_plane/train_normal_th3.csv \
        --anomaly-csv  data_fs_plane/test_anomaly_th3.csv \
        --interval 30 \
        --log data_fs_plane/impact_log.csv
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime

from rich import box
from rich.console import Console
from rich.table import Table

# Allow running from experimental_phase0/ dir
sys.path.insert(0, os.path.dirname(__file__))

from baseline_stats import BaselineStats
from impact_quantifier import ImpactQuantifier, ImpactResult, KPIImpact
from live_fetcher import LiveFetcher
from stress_tracker import StressTracker

console = Console()

SEVERITY_STYLE = {
    "CRITICAL": "bold red",
    "WARNING":  "yellow",
    "INFO":     "cyan",
    "NORMAL":   "dim",
}

# KPI substrings to include in the table (infra + radio)
_INFRA = ("cpu_pct", "memory_bytes", "eth0_tx_bytes", "eth0_rx_bytes")
_RADIO = ("ul_brate", "bsr", "dl_brate", "harq_error_rate", "cqi", "pucch_snr_db", "pusch_snr_db")
_SHOW  = _INFRA + _RADIO

LOG_FIELDS = [
    "timestamp", "container", "node_type", "stress_type", "intensity",
    "kpi", "current", "baseline_mean", "z_score", "pct_deviation",
    "severity", "expected",
]


def _is_infra(kpi: str) -> bool:
    return any(k in kpi for k in _INFRA)


def _is_relevant(kpi: str, affected_pcis: list[int]) -> bool:
    """Show infra KPIs for the stressed container + radio KPIs for its PCIs."""
    if _is_infra(kpi):
        return True
    # radio KPI: only show if PCI belongs to the stressed container
    for pci in affected_pcis:
        if f"PCI-{pci}_" in kpi:
            return True
    return False


def _fmt(kpi: str, value: float) -> str:
    if "brate" in kpi:
        return f"{value / 1e6:.2f} Mbps"
    if "memory_bytes" in kpi:
        return f"{value / 1e6:.1f} MB"
    if "eth0_" in kpi and "bytes" in kpi:
        return f"{value / 1e6:.2f} MB/s"
    if "cpu_pct" in kpi:
        return f"{value:.1f}%"
    if "harq_error" in kpi:
        return f"{value:.3f}"
    if "snr" in kpi:
        return f"{value:.1f} dB"
    if "bsr" in kpi:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _layer(kpi: str) -> str:
    return "[INFRA]" if _is_infra(kpi) else "[RADIO]"


def _print_table(results: list[ImpactResult], ts: str):
    console.clear()
    if not results:
        console.print(f"[{ts}]  No active stress — monitoring...", style="dim")
        return

    for res in results:
        title = (
            f"[bold]{ts}[/bold]  "
            f"[cyan]{res.container}[/cyan] ([yellow]{res.node_type}[/yellow]) "
            f"→ [bold magenta]{res.stress_name}[/bold magenta]  "
            f"intensity={res.intensity:.0f}  PCIs={res.affected_pcis}"
        )
        t = Table(title=title, box=box.SIMPLE_HEAD, expand=False, title_justify="left")
        t.add_column("Layer+KPI",     style="bold", no_wrap=True, min_width=42)
        t.add_column("Current",       justify="right")
        t.add_column("Baseline",      justify="right")
        t.add_column("Z",             justify="right")
        t.add_column("%Dev",          justify="right")
        t.add_column("Severity",      justify="center")
        t.add_column("Expected?",     justify="center")

        # Collect rows: always show infra KPIs for this container,
        # show radio KPIs only when non-NORMAL
        rows = []
        for kpi, imp in res.kpi_impacts.items():
            if not any(k in kpi for k in _SHOW):
                continue
            if not _is_relevant(kpi, res.affected_pcis):
                continue
            # Always include infra; only include radio if noteworthy
            if _is_infra(kpi) or imp.severity != "NORMAL":
                rows.append((kpi, imp))

        # Sort: infra first, then by |z| descending
        rows.sort(key=lambda x: (0 if _is_infra(x[0]) else 1, -abs(x[1].z_score)))

        for kpi, imp in rows[:20]:
            style  = SEVERITY_STYLE.get(imp.severity, "")
            z_str  = f"{imp.z_score:+.2f}"
            pct    = f"{imp.pct_deviation:+.1f}%"
            exp    = "✓" if imp.expected else "✗"
            t.add_row(
                f"{_layer(kpi)} {kpi}",
                _fmt(kpi, imp.current),
                _fmt(kpi, imp.baseline_mean),
                z_str, pct, imp.severity, exp,
                style=style,
            )

        console.print(t)
        console.print()


def _log_csv(results: list[ImpactResult], log_path: str, ts: str):
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        for res in results:
            for kpi, imp in res.kpi_impacts.items():
                if not any(k in kpi for k in _SHOW):
                    continue
                writer.writerow({
                    "timestamp":     ts,
                    "container":     res.container,
                    "node_type":     res.node_type,
                    "stress_type":   res.stress_name,
                    "intensity":     res.intensity,
                    "kpi":           kpi,
                    "current":       imp.current,
                    "baseline_mean": imp.baseline_mean,
                    "z_score":       imp.z_score,
                    "pct_deviation": imp.pct_deviation,
                    "severity":      imp.severity,
                    "expected":      imp.expected,
                })


def main():
    parser = argparse.ArgumentParser(description="O-RAN live stress impact monitor")
    parser.add_argument("--baseline-csv", required=True,
                        help="Path to train_normal_th3.csv")
    parser.add_argument("--anomaly-csv",  required=True,
                        help="Path to test_anomaly_th3.csv (tailed for active stress labels)")
    parser.add_argument("--interval",     type=int, default=30,
                        help="Fetch + refresh interval in seconds (default: 30)")
    parser.add_argument("--log",          default=None,
                        help="Append impact rows to this CSV (optional)")
    args = parser.parse_args()

    console.print("[bold green]O-RAN Stress Impact Monitor[/bold green]")
    console.print(f"Loading baseline from [cyan]{args.baseline_csv}[/cyan] ...")
    baseline = BaselineStats(args.baseline_csv)
    console.print(f"  [green]✓[/green] {len(baseline.stats)} KPI baselines loaded")

    console.print("Resolving container IDs + building Prometheus queries ...")
    fetcher    = LiveFetcher()
    tracker    = StressTracker()
    quantifier = ImpactQuantifier()

    console.print(f"[bold green]Monitoring started[/bold green]  interval={args.interval}s")
    if args.log:
        console.print(f"Impact log → [cyan]{args.log}[/cyan]")
    console.print()

    while True:
        tick = time.time()
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        live     = fetcher.fetch(window_seconds=args.interval)
        stresses = tracker.get_active(args.anomaly_csv)
        impacts  = quantifier.quantify(live, stresses, baseline)

        _print_table(impacts, ts)

        if args.log and impacts:
            _log_csv(impacts, args.log, ts)

        elapsed   = time.time() - tick
        sleep_for = max(0.0, args.interval - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
