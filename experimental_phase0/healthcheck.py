#!/usr/bin/env python3
"""
O-RAN srsRAN FALCON - Live pipeline watchdog
==============================================
Runs alongside test_traffic.py + scrapper_prometheus_backup.py during a
6.5h+ data collection run. Every tick (default 10s) it:

  1. Checks `iperf -s` server inside open5gs_5gc. AUTO-RESTARTS if dead.
  2. Counts live `iperf -c` clients inside each UE.
  3. Queries InfluxDB for the newest ue_info write and the mean ul_brate
     over the last 60s. Flags stale writes (>60s old) or zero-throughput.
  4. Queries Prometheus for cAdvisor scrape recency + a sentinel cpu_user
     metric per target container.
  5. Confirms the scrapper_prometheus_backup.py process is alive.
  6. Prints a one-line PASS/WARN/FAIL status with ANSI colors. Writes the
     same line + per-component detail to a log file. Beeps on any FAIL
     transition (so it is *visible* when something breaks at 03:00 AM).

Usage:
    export INFLUXDB_TOKEN=...
    python3 healthcheck.py [--interval 10] [--duration 24000] \
        [--log-file data_fs_plane/healthcheck.log]

Exit: Ctrl+C. The watchdog never restarts test_traffic.py or scrapper —
it only restarts the iperf SERVER (cheap, side-effect free) and reports
the rest so the operator can intervene.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


ANSI_GREEN = "\033[1;32m"
ANSI_YELLOW = "\033[1;33m"
ANSI_RED = "\033[1;31m"
ANSI_CYAN = "\033[1;36m"
ANSI_RESET = "\033[0m"
ANSI_BELL = "\a"

OPEN5GS_CONTAINER = "open5gs_5gc"
UE_CONTAINERS = ["srsue0", "srsue1", "srsue2", "srsue3", "srsue4", "srsue5"]
TARGET_CONTAINERS = [
    "srscu0", "srscu1", "srscu2",
    "srsdu0", "srsdu1", "srsdu2", "srsdu3", "srsdu4", "srsdu5",
]
IPERF_SERVER_CMD = "iperf -s -u -B 10.45.1.1"

INFLUX_URL = "http://localhost:8086"
INFLUX_ORG = "srs"
INFLUX_BUCKET = "srsran"
PROM_URL = "http://localhost:9090"

STALE_INFLUX_SEC = 60         # InfluxDB write must be newer than this
STALE_PROM_SEC = 60           # Prometheus sample must be newer than this
MIN_UL_BRATE_BPS = 1_000_000  # 1 Mbps aggregate to call traffic "flowing"


# ---------------------------------------------------------------------------
@dataclass
class TickStatus:
    """Per-tick summary used for rendering + change detection."""
    ts: str = ""
    iperf_server_up: bool = False
    iperf_server_restarted: bool = False
    ue_iperf_count: int = 0
    ue_iperf_missing: list[str] = field(default_factory=list)
    influx_age_sec: float = -1.0
    influx_ul_brate_mbps: float = -1.0
    influx_active_pcis: int = 0
    prom_cadvisor_age_sec: float = -1.0
    prom_weak_containers: list[str] = field(default_factory=list)
    scrapper_alive: bool = False
    overall: str = "FAIL"  # PASS | WARN | FAIL


# ---------------------------------------------------------------------------
def _shell(cmd: str, timeout: int = 5) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as exc:
        return 1, "", str(exc)


def _http_post(url: str, body: bytes, headers: dict[str, str], timeout: int = 5) -> str:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _http_get(url: str, timeout: int = 5) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
def check_iperf_server() -> tuple[bool, bool]:
    """Returns (is_up_after_check, was_restarted)."""
    rc, out, _ = _shell(
        f"docker exec {OPEN5GS_CONTAINER} pgrep -c iperf", timeout=4,
    )
    try:
        count = int(out) if out else 0
    except ValueError:
        count = 0
    if count > 0:
        return True, False

    # restart server (detached one-shot)
    _shell(
        f"docker exec -d {OPEN5GS_CONTAINER} {IPERF_SERVER_CMD}", timeout=4,
    )
    time.sleep(1.5)
    rc, out, _ = _shell(
        f"docker exec {OPEN5GS_CONTAINER} pgrep -c iperf", timeout=4,
    )
    try:
        count = int(out) if out else 0
    except ValueError:
        count = 0
    return count > 0, True


def check_ue_iperf() -> tuple[int, list[str]]:
    alive = 0
    missing: list[str] = []
    for ue in UE_CONTAINERS:
        rc, out, _ = _shell(
            f"docker exec {ue} pgrep -c iperf", timeout=4,
        )
        try:
            cnt = int(out) if out else 0
        except ValueError:
            cnt = 0
        if cnt > 0:
            alive += 1
        else:
            missing.append(ue)
    return alive, missing


def _influx_query(flux: str, token: str, timeout: int = 6) -> str:
    url = f"{INFLUX_URL}/api/v2/query?org={urllib.parse.quote(INFLUX_ORG)}"
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/vnd.flux",
        "Accept": "application/csv",
    }
    return _http_post(url, flux.encode("utf-8"), headers, timeout=timeout)


def check_influx(token: str) -> tuple[float, float, int]:
    """Return (age_seconds_of_newest_write, mean_ul_brate_mbps_60s, active_pci_count)."""
    age = -1.0
    mean_brate_mbps = -1.0
    active_pcis = 0

    flux_recent = (
        f'from(bucket:"{INFLUX_BUCKET}") '
        f'|> range(start:-2m) '
        f'|> filter(fn: (r) => r["_measurement"] == "ue_info") '
        f'|> filter(fn: (r) => r["_field"] == "ul_brate") '
        f'|> last()'
    )
    try:
        csv_data = _influx_query(flux_recent, token, timeout=5)
        newest_ts: Optional[datetime] = None
        per_pci_brate: dict[str, float] = {}
        time_idx, value_idx, pci_idx = -1, -1, -1
        for line in csv_data.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if time_idx < 0 and "_time" in parts:
                time_idx = parts.index("_time")
                value_idx = parts.index("_value") if "_value" in parts else -1
                pci_idx = parts.index("pci") if "pci" in parts else -1
                continue
            if time_idx < 0 or value_idx < 0:
                continue
            try:
                ts = datetime.fromisoformat(parts[time_idx].replace("Z", "+00:00"))
                val = float(parts[value_idx])
            except (ValueError, IndexError):
                continue
            pci = parts[pci_idx] if pci_idx >= 0 and len(parts) > pci_idx else ""
            if newest_ts is None or ts > newest_ts:
                newest_ts = ts
            if pci:
                per_pci_brate[pci] = max(per_pci_brate.get(pci, 0.0), val)
        if newest_ts is not None:
            age = (datetime.now(timezone.utc) - newest_ts).total_seconds()
        if per_pci_brate:
            active_pcis = sum(1 for v in per_pci_brate.values() if v > 0)
    except Exception:
        pass

    flux_mean = (
        f'from(bucket:"{INFLUX_BUCKET}") '
        f'|> range(start:-60s) '
        f'|> filter(fn: (r) => r["_measurement"] == "ue_info") '
        f'|> filter(fn: (r) => r["_field"] == "ul_brate") '
        f'|> mean()'
    )
    try:
        csv_data = _influx_query(flux_mean, token, timeout=5)
        # The mean() result schema omits _time. Locate _value column from the
        # CSV header so we don't depend on a hardcoded index.
        value_idx = -1
        total_bps = 0.0
        for line in csv_data.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if value_idx < 0 and "_value" in parts:
                value_idx = parts.index("_value")
                continue
            if value_idx < 0 or len(parts) <= value_idx:
                continue
            try:
                total_bps += float(parts[value_idx])
            except (ValueError, IndexError):
                continue
        mean_brate_mbps = total_bps / 1e6
    except Exception:
        pass

    return age, mean_brate_mbps, active_pcis


def check_prom(target_containers: list[str]) -> tuple[float, list[str]]:
    """Return (newest cadvisor scrape age in seconds, weak containers)."""
    age = -1.0
    weak: list[str] = []

    # Newest scrape: instant query 'time() - timestamp(up{job="cadvisor"})'
    try:
        q = urllib.parse.quote('time() - timestamp(up{job="cadvisor"})')
        body = _http_get(f'{PROM_URL}/api/v1/query?query={q}', timeout=5)
        data = json.loads(body)
        if data.get("status") == "success":
            results = data["data"]["result"]
            if results:
                age = float(results[0]["value"][1])
    except Exception:
        pass

    # Each target container: cpu_user counter rate must be present (any value)
    for cname in target_containers:
        try:
            q = urllib.parse.quote(
                f'sum(rate(container_cpu_user_seconds_total{{name="{cname}"}}[30s]))'
            )
            body = _http_get(f'{PROM_URL}/api/v1/query?query={q}', timeout=4)
            data = json.loads(body)
            if data.get("status") != "success":
                weak.append(cname)
                continue
            results = data["data"]["result"]
            if not results:
                weak.append(cname)
                continue
            val = float(results[0]["value"][1])
            if val < 0:
                weak.append(cname)
        except Exception:
            weak.append(cname)

    return age, weak


def check_scrapper() -> bool:
    # Match python interpreter running scrapper_prometheus_backup.py — excludes
    # incidental matches from grep / shell command lines or this watchdog itself.
    rc, out, _ = _shell(
        "pgrep -af 'python.* scrapper_prometheus_backup\\.py' | grep -v healthcheck | grep -v grep",
        timeout=3,
    )
    return bool(out.strip())


# ---------------------------------------------------------------------------
def evaluate(status: TickStatus) -> str:
    fails: list[str] = []
    warns: list[str] = []

    if not status.iperf_server_up:
        fails.append("iperf-srv")
    if status.iperf_server_restarted:
        warns.append("iperf-srv RESTARTED")
    if status.ue_iperf_count < 6:
        if status.ue_iperf_count == 0:
            fails.append("no UE iperf")
        else:
            warns.append(f"{6 - status.ue_iperf_count} UE iperf missing")
    if status.influx_age_sec < 0 or status.influx_age_sec > STALE_INFLUX_SEC:
        fails.append(f"Influx stale ({status.influx_age_sec:.0f}s)")
    if status.influx_ul_brate_mbps < 1.0 and status.influx_age_sec >= 0:
        warns.append(f"Influx ul_brate low ({status.influx_ul_brate_mbps:.2f} Mbps)")
    if status.influx_active_pcis < 6 and status.influx_active_pcis >= 0:
        warns.append(f"Influx PCIs active {status.influx_active_pcis}/6")
    if status.prom_cadvisor_age_sec < 0 or status.prom_cadvisor_age_sec > STALE_PROM_SEC:
        fails.append(f"Prom stale ({status.prom_cadvisor_age_sec:.0f}s)")
    if status.prom_weak_containers:
        warns.append(f"Prom weak: {len(status.prom_weak_containers)}")
    if not status.scrapper_alive:
        fails.append("scrapper DEAD")

    if fails:
        return "FAIL"
    if warns:
        return "WARN"
    return "PASS"


def render(status: TickStatus, prev_overall: str, no_color: bool, log_fp) -> None:
    overall_color = {
        "PASS": ANSI_GREEN,
        "WARN": ANSI_YELLOW,
        "FAIL": ANSI_RED,
    }.get(status.overall, ANSI_RESET)

    short = (
        f"srv={'UP' if status.iperf_server_up else 'DOWN'}"
        f"{'(RESTART)' if status.iperf_server_restarted else ''} "
        f"ue_iperf={status.ue_iperf_count}/6 "
        f"influx_age={status.influx_age_sec:.0f}s "
        f"ul_mbps={status.influx_ul_brate_mbps:.2f} "
        f"active_pci={status.influx_active_pcis}/6 "
        f"prom_age={status.prom_cadvisor_age_sec:.0f}s "
        f"weak_prom={len(status.prom_weak_containers)} "
        f"scrapper={'YES' if status.scrapper_alive else 'NO'}"
    )

    bell = ANSI_BELL if status.overall == "FAIL" and prev_overall != "FAIL" else ""
    if no_color:
        line = f"[{status.ts}] {status.overall:<4} {short}"
    else:
        line = f"[{ANSI_CYAN}{status.ts}{ANSI_RESET}] {overall_color}{status.overall:<4}{ANSI_RESET} {short}"

    sys.stdout.write(bell + line + "\n")
    sys.stdout.flush()

    log_line = f"[{status.ts}] {status.overall} {short}"
    log_fp.write(log_line + "\n")
    if status.ue_iperf_missing:
        log_fp.write(f"  ue_iperf_missing: {status.ue_iperf_missing}\n")
    if status.prom_weak_containers:
        log_fp.write(f"  prom_weak: {status.prom_weak_containers}\n")
    log_fp.flush()


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="O-RAN FALCON live pipeline watchdog")
    p.add_argument("--interval", type=float, default=10.0,
                   help="Seconds between ticks (default 10)")
    p.add_argument("--duration", type=float, default=24 * 3600,
                   help="Total seconds to run before self-exit (default 24h)")
    p.add_argument("--log-file", default="data_fs_plane/healthcheck.log",
                   help="Path to append-only log file")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("INFLUXDB_TOKEN")
    if not token:
        print(f"{ANSI_RED}ERROR: INFLUXDB_TOKEN env var not set.{ANSI_RESET}",
              file=sys.stderr)
        return 2

    no_color = args.no_color or not sys.stdout.isatty()
    log_path = args.log_file
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    log_fp = open(log_path, "a", buffering=1)
    log_fp.write(f"\n===== healthcheck started at {datetime.now().isoformat()} =====\n")

    def _exit_handler(signum, frame):
        log_fp.write(f"===== healthcheck stopped at {datetime.now().isoformat()} =====\n")
        log_fp.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _exit_handler)
    signal.signal(signal.SIGTERM, _exit_handler)

    if shutil.which("docker") is None:
        print(f"{ANSI_RED}ERROR: docker CLI not found in PATH{ANSI_RESET}",
              file=sys.stderr)
        return 2

    print(f"{ANSI_CYAN}Watchdog interval={args.interval}s duration={args.duration:.0f}s "
          f"log={log_path}{ANSI_RESET}")
    print(f"{ANSI_CYAN}Sentinels: STALE_INFLUX_SEC={STALE_INFLUX_SEC} "
          f"STALE_PROM_SEC={STALE_PROM_SEC} "
          f"MIN_UL_BRATE_BPS={MIN_UL_BRATE_BPS}{ANSI_RESET}")

    deadline = time.time() + args.duration
    prev_overall = ""

    while time.time() < deadline:
        loop_start = time.time()
        status = TickStatus()
        status.ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        status.iperf_server_up, status.iperf_server_restarted = check_iperf_server()
        status.ue_iperf_count, status.ue_iperf_missing = check_ue_iperf()
        status.influx_age_sec, status.influx_ul_brate_mbps, status.influx_active_pcis = \
            check_influx(token)
        status.prom_cadvisor_age_sec, status.prom_weak_containers = \
            check_prom(TARGET_CONTAINERS)
        status.scrapper_alive = check_scrapper()
        status.overall = evaluate(status)

        render(status, prev_overall, no_color, log_fp)
        prev_overall = status.overall

        elapsed = time.time() - loop_start
        sleep_for = max(0.5, args.interval - elapsed)
        time.sleep(sleep_for)

    log_fp.write(f"===== healthcheck completed normally at {datetime.now().isoformat()} =====\n")
    log_fp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
