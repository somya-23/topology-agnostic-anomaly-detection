#!/usr/bin/env python3
"""
O-RAN srsRAN FALCON - Filesystem Data Collection Pipeline (0.1s resolution)
============================================================================
Reads ALL container metrics directly from kernel cgroup files + /proc — no
Prometheus HTTP overhead. Achieves true 0.1s collection intervals.

Pipeline:
  1. Collects NORMAL baseline data for --normal-hours (default 5h)
  2. Auto-transitions to ANOMALY mode
  3. Injects every fault type into every target container --fault-reps times (default 2)
  4. Monitors UE containers for RRC Release throughout
  5. Verifies stress is actually applied before labelling rows

Usage:
    python3 scrapper_file_system.py [--normal-hours 5] [--fault-reps 2] [--interval 0.1]
"""

import asyncio
import time
from datetime import datetime
import logging
import os
import csv
import json
import subprocess
import threading
import signal
import sys
import random
import re
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync

# =====================================================================
# 1. CONFIGURATION (CLI-driven)
# =====================================================================
parser = argparse.ArgumentParser(description="O-RAN FALCON filesystem data collection pipeline")
parser.add_argument("--normal-hours", type=float, default=5.0,
                    help="Hours of NORMAL baseline collection (default: 5)")
parser.add_argument("--fault-reps", type=int, default=2,
                    help="Number of times to repeat ALL faults on ALL containers (default: 2)")
parser.add_argument("--stress-duration", type=int, default=180,
                    help="Stress injection duration in seconds (default: 180)")
parser.add_argument("--cooldown-period", type=int, default=420,
                    help="Cooldown between faults in seconds (default: 420)")
parser.add_argument("--interval", type=float, default=0.1,
                    help="Collection interval in seconds (default: 0.1)")
parser.add_argument("--normal-only", action="store_true",
                    help="Collect normal-phase data only; never transition to ANOMALY, never spawn stress_loop")
args = parser.parse_args()

NORMAL_DURATION_SEC = int(args.normal_hours * 3600)
FAULT_REPS = args.fault_reps
STRESS_DURATION = args.stress_duration
COOLDOWN_PERIOD = args.cooldown_period
FETCH_INTERVAL = args.interval

# Phase tracking
current_phase = "ANOMALY"
phase_lock = threading.Lock()
normal_phase_done = threading.Event()
experiment_complete = threading.Event()

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE_DIR, "experimental_phase0")
DATA_DIR = os.path.join(EXP_DIR, "data_fs_plane")
os.makedirs(DATA_DIR, exist_ok=True)

if args.normal_only:
    NORMAL_OUTPUT = os.path.join(DATA_DIR, "train_normal_only_fs.csv")
else:
    NORMAL_OUTPUT = os.path.join(DATA_DIR, "train_normal_fs.csv")
ANOMALY_OUTPUT = os.path.join(DATA_DIR, "test_anomaly_fs.csv")
influxDB_Token = "605bc59413b7d5457d181ccf20f9fda15693f81b068d70396cc183081b264f3b"
org = "srs"

# Containers
TARGET_PREFIXES = ["srscu", "srsdu"]
TARGET_CONTAINERS = ["srscu0", "srscu1", "srscu2", "srsdu0", "srsdu1", "srsdu2", "srsdu3", "srsdu4", "srsdu5"]
UE_CONTAINERS = ["srsue0", "srsue1", "srsue2", "srsue3", "srsue4", "srsue5"]
CGROUP_BASE = "/sys/fs/cgroup/system.slice"

# Pre-flight thresholds
PCI_UL_BRATE_MIN = 5_000_000
DRIFT_WARN_PCT = 0.20
DRIFT_CHECK_EVERY = 3000  # every 3000 rows (~5 min at 0.1s)

# Stress tracking
global_stress_data: Dict[str, List[int]] = {}
stress_lock = threading.Lock()

# RRC Release tracking
rrc_release_flags: Dict[str, bool] = {ue: False for ue in UE_CONTAINERS}
rrc_lock = threading.Lock()

log_file = os.path.join(DATA_DIR, "experiment_run.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

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
    except Exception:
        pass


def signal_handler(sig, frame):
    logging.info("Received termination signal. Cleaning up...")
    nuclear_cleanup()
    experiment_complete.set()
    sys.exit(0)


def get_container_name(container_id: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            f"docker inspect --format '{{{{.Name}}}}' {container_id}",
            shell=True, stderr=subprocess.DEVNULL
        ).decode().strip().lstrip('/')
    except Exception:
        return None


def check_for_crashed_containers():
    logging.info("=== POST-EXPERIMENT CRASH CHECK ===")
    try:
        cmd = "docker ps -a --filter 'status=exited' --format '{{.Names}} (Exit {{.Status}})'"
        crashed = subprocess.check_output(cmd, shell=True).decode().splitlines()
        if crashed:
            logging.error("WARNING! The following containers CRASHED during the experiment:")
            for c in crashed:
                logging.error(f"  -> {c}")
        else:
            logging.info("SUCCESS! All containers survived the fault injections.")
    except Exception as e:
        logging.error(f"Failed to check crashes: {e}")


# =====================================================================
# 3. RRC RELEASE MONITOR
# =====================================================================
def check_rrc_release_once() -> Dict[str, bool]:
    results = {}
    for ue in UE_CONTAINERS:
        try:
            logs = subprocess.check_output(
                ["docker", "logs", "--tail", "100", ue],
                text=True, stderr=subprocess.DEVNULL
            )
            found = "Received RRC Release" in logs
            results[ue] = found
            if found:
                with rrc_lock:
                    if not rrc_release_flags[ue]:
                        rrc_release_flags[ue] = True
                        logging.warning(f"RRC RELEASE detected in {ue}!")
        except Exception:
            results[ue] = False
    return results


def rrc_monitor_loop():
    logging.info("[RRC Monitor] Started — checking UE logs every 10s")
    while not experiment_complete.is_set():
        detections = check_rrc_release_once()
        for ue, detected in detections.items():
            if detected:
                logging.warning(f"[RRC Monitor] {ue} received RRC Release — attempting restart")
                try:
                    subprocess.run(f"docker restart {ue}", shell=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=60)
                    logging.info(f"[RRC Monitor] {ue} restarted successfully")
                    with rrc_lock:
                        rrc_release_flags[ue] = False
                except Exception as e:
                    logging.error(f"[RRC Monitor] Failed to restart {ue}: {e}")
        experiment_complete.wait(timeout=10)
    logging.info("[RRC Monitor] Stopped")


# =====================================================================
# 4. CONTAINER DISCOVERY & CGROUP PATH RESOLUTION
# =====================================================================
@dataclass
class ContainerInfo:
    name: str
    short_id: str
    full_id: str
    pid: int
    cgroup_path: str
    locked_ifaces: List[str]
    cpu_cores_allocated: float


def read_file_safe(path: str) -> str:
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception:
        return ""


def read_cpu_allocation(cgroup: str) -> float:
    content = read_file_safe(f"{cgroup}/cpu.max").strip()
    if not content:
        return float(os.cpu_count() or 1)
    parts = content.split()
    if parts[0] == 'max':
        return float(os.cpu_count() or 1)
    quota = float(parts[0])
    period = float(parts[1]) if len(parts) > 1 else 100000.0
    return quota / period


def discover_containers() -> List[ContainerInfo]:
    containers = []
    output = subprocess.check_output(
        "docker ps --format '{{.ID}} {{.Names}}'", shell=True
    ).decode().splitlines()

    for line in output:
        parts = line.split()
        if len(parts) < 2:
            continue
        short_id, name = parts[0], parts[1]

        if not any(name.startswith(p) for p in TARGET_PREFIXES):
            continue

        try:
            full_id = subprocess.check_output(
                f"docker inspect --format '{{{{.Id}}}}' {name}",
                shell=True, stderr=subprocess.DEVNULL
            ).decode().strip()
            pid = int(subprocess.check_output(
                f"docker inspect --format '{{{{.State.Pid}}}}' {name}",
                shell=True, stderr=subprocess.DEVNULL
            ).decode().strip())
            cgroup_path = f"{CGROUP_BASE}/docker-{full_id}.scope"

            if os.path.isdir(cgroup_path):
                ifaces = discover_network_interfaces(pid)
                cpu_alloc = read_cpu_allocation(cgroup_path)
                containers.append(ContainerInfo(name, short_id, full_id, pid, cgroup_path, ifaces, cpu_alloc))
                logging.info(f"  Discovered: {name} (PID={pid}, CPUs={cpu_alloc:.1f}, ifaces={ifaces})")
            else:
                logging.warning(f"  {name}: cgroup path not found at {cgroup_path}")
        except Exception as e:
            logging.warning(f"  Failed to discover {name}: {e}")

    return sorted(containers, key=lambda c: c.name)


# =====================================================================
# 5. KERNEL FILE READERS (True 0.1s Resolution)
# =====================================================================
def read_cpu_stat(cgroup: str) -> Dict[str, float]:
    data = {}
    content = read_file_safe(f"{cgroup}/cpu.stat")
    for line in content.splitlines():
        parts = line.split()
        if len(parts) == 2:
            data[parts[0]] = float(parts[1])
    return data


def read_memory(cgroup: str) -> Dict[str, float]:
    result = {}
    current = read_file_safe(f"{cgroup}/memory.current").strip()
    result['memory_current_bytes'] = float(current) if current else 0.0

    swap = read_file_safe(f"{cgroup}/memory.swap.current").strip()
    result['memory_swap_bytes'] = float(swap) if swap else 0.0

    content = read_file_safe(f"{cgroup}/memory.stat")
    keep_fields = {
        'anon', 'file', 'kernel', 'kernel_stack', 'pagetables',
        'sock', 'slab', 'slab_reclaimable', 'slab_unreclaimable',
        'pgfault', 'pgmajfault', 'workingset_refault_anon',
        'workingset_refault_file', 'inactive_anon', 'active_anon',
        'inactive_file', 'active_file', 'unevictable',
    }
    for line in content.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in keep_fields:
            result[f'memstat_{parts[0]}'] = float(parts[1])

    return result


def read_io_stat(cgroup: str) -> Dict[str, float]:
    totals = {'io_rbytes': 0.0, 'io_wbytes': 0.0, 'io_rios': 0.0, 'io_wios': 0.0}
    content = read_file_safe(f"{cgroup}/io.stat")
    for line in content.splitlines():
        for key in ['rbytes=', 'wbytes=', 'rios=', 'wios=']:
            match = re.search(rf'{key}(\d+)', line)
            if match:
                mapped = 'io_' + key.rstrip('=')
                totals[mapped] += float(match.group(1))
    return totals


def read_pids(cgroup: str) -> float:
    val = read_file_safe(f"{cgroup}/pids.current").strip()
    return float(val) if val else 0.0


def read_pressure(cgroup: str, resource: str) -> Dict[str, float]:
    result = {}
    content = read_file_safe(f"{cgroup}/{resource}.pressure")
    for line in content.splitlines():
        parts = line.split()
        if not parts:
            continue
        kind = parts[0]
        for kv in parts[1:]:
            if '=' in kv:
                k, v = kv.split('=', 1)
                result[f'psi_{resource}_{kind}_{k}'] = float(v)
    return result


def discover_network_interfaces(pid: int) -> List[str]:
    ifaces = []
    content = read_file_safe(f"/proc/{pid}/net/dev")
    for line in content.splitlines():
        if ':' not in line:
            continue
        iface = line.split(':', 1)[0].strip()
        if iface != 'lo':
            ifaces.append(iface)
    return sorted(ifaces)


NET_SUFFIXES = ['rx_bytes', 'rx_packets', 'rx_errors', 'rx_drops',
                'tx_bytes', 'tx_packets', 'tx_errors', 'tx_drops']


def read_network_stats(pid: int, locked_ifaces: List[str]) -> Dict[str, float]:
    result = {}
    for iface in locked_ifaces:
        for suffix in NET_SUFFIXES:
            result[f'net_{iface}_{suffix}'] = 0.0

    content = read_file_safe(f"/proc/{pid}/net/dev")
    for line in content.splitlines():
        if ':' not in line:
            continue
        iface, stats = line.split(':', 1)
        iface = iface.strip()
        if iface not in locked_ifaces:
            continue
        fields = stats.split()
        if len(fields) < 16:
            continue
        result[f'net_{iface}_rx_bytes'] = float(fields[0])
        result[f'net_{iface}_rx_packets'] = float(fields[1])
        result[f'net_{iface}_rx_errors'] = float(fields[2])
        result[f'net_{iface}_rx_drops'] = float(fields[3])
        result[f'net_{iface}_tx_bytes'] = float(fields[8])
        result[f'net_{iface}_tx_packets'] = float(fields[9])
        result[f'net_{iface}_tx_errors'] = float(fields[10])
        result[f'net_{iface}_tx_drops'] = float(fields[11])
    return result


def read_all_container_metrics(container: ContainerInfo) -> Dict[str, float]:
    cg = container.cgroup_path
    metrics = {}

    cpu = read_cpu_stat(cg)
    metrics['cpu_usage_usec'] = cpu.get('usage_usec', 0.0)
    metrics['cpu_user_usec'] = cpu.get('user_usec', 0.0)
    metrics['cpu_system_usec'] = cpu.get('system_usec', 0.0)
    metrics['cpu_nr_throttled'] = cpu.get('nr_throttled', 0.0)
    metrics['cpu_throttled_usec'] = cpu.get('throttled_usec', 0.0)

    mem = read_memory(cg)
    metrics.update(mem)

    io = read_io_stat(cg)
    metrics.update(io)

    metrics['pids_current'] = read_pids(cg)

    metrics.update(read_pressure(cg, 'cpu'))
    metrics.update(read_pressure(cg, 'memory'))
    metrics.update(read_pressure(cg, 'io'))

    net = read_network_stats(container.pid, container.locked_ifaces)
    metrics.update(net)

    return metrics


# =====================================================================
# 6. RATE COMPUTATION (Delta / Elapsed)
# =====================================================================
COUNTER_FIELDS = {
    'cpu_usage_usec', 'cpu_user_usec', 'cpu_system_usec',
    'cpu_throttled_usec', 'cpu_nr_throttled',
    'io_rbytes', 'io_wbytes', 'io_rios', 'io_wios',
    'memstat_pgfault', 'memstat_pgmajfault',
    'memstat_workingset_refault_anon', 'memstat_workingset_refault_file',
}


def is_counter_field(field: str) -> bool:
    if field in COUNTER_FIELDS:
        return True
    if field.startswith('net_') and any(
        field.endswith(s) for s in ['_bytes', '_packets', '_errors', '_drops']
    ):
        return True
    return False


def compute_rates(
    current: Dict[str, Dict[str, float]],
    previous: Dict[str, Dict[str, float]],
    elapsed: float,
    cpu_alloc: Dict[str, float],
) -> Dict[str, float]:
    result = {}
    for cname, cur_metrics in current.items():
        prev_metrics = previous.get(cname, {})
        alloc = cpu_alloc.get(cname, 1.0)

        for field, cur_val in cur_metrics.items():
            col = f"{cname}_{field}"

            if is_counter_field(field) and field in prev_metrics and elapsed > 0:
                delta = cur_val - prev_metrics[field]
                if delta < 0:
                    delta = 0

                if field in ('cpu_usage_usec', 'cpu_user_usec', 'cpu_system_usec'):
                    rate = (delta / (elapsed * 1_000_000 * alloc)) * 100.0
                    col_name = field.replace('_usec', '_pct')
                    result[f"{cname}_{col_name}"] = rate
                elif field == 'cpu_throttled_usec':
                    result[f"{cname}_cpu_throttled_pct"] = (delta / (elapsed * 1_000_000 * alloc)) * 100.0
                else:
                    result[col + '_rate'] = delta / elapsed
            else:
                result[col] = cur_val

    return result


# =====================================================================
# 7. HOST-LEVEL METRICS
# =====================================================================
def read_host_metrics() -> Dict[str, float]:
    result = {}
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('MemTotal:'):
                result['host_mem_total_bytes'] = float(line.split()[1]) * 1024
            elif line.startswith('MemAvailable:'):
                result['host_mem_available_bytes'] = float(line.split()[1]) * 1024
                break
    try:
        result['host_cpu_cores'] = float(os.cpu_count() or 1)
    except Exception:
        result['host_cpu_cores'] = 1.0
    return result


# =====================================================================
# 8. INFLUXDB CLIENT
# =====================================================================
async def fetch_influx_data() -> pd.DataFrame:
    df, _ = await fetch_influx_data_with_time()
    return df


async def fetch_influx_data_with_time():
    """Fetch latest UE radio metrics + the Influx record _time for skew tracking.

    Returns: (DataFrame with one row of metrics, datetime of newest _time or None)
    """
    data = {}
    newest_time = None
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
                    rec_time = record.values.get("_time")
                    if rec_time is not None and (newest_time is None or rec_time > newest_time):
                        newest_time = rec_time
                    for field, value in record.values.items():
                        if field in ['_start', '_stop', '_time', '_measurement', 'rnti', 'pci', 'testbed', 'result', 'table']:
                            continue
                        key = f"PCI-{pci}_RNTI-4601_{field}"
                        try:
                            data[key] = float(value) if value is not None and value != 'n/a' else 0.0
                        except (ValueError, TypeError):
                            data[key] = 0.0
    except Exception:
        pass
    return pd.DataFrame([data]), newest_time


# Influx producer cache (read by main loop, written by influx_producer())
influx_cache: Dict[str, float] = {}
influx_cache_meta = {"record_time": None, "fetched_at": None}
_influx_async_lock = None  # asyncio.Lock created lazily inside event loop


async def influx_producer():
    """Fetch InfluxDB ue_info every FETCH_INTERVAL on its own tick, in parallel with the kernel loop."""
    global _influx_async_lock
    if _influx_async_lock is None:
        _influx_async_lock = asyncio.Lock()
    while not experiment_complete.is_set():
        tick_start = time.monotonic()
        try:
            df, rec_time = await fetch_influx_data_with_time()
            new_data = df.iloc[0].to_dict() if not df.empty else {}
            async with _influx_async_lock:
                influx_cache.clear()
                influx_cache.update(new_data)
                influx_cache_meta["record_time"] = rec_time
                influx_cache_meta["fetched_at"] = time.time()
        except Exception as e:
            logging.debug(f"[InfluxProducer] fetch failed: {e}")
        elapsed = time.monotonic() - tick_start
        await asyncio.sleep(max(0, FETCH_INTERVAL - elapsed))


# =====================================================================
# 9. STRESS ENGINE (with verification & N repetitions)
# =====================================================================
def verify_stress_applied(container_id: str, stress_type: int) -> bool:
    cname = get_container_name(container_id) or container_id
    try:
        if stress_type in (1, 2):
            result = subprocess.run(
                f"docker exec {container_id} pgrep -c stress-ng",
                shell=True, capture_output=True, text=True
            )
            count = int(result.stdout.strip()) if result.stdout.strip() else 0
            label = "CPU" if stress_type == 1 else "MEM"
            if count > 0:
                logging.info(f"  [VERIFY] {cname} {label} stress confirmed ({count} stress-ng processes)")
                return True
            logging.warning(f"  [VERIFY] {cname} {label} stress NOT detected!")
            return False
        elif stress_type == 3:
            result = subprocess.run(
                f"docker exec {container_id} tc qdisc show dev eth0",
                shell=True, capture_output=True, text=True
            )
            if "netem" in result.stdout:
                logging.info(f"  [VERIFY] {cname} NET stress confirmed (netem rule active)")
                return True
            logging.warning(f"  [VERIFY] {cname} NET stress NOT detected!")
            return False
    except Exception as e:
        logging.warning(f"  [VERIFY] {cname} verification failed: {e}")
    return False


def injectStress(container_id: str, typeOfStress: int, duration: int):
    """Inject stress and label the container ONLY while the stressor is kernel-confirmed live.

    Edge tightness:
      1. Start stressor.
      2. Poll verify_stress_applied() every 50ms (max 5s) until kernel confirms — only THEN
         set global_stress_data[cname] = [type, intensity].
      3. Hold for duration.
      4. Tear down stressor; poll until kernel confirms it's gone — only THEN clear label.
    Result: per-row stress label edges align with actually-applied window to <=1 sample (<=500ms).
    """
    cname = get_container_name(container_id)
    intensity = random.randint(85, 95)
    proc = None
    POLL = 0.05
    CONFIRM_TIMEOUT = 5.0
    CLEAR_TIMEOUT = 5.0

    try:
        # 1. Fire stressor (label still OFF)
        if typeOfStress == 1:
            cmd = f"docker exec {container_id} stress-ng --cpu 4 --cpu-load {intensity} --timeout {duration}s"
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif typeOfStress == 2:
            mb = int((intensity / 100.0) * 512)
            cmd = f"docker exec {container_id} stress-ng --vm 1 --vm-bytes {mb}M --timeout {duration}s"
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif typeOfStress == 3:
            loss = 2.0
            cmd = f"docker exec {container_id} tc qdisc replace dev eth0 root netem loss {loss:.2f}%"
            subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)
        else:
            return

        # 2. Wait for kernel confirmation, THEN set label
        deadline = time.monotonic() + CONFIRM_TIMEOUT
        confirmed = False
        while time.monotonic() < deadline:
            if verify_stress_applied(container_id, typeOfStress):
                confirmed = True
                break
            time.sleep(POLL)

        if not confirmed:
            logging.warning(f"[Stress] {cname} type={typeOfStress} NEVER confirmed live — "
                            f"NOT labelling (skipping). Tearing down.")
            return  # finally: cleans up

        with stress_lock:
            global_stress_data[cname] = [typeOfStress, intensity]
        logging.info(f"[Stress] {cname} type={typeOfStress} label ON (confirmed)")

        # 3. Hold for duration
        if typeOfStress in (1, 2) and proc is not None:
            proc.wait()
        else:
            time.sleep(duration)

    except Exception as e:
        logging.error(f"Stress injection failed on {cname}: {e}")
    finally:
        # 4. Tear down + wait for kernel-confirmed off, THEN clear label
        if typeOfStress == 3:
            subprocess.run(
                f"docker exec {container_id} tc qdisc del dev eth0 root",
                shell=True, stderr=subprocess.DEVNULL
            )

        deadline = time.monotonic() + CLEAR_TIMEOUT
        while time.monotonic() < deadline:
            if not verify_stress_applied(container_id, typeOfStress):
                break
            time.sleep(POLL)

        with stress_lock:
            global_stress_data[cname] = [0, 0]
        logging.info(f"[Stress] {cname} type={typeOfStress} label OFF (cleared)")


def stress_loop():
    logging.info(f"[Stress] Waiting for NORMAL phase to complete ({args.normal_hours}h)...")
    normal_phase_done.wait()

    if experiment_complete.is_set():
        return

    logging.info("[Stress] ANOMALY phase started. Waiting 5m for anomaly baseline...")
    time.sleep(300)

    output = subprocess.check_output(
        "docker ps --format '{{.ID}} {{.Names}}'", shell=True
    ).decode().splitlines()
    cmap = {line.split()[1]: line.split()[0] for line in output if len(line.split()) >= 2}
    victims = [(cmap[name], name) for name in TARGET_CONTAINERS if name in cmap]

    if not victims:
        logging.error("[Stress] No target containers found!")
        experiment_complete.set()
        return

    stress_names = {1: "CPU", 2: "MEM", 3: "NET"}
    total_faults = len(victims) * 3 * FAULT_REPS
    fault_num = 0

    for rep in range(1, FAULT_REPS + 1):
        logging.info(f"=== FAULT REPETITION {rep}/{FAULT_REPS} ===")
        for victim_id, victim_name in victims:
            for s_type in [1, 2, 3]:
                fault_num += 1
                logging.info(
                    f"[{fault_num}/{total_faults}] START STRESS: {victim_name} "
                    f"Type:{stress_names[s_type]} Rep:{rep}/{FAULT_REPS}"
                )
                injectStress(victim_id, s_type, STRESS_DURATION)
                logging.info(f"COOLDOWN: {COOLDOWN_PERIOD}s...")
                time.sleep(COOLDOWN_PERIOD)

    logging.info("=== ALL FAULTS INJECTED. EXPERIMENT COMPLETE. ===")
    check_for_crashed_containers()

    logging.info("Collecting 5 min of post-fault baseline...")
    time.sleep(300)
    experiment_complete.set()


# =====================================================================
# 10. PRE-FLIGHT & BASELINE VALIDATION
# =====================================================================
def pre_flight_check_fs(
    containers: List[ContainerInfo],
    init_influx_df: pd.DataFrame,
) -> bool:
    logging.info("=== PRE-FLIGHT CHECK (filesystem) ===")
    passed = True

    # 1. All PCIs must have active UL traffic from InfluxDB
    for col in sorted(init_influx_df.columns):
        if 'ul_brate' not in col:
            continue
        pci_tag = col.split('_')[0]
        val = float(init_influx_df[col].values[0])
        ok = val >= PCI_UL_BRATE_MIN
        logging.info(f"  {pci_tag} ul_brate = {val / 1e6:.2f} Mbps  [{'OK' if ok else 'FAIL — UE NOT ATTACHED'}]")
        if not ok:
            passed = False

    # 2. Check each container is alive and has CPU activity
    for c in containers:
        cpu = read_cpu_stat(c.cgroup_path)
        usage = cpu.get('usage_usec', 0)
        mem_bytes = float(read_file_safe(f"{c.cgroup_path}/memory.current").strip() or '0')
        pids = read_pids(c.cgroup_path)
        logging.info(f"  {c.name:12s} CPU_usec={usage:>15,.0f}  MEM={mem_bytes / 1e6:.0f}MB  PIDs={pids:.0f}  [{'OK' if pids > 0 else 'FAIL'}]")
        if pids == 0:
            passed = False

    # 3. Check for RRC Release
    logging.info("  Checking UE containers for RRC Release...")
    rrc_results = check_rrc_release_once()
    for ue, detected in rrc_results.items():
        if detected:
            logging.warning(f"  {ue}: RRC Release detected BEFORE collection — restarting UE")
            subprocess.run(f"docker restart {ue}", shell=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            time.sleep(10)
            passed = False
        else:
            logging.info(f"  {ue}: No RRC Release [OK]")

    if passed:
        logging.info("PRE-FLIGHT PASSED — starting collection")
    else:
        logging.error("PRE-FLIGHT FAILED — fix issues before collecting")
        logging.error("Aborting — baseline would be corrupted. Fix and re-run.")
        sys.exit(1)

    logging.info("=== END PRE-FLIGHT ===")
    return passed


def save_baseline_snapshot(row: dict, snapshot_file: str):
    keep_keys = ['cpu_user_pct', 'memory_current', 'net_eth0_tx_bytes_rate',
                 'net_eth0_rx_bytes_rate', 'ul_brate', 'bsr']
    snapshot = {k: v for k, v in row.items()
                if any(x in k for x in keep_keys) and k != 'Timestamp'}
    snapshot['_timestamp'] = row['Timestamp']
    snapshot['_phase'] = row.get('phase', 'NORMAL')
    with open(snapshot_file, 'w') as f:
        json.dump(snapshot, f, indent=2)
    logging.info(f"Baseline snapshot saved -> {snapshot_file}")


def check_baseline_drift(snapshot: dict, row: dict):
    drifted = []
    for key, snap_val in snapshot.items():
        if key.startswith('_') or snap_val == 0:
            continue
        curr_val = row.get(key, 0)
        drift = abs(curr_val - snap_val) / (abs(snap_val) + 1e-9)
        if drift > DRIFT_WARN_PCT:
            drifted.append(f"{key[:55]}: baseline={snap_val:.2f} now={curr_val:.2f} drift={drift * 100:.0f}%")
    if drifted:
        logging.warning("=== BASELINE DRIFT DETECTED ===")
        for msg in drifted:
            logging.warning(f"  {msg}")
        logging.warning("================================")


# =====================================================================
# 11. MAIN SCRAPER (Kernel FS + InfluxDB, 0.1s resolution)
# =====================================================================
async def main():
    global current_phase

    # A. Discover containers
    logging.info("=== DISCOVERING CONTAINERS ===")
    containers = discover_containers()
    if not containers:
        logging.error("No target containers found! Are they running?")
        sys.exit(1)
    logging.info(f"Found {len(containers)} containers: {[c.name for c in containers]}")

    cpu_alloc = {c.name: c.cpu_cores_allocated for c in containers}

    host_metrics = read_host_metrics()
    num_cpus = os.cpu_count() or 1
    logging.info(f"Host: {num_cpus} CPUs, {host_metrics.get('host_mem_total_bytes', 0) / 1e9:.1f} GB RAM")
    for c in containers:
        logging.info(f"  {c.name}: {c.cpu_cores_allocated:.1f} cores allocated")

    # B. InfluxDB discovery & schema lock
    logging.info("Locking InfluxDB Schema...")
    init_df = await fetch_influx_data()
    influx_headers = sorted(list(init_df.columns))

    # C. Pre-flight check
    pre_flight_check_fs(containers, init_df)

    # D. First sample to discover all column names
    logging.info("Taking first sample to lock CSV headers...")
    first_raw = {}
    for c in containers:
        first_raw[c.name] = read_all_container_metrics(c)

    # NOTE: pass first_raw as BOTH current AND previous. With non-empty previous,
    # compute_rates takes the rate path for every counter field and emits the
    # *_rate / *_pct column names that the runtime loop will later produce.
    # If we pass {} here the schema locks to raw-counter names and DictWriter
    # silently drops every *_rate value at runtime → all counter columns become 0.
    dummy_rates = compute_rates(first_raw, first_raw, 1.0, cpu_alloc)
    fs_headers = sorted(dummy_rates.keys())

    # E. Stress headers
    cnames = sorted([c.name for c in containers if c.name.startswith(('srscu', 'srsdu'))])
    stress_headers = []
    for c in cnames:
        stress_headers.extend([f"{c}_stressType", f"{c}_stepStress"])
        with stress_lock:
            global_stress_data[c] = [0, 0]

    # F. Host metric headers
    host_headers = sorted(host_metrics.keys())

    # G. Extra columns (locked here BEFORE headers concatenation so they appear in row 1)
    extra_headers = ["phase", "rrc_release_detected", "any_stress_active",
                     "influx_record_time", "influx_age_ms"]

    # H. Initialize CSV
    headers = ["Timestamp"] + fs_headers + influx_headers + host_headers + stress_headers + extra_headers

    for out_file in [NORMAL_OUTPUT, ANOMALY_OUTPUT]:
        if not os.path.exists(out_file):
            with open(out_file, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=headers).writeheader()

    # I. Log experiment plan
    total_fault_time = len(TARGET_CONTAINERS) * 3 * FAULT_REPS * (STRESS_DURATION + COOLDOWN_PERIOD)
    logging.info("=" * 60)
    logging.info("EXPERIMENT PLAN:")
    logging.info(f"  Collection interval: {FETCH_INTERVAL}s")
    logging.info(f"  NORMAL phase:  {args.normal_hours}h ({NORMAL_DURATION_SEC}s)")
    logging.info(f"  ANOMALY phase: {len(TARGET_CONTAINERS)} containers x 3 faults x {FAULT_REPS} reps")
    logging.info(f"  Per fault:     {STRESS_DURATION}s stress + {COOLDOWN_PERIOD}s cooldown")
    logging.info(f"  Est. anomaly:  {total_fault_time / 3600:.1f}h")
    logging.info(f"  Est. total:    {(NORMAL_DURATION_SEC + total_fault_time + 600) / 3600:.1f}h")
    logging.info(f"  CSV columns:   {len(headers)} total "
                 f"({len(fs_headers)} filesystem + {len(influx_headers)} influx + "
                 f"{len(host_headers)} host + {len(stress_headers)} stress + {len(extra_headers)} extra)")
    logging.info(f"  Normal output: {NORMAL_OUTPUT}")
    logging.info(f"  Anomaly output: {ANOMALY_OUTPUT}")
    logging.info("=" * 60)
    logging.info("=== COLLECTION STARTED ===")

    # J. Collection Loop
    snapshot_file = os.path.join(DATA_DIR, "baseline_snapshot.json")
    baseline_snapshot = {}
    row_count = 0
    prev_raw: Dict[str, Dict[str, float]] = {}
    prev_time = time.monotonic()
    normal_start = time.time()

    # InfluxDB ticks at FETCH_INTERVAL on its own concurrent task; main loop
    # only reads the module-level influx_cache populated by influx_producer().
    producer_task = asyncio.create_task(influx_producer())

    while not experiment_complete.is_set():
        loop_start = time.monotonic()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Check phase transition (skipped entirely in --normal-only mode)
        with phase_lock:
            if (not args.normal_only
                    and current_phase == "NORMAL"
                    and (time.time() - normal_start) >= NORMAL_DURATION_SEC):
                current_phase = "ANOMALY"
                logging.info("=" * 60)
                logging.info("PHASE TRANSITION: NORMAL -> ANOMALY")
                logging.info(f"  Normal rows collected: {row_count}")
                logging.info("=" * 60)
                normal_phase_done.set()
            active_phase = current_phase

        # 1. Read all container metrics from kernel (fast — pure file reads)
        current_raw: Dict[str, Dict[str, float]] = {}
        for c in containers:
            current_raw[c.name] = read_all_container_metrics(c)

        elapsed = loop_start - prev_time

        # 2. Compute rates from deltas
        if prev_raw and elapsed > 0:
            rates = compute_rates(current_raw, prev_raw, elapsed, cpu_alloc)
        else:
            rates = compute_rates(current_raw, {}, 1.0, cpu_alloc)

        prev_raw = current_raw
        prev_time = loop_start

        # 3. Snapshot Influx cache (populated by concurrent influx_producer at FETCH_INTERVAL)
        if _influx_async_lock is not None:
            async with _influx_async_lock:
                row_influx = dict(influx_cache)
                rec_time = influx_cache_meta["record_time"]
        else:
            row_influx = {}
            rec_time = None

        if rec_time is not None:
            try:
                influx_age_ms = (datetime.now(rec_time.tzinfo) - rec_time).total_seconds() * 1000.0
            except Exception:
                influx_age_ms = -1.0
            influx_record_time_str = rec_time.isoformat()
        else:
            influx_age_ms = -1.0
            influx_record_time_str = ""

        # 4. Build row
        row = {"Timestamp": ts}

        for h in fs_headers:
            row[h] = rates.get(h, 0.0)

        for h in influx_headers:
            row[h] = row_influx.get(h, 0.0)

        current_host = read_host_metrics()
        for h in host_headers:
            row[h] = current_host.get(h, 0.0)

        # Stress labels
        with stress_lock:
            sc = global_stress_data.copy()
        for c in cnames:
            row[f"{c}_stressType"], row[f"{c}_stepStress"] = sc.get(c, [0, 0])

        any_stress = any(v[0] != 0 for v in sc.values())

        # RRC Release
        with rrc_lock:
            any_rrc = any(rrc_release_flags.values())

        row["phase"] = active_phase
        row["rrc_release_detected"] = 1 if any_rrc else 0
        row["any_stress_active"] = 1 if any_stress else 0
        row["influx_record_time"] = influx_record_time_str
        row["influx_age_ms"] = influx_age_ms

        # 5. Write to CSV
        output_file = NORMAL_OUTPUT if active_phase == "NORMAL" else ANOMALY_OUTPUT
        with open(output_file, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=headers, extrasaction='ignore').writerow(row)

        # 6. Baseline tracking
        row_count += 1
        if row_count == 1:
            save_baseline_snapshot(row, snapshot_file)
            baseline_snapshot = {k: v for k, v in row.items()
                                if any(x in k for x in ['cpu_user_pct', 'memory_current', 'tx_bytes_rate', 'rx_bytes_rate', 'ul_brate', 'bsr'])}

        if row_count % DRIFT_CHECK_EVERY == 0:
            logging.info(f"Rows collected: {row_count} | Phase: {active_phase} | Stress: {any_stress} | RRC: {any_rrc}")
            if active_phase == "NORMAL":
                check_baseline_drift(baseline_snapshot, row)

        if row_count % 600 == 0:  # every ~60s at 0.1s interval
            actual_elapsed = time.monotonic() - loop_start
            logging.info(f"Row {row_count} | loop={actual_elapsed * 1000:.1f}ms | ts={ts}")

        # 7. Accurate timing
        loop_elapsed = time.monotonic() - loop_start
        await asyncio.sleep(max(0, FETCH_INTERVAL - loop_elapsed))

    # Stop the Influx producer cleanly
    producer_task.cancel()
    try:
        await producer_task
    except (asyncio.CancelledError, Exception):
        pass

    logging.info("=" * 60)
    logging.info("DATA COLLECTION COMPLETE")
    logging.info(f"  Total rows: {row_count}")
    logging.info(f"  Normal data: {NORMAL_OUTPUT}")
    logging.info(f"  Anomaly data: {ANOMALY_OUTPUT}")
    logging.info("=" * 60)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    nuclear_cleanup()

    if not args.normal_only:
        threading.Thread(target=stress_loop, daemon=True).start()
    else:
        logging.info("[--normal-only] stress_loop disabled — pure normal-phase collection")
    threading.Thread(target=rrc_monitor_loop, daemon=True).start()

    asyncio.run(main())
