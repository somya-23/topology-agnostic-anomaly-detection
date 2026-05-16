import subprocess
import time
import signal
import sys

# Global configuration
docker_ids = []
open5gs_id = "open5gs_5gc" 
active_procs = []

# ================= HELPER FUNCTIONS =================

result = subprocess.run("pwd", capture_output=True, text=True)
pwd = result.stdout.strip()
print(f'Current working directory is {pwd}')

def get_docker_ids():
    """Finds all srsue container IDs."""
    try:
        cmd = "docker ps --filter 'name=srsue' --format '{{.ID}}'"
        result = subprocess.check_output(cmd, shell=True, text=True)
        return result.strip().split('\n') if result.strip() else []
    except:
        return []

def force_kill_all_iperf():
    """Cleans up active local processes and remote iperf instances."""
    global active_procs
    for p in active_procs:
        try:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=0.2)
        except: pass
    active_procs = []

    # Kill iperf inside all relevant containers
    ids = get_docker_ids() + [open5gs_id]
    for cid in ids:
        subprocess.run(f"docker exec {cid} pkill -9 iperf", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def start_iperf_server():
    """Starts iperf server mode on the Open5GS container."""
    print(f"Starting iperf server on {open5gs_id}...")
    subprocess.run(f"docker exec -d {open5gs_id} iperf -s -u -B 10.45.1.1", shell=True)

def start_uplink_iperf(ue_id, mbps_rate, duration=900):
    """UE -> Core. Sends UDP traffic at a specific bitrate for `duration` seconds."""
    target_ip = "10.45.1.1"
    cmd = ["docker", "exec", ue_id, "iperf", "-c", target_ip, "-u",
           "-b", f"{mbps_rate}M", "-t", str(int(duration))]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        active_procs.append(p)
    except Exception as e:
        print(f"Error starting iperf on {ue_id}: {e}")

def cleanup_and_exit(signum=None, frame=None):
    print("\n[TrafficGen] Cleaning up and exiting...")
    force_kill_all_iperf()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)

# ================= MAIN LOGIC =================

# Total Gb per 15-minute interval for normal traffic phase
DISTRIBUTION = [11.0, 8.1, 5.6, 15.0]
WINDOW_SEC = 900  # 15 minutes


def get_srsue2_ids():
    """Finds only srsue2 container IDs (for the half-traffic phase)."""
    try:
        cmd = "docker ps --filter 'name=srsue2' --format '{{.ID}}'"
        result = subprocess.check_output(cmd, shell=True, text=True)
        return result.strip().split('\n') if result.strip() else []
    except:
        return []


def run_traffic_phase(ue_ids: list, distribution: list, phase_name: str, rate_scale: float = 1.0):
    """Runs one hour of traffic across 4x 15-minute intervals.

    Args:
        ue_ids: Container IDs to send traffic from.
        distribution: List of 4 Gb totals per interval.
        phase_name: Label for log output.
        rate_scale: Multiply computed per-UE rate by this factor (0.5 = half traffic).
    """
    print(f"\n{'='*60}")
    print(f"[{phase_name}] Starting — {len(ue_ids)} UE(s), scale={rate_scale}")
    print(f"{'='*60}")

    for i, gb_total in enumerate(distribution):
        dist_path = f'{pwd}/trafficGenerator/traffic_distribution.txt'
        try:
            with open(dist_path, 'w') as file:
                file.write(f"{(gb_total * rate_scale) / max(distribution)}\n")
        except FileNotFoundError:
            print(f"Warning: Could not write to {dist_path}")

        total_mbps = (gb_total * 1000) / WINDOW_SEC
        mbps_per_ue = (total_mbps / len(ue_ids)) * rate_scale

        print(f"[{phase_name}] Interval {i+1}/4: {gb_total} Gb | "
              f"{total_mbps:.2f} Mbps total | {mbps_per_ue:.2f} Mbps/UE")

        interval_start = time.time()
        for cid in ue_ids:
            start_uplink_iperf(cid, mbps_per_ue, duration=WINDOW_SEC)

        while time.time() - interval_start < WINDOW_SEC:
            time.sleep(min(30, WINDOW_SEC - (time.time() - interval_start)))
            now = time.time()
            remaining = max(1, int(WINDOW_SEC - (now - interval_start)))
            active_procs[:] = [p for p in active_procs if p.poll() is None]

            current_ids = get_docker_ids() if rate_scale == 1.0 else get_srsue2_ids()
            for cid in current_ids:
                still_running = any(
                    f" {cid} " in " ".join(p.args) if isinstance(p.args, list)
                    else cid in str(p.args)
                    for p in active_procs
                )
                if not still_running and remaining > 5:
                    print(f"  [{phase_name}] UE {cid} iperf died — restarting for {remaining}s")
                    start_uplink_iperf(cid, mbps_per_ue, duration=remaining)

        force_kill_all_iperf()

    print(f"\n[{phase_name}] Complete.")


def main():
    global docker_ids

    print("Initial cleanup...")
    force_kill_all_iperf()
    start_iperf_server()

    docker_ids = get_docker_ids()
    if not docker_ids:
        print("No UEs found. Exiting.")
        sys.exit(1)

    # Phase 1: Normal traffic — all UEs, full rate, 1 hour
    run_traffic_phase(docker_ids, DISTRIBUTION, phase_name="NORMAL", rate_scale=1.0)

    # Cooldown before Phase 2
    print("\n[Cooldown] Waiting 60s before half-traffic phase...")
    time.sleep(60)

    # Phase 2: Half traffic — srsue2 only, 50% rate, 1 hour
    srsue2_ids = get_srsue2_ids()
    if not srsue2_ids:
        print("[HALF] No srsue2 containers found — skipping Phase 2.")
        return

    run_traffic_phase(srsue2_ids, DISTRIBUTION, phase_name="HALF", rate_scale=0.5)


if __name__ == "__main__":
    main()
