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

def main():
    global docker_ids
    
    print("Initial cleanup...")
    force_kill_all_iperf()
    start_iperf_server()

    docker_ids = get_docker_ids()
    if not docker_ids:
        print("No UEs found. Waiting...")
        time.sleep(10)
        return

    # Raw Distribution (Total Gb per 15-minute interval)
    distribution = [
        11.0, 8.1, 5.6, 3.6, 2.7, 1.9, 3.0, 5.0, 7.1, 11.1, 11.2, 11.9,
        12.4, 12.3, 13.0, 13.1, 12.9, 12.7, 12.4, 12.2, 12.0, 13.0, 14.0, 15.0, 14.0
    ]
        

    for i, gb_total in enumerate(distribution):
        dist_path = f'{pwd}/trafficGenerator/traffic_distribution.txt'
        try:
            with open(dist_path, 'w') as file:
                file.write(f"{gb_total/max(distribution)}\n")
        except FileNotFoundError:
            print(f"Warning: Could not write to {dist_path}")
        # Calculation: Total Gb -> Mbps -> Per UE rate
        total_mbps = (gb_total * 1000) / 900
        mbps_per_ue = total_mbps / len(docker_ids)

        print(f"Interval {i+1}: Target {gb_total} Gb | Total Rate: {total_mbps:.2f} Mbps | Per UE: {mbps_per_ue:.2f} Mbps (flat 15min)")

        # Flat traffic for the whole 15-minute window: ONE long iperf per UE.
        # Rate stays constant inside the interval; only changes at interval boundary.
        WINDOW_SEC = 900
        interval_start = time.time()
        for cid in docker_ids:
            start_uplink_iperf(cid, mbps_per_ue, duration=WINDOW_SEC)

        # Lightweight watchdog: re-arm dead UEs with the remaining time.
        # Cheap (every 30s, no docker exec) and self-heals UE drops without
        # waiting a full 15 min for the next interval.
        last_check = interval_start
        while time.time() - interval_start < WINDOW_SEC:
            time.sleep(min(30, WINDOW_SEC - (time.time() - interval_start)))
            now = time.time()
            remaining = max(1, int(WINDOW_SEC - (now - interval_start)))
            active_procs[:] = [p for p in active_procs if p.poll() is None]

            # Detect any UE whose iperf has died and restart for the remaining window
            current_ids = get_docker_ids()
            for cid in current_ids:
                still_running = any(
                    f" {cid} " in " ".join(p.args) if isinstance(p.args, list)
                    else cid in str(p.args)
                    for p in active_procs
                )
                if not still_running and remaining > 5:
                    print(f"  UE {cid} iperf died — restarting for remaining {remaining}s")
                    start_uplink_iperf(cid, mbps_per_ue, duration=remaining)
            last_check = now

        # Hard boundary: kill anything still running before next interval changes the rate
        force_kill_all_iperf()

if __name__ == "__main__":
    while True:
        try:
            main()
        except KeyboardInterrupt:
            cleanup_and_exit()
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(5)