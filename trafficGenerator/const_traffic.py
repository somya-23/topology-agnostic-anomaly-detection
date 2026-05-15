import subprocess
import time
import signal
import sys

open5gs_id = "open5gs_5gc"
active_streams = {}

def get_docker_ids():
    """Finds all running srsue container IDs."""
    try:
        cmd = "docker ps --filter 'name=srsue' --format '{{.ID}}'"
        result = subprocess.check_output(cmd, shell=True, text=True)
        return result.strip().split('\n') if result.strip() else []
    except:
        return []

def force_kill_all_iperf():
    """Cleans up all active local processes and remote iperf instances."""
    global active_streams
    for cid, p in active_streams.items():
        try:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=0.2)
        except: pass
    active_streams.clear()

    ids = get_docker_ids() + [open5gs_id]
    for cid in ids:
        subprocess.run(f"docker exec {cid} pkill -9 iperf", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def start_iperf_server():
    """Starts iperf server mode on the Open5GS container."""
    print(f"Starting iperf server on {open5gs_id}...")
    subprocess.run(f"docker exec -d {open5gs_id} iperf -s -u -B 10.45.1.1", shell=True)

def start_uplink_iperf(ue_id, mbps_rate):
    """UE -> Core. Sends continuous UDP traffic."""
    target_ip = "10.45.1.1"
    # -t 86400 keeps the stream running for 24 hours
    cmd = ["docker", "exec", ue_id, "iperf", "-c", target_ip, "-u", "-b", f"{mbps_rate}M", "-t", "86400"]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        active_streams[ue_id] = p
        print(f"Started continuous {mbps_rate} Mbps uplink stream for UE: {ue_id}")
    except Exception as e:
        print(f"Error starting iperf on {ue_id}: {e}")

def cleanup_and_exit(signum=None, frame=None):
    print("\n[TrafficGen] Cleaning up and exiting...")
    force_kill_all_iperf()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)

def main():
    global active_streams

    print("Initial cleanup...")
    force_kill_all_iperf()
    start_iperf_server()

    docker_ids = get_docker_ids()
    if not docker_ids:
        print("No UEs found. Waiting...")
        time.sleep(10)
        return

    # ========================================================
    # CONSTANT TRAFFIC RATE PER UE
    # ========================================================
    TARGET_MBPS_PER_UE = 10.0

    print(f"\n>>> RUNNING CONSTANT TRAFFIC <<<")
    print(f"Target Rate Per UE: {TARGET_MBPS_PER_UE} Mbps")
    print(f"Total Active UEs Detected: {len(docker_ids)}")
    print(f"Total Network Load: {TARGET_MBPS_PER_UE * len(docker_ids)} Mbps")
    print("Traffic will run continuously. Press Ctrl+C to stop.\n")

    # Start initial streams for all detected UEs
    for cid in docker_ids:
        start_uplink_iperf(cid, TARGET_MBPS_PER_UE)

    # Health monitoring loop
    while True:
        time.sleep(5)
        current_docker_ids = get_docker_ids()

        # Check for newly spawned UEs or crashed streams
        for cid in current_docker_ids:
            if cid not in active_streams or active_streams[cid].poll() is not None:
                if cid in active_streams:
                    print(f"Stream died for UE {cid}. Restarting...")
                start_uplink_iperf(cid, TARGET_MBPS_PER_UE)

        # Cleanup dictionary for UEs that no longer exist
        dead_cids = [cid for cid in active_streams.keys() if cid not in current_docker_ids]
        for dead_cid in dead_cids:
            print(f"UE {dead_cid} no longer active. Removing from monitor.")
            del active_streams[dead_cid]

if __name__ == "__main__":
    while True:
        try:
            main()
        except KeyboardInterrupt:
            cleanup_and_exit()
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(5)
