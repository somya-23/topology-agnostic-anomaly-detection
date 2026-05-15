import subprocess
import time
import psutil
import signal
import sys
from datetime import datetime
import argparse  # Import argparse for command-line argument parsing

# Set up argparse to handle command-line arguments
parser = argparse.ArgumentParser(description="Start and monitor system with resource constraints.")
parser.add_argument("--resource-constraint", type=str, required=False, help="Specify resource constraints for the system.", default="no", choices=["yes", "no"])
args = parser.parse_args()

resource_constraint = args.resource_constraint

# Handle the resource constraint argument
if args.resource_constraint=='yes':
    print(f"Resource constraints set to: {args.resource_constraint}")
else:
    print("No resource constraints provided.")
    

result = subprocess.run("pwd", capture_output=True, text=True)
pwd = result.stdout.strip()
print(f'Current working directory is {pwd}')

# Define processes and containers to manage
processes_to_monitor = [
    f"{pwd}/trafficGenerator/test_traffic.py",
    f"{pwd}/stresser/stress_tester_icc.py"
]

print(f"Processes to monitor: {processes_to_monitor}")
# Log Files
TRAFFIC_LOG = "traffic_gen.log"
STRESS_LOG  = "stress_script.log"  # (For future use if you auto-run stresser)

# =================================================

# processes_to_monitor = ["stresser/stress_tester2.py", "trafficGenerator/test_traffic.py"]
# processes_to_monitor = [f"{pwd}/stresser/final_stresser_&_data_Scrapper.py", f"{pwd}/trafficGenerator/trafficGenerator.py"]
# processes_to_monitor = [f"{pwd}/trafficGenerator/trafficGenerator.py"]
containers_to_check = ["srscu0", "srscu1", "srscu2", "srsdu2", "srsdu1", "srsdu0", "srsdu3", "srsdu4", "srsdu5"]
containers_to_check_logs = ["srsue0", "srsue1", "srsue2", "srsue3", "srsue4", "srsue5"]
log_keyword = "Received RRC Release"

def wait_for_log(container_name, log_string):
    while True:
        # Fetch logs and check for the presence of the log string
        logs = subprocess.check_output(["docker", "logs", container_name], text=True)
        if log_string in logs:
            print(f"Log message '{log_string}' found in {container_name} logs. Proceeding...")
            break
        print(f"Waiting for log message '{log_string}' in container '{container_name}'...")
        time.sleep(2)  # Wait before checking again


# Kill all specified processes
def kill_processes():
    for proc in psutil.process_iter(attrs=['pid', 'cmdline']):
        try:
            if proc.info['cmdline']:
                for pname in processes_to_monitor:
                    if pname in " ".join(proc.info['cmdline']):
                        print(f"Killing process: {proc.info['cmdline']} (PID: {proc.info['pid']})")
                        proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

# Stop all running containers
def stop_containers():
    print("Stopping all running containers...")
    subprocess.run("docker stop $(docker ps -q)", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run("docker rm $(docker ps -aq)", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Start processes if they are not running
processes = {}

def start_processes():
    # Open log file in Append mode
    traffic_log_file = open(TRAFFIC_LOG, "a")
    
    for script in processes_to_monitor:
        script_found = False
        for pid, process in processes.items():
            if process.poll() is None:  # Process is running
                # Check if the running process matches the script
                try:
                    cmdline = process.args
                    if script in cmdline:
                        script_found = True
                        break
                except Exception:
                    pass

        if not script_found:
            print(f"Starting process: {script}")
            
            # Enable Logging: Redirect output to file instead of black hole
            if "test_traffic.py" in script:
                p = subprocess.Popen(["python3", script], stdout=traffic_log_file, stderr=traffic_log_file)
            if "stress_tester_icc.py" in script:
                p = subprocess.Popen(["python3", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                p = subprocess.Popen(["python3", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            processes[p.pid] = p

# Restart Docker services
def restart_services():
    print("Restarting RAN...")

    if(resource_constraint=='yes'):
        # subprocess.run(f"cd {pwd}/RIC/oran-sc-ric && docker compose up -d", shell=True)
        subprocess.run(f"cd {pwd} && docker compose -f docker-compose-cu++.yaml up -d", shell=True)

        CU_Count = 0
        DU_Count = 0

        while(CU_Count<2):
            wait_for_log(f'srscu{CU_Count}', 'F1-C')
            CU_Count += 1

        subprocess.run(f"cd {pwd} && docker compose -f docker-compose-du.yaml up -d", shell=True)
        
        while(DU_Count<3):
            wait_for_log(f'srsdu{DU_Count}', '==== DU started ===')
            DU_Count += 1
        subprocess.run(f"cd {pwd} && docker compose -f docker-compose-ue++.yaml up -d", shell=True)
    else:
        # subprocess.run(f"cd {pwd}/RIC/oran-sc-ric && docker compose up -d", shell=True)
        subprocess.run(f"cd {pwd} && docker compose -f docker-compose.yaml up -d", shell=True)
    
# Restart monitoring services
def restart_monitoring():
    print("Restarting monitoring services...")
    subprocess.run("docker rm -f prometheus cadvisor node-exporter", shell=True)
    subprocess.run(
        "docker pull prom/prometheus:latest && docker run -d --name=prometheus --network=oran-intel -p 9090:9090 "
        "-v=$PWD/setup/prometheus:/prometheus-data prom/prometheus:latest --config.file=/prometheus-data/prometheus.yml",
        shell=True
    )
    subprocess.run(
        "docker pull gcr.io/cadvisor/cadvisor:latest && docker run --name=cadvisor --network=oran-intel "
        "--volume=/:/rootfs:ro --volume=/var/run:/var/run:rw --volume=/sys:/sys:ro --volume=/var/lib/docker/:/var/lib/docker:ro "
        "--publish=8080:8080 --detach=true gcr.io/cadvisor/cadvisor:latest",
        shell=True
    )
    # subprocess.run("docker run -d --name=node-exporter --network=oran-intel -p 9100:9100 prom/node-exporter:latest", shell=True)
    subprocess.run(
    'docker run -d --name=node-exporter '
    '--network=oran-intel '
    '-p 9100:9100 '
    '--pid=host '
    '-v /proc:/host/proc:ro '
    '-v /sys:/host/sys:ro '
    '-v /:/host/root:ro '
    'prom/node-exporter:latest '
    '--path.procfs=/host/proc '
    '--path.sysfs=/host/sys '
    '--path.rootfs=/host/root',
    shell=True
)
    
def cleanup(signal_received=None, frame=None):
    print("\n[MONITOR] Received termination signal. Initiating full cleanup...")
    
    # 1. Force kill the child scripts immediately
    # We use 'pkill -9 -f' to find and destroy the specific python scripts by name.
    # This prevents them from hanging or running in the background.
    print("   -> Killing child scripts...")
    subprocess.run("pkill -9 -f stress_tester_icc.py", shell=True)
    subprocess.run("pkill -9 -f test_traffic.py", shell=True)
    
    # 2. Run your existing psutil cleanup (as a backup)
    kill_processes()
    
    # 3. Stop Docker containers
    stop_containers()
    
    print("[MONITOR] Cleanup complete. System is clean. Exiting.")
    sys.exit(0)

# Handle termination signals (Ctrl+C, kill)
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# Check container and process health
def check_system_health():
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            # Check running containers
            running_containers = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True).splitlines()
            for container_name in containers_to_check:
                if container_name not in running_containers:
                    print(f"{timestamp} WARNING: Container '{container_name}' is not running. Restarting it...")
                    # subprocess.run(f"docker start {container_name}", shell=True)
                    kill_processes()
                    restart_services()
                    start_processes()

            # Check container logs for errors
            for container_name in containers_to_check_logs:
                if container_name in running_containers:
                    logs = subprocess.check_output(["docker", "logs", "--tail", "50", container_name], text=True)
                    if log_keyword in logs:
                        print(f"{timestamp} WARNING: Found '{log_keyword}' in logs of container '{container_name}'. Restarting services...")
                        kill_processes()
                        restart_services()
                        start_processes()

            # Ensure processes are running correctly
            running_processes = []
            for proc in psutil.process_iter(attrs=['pid', 'cmdline']):
                try:
                    if proc.info['cmdline']:
                        running_processes.append(" ".join(proc.info['cmdline']))  # Store full command line
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            for script in processes_to_monitor:
                script_found = any(script in cmd for cmd in running_processes)
                
                if not script_found:
                    print(f"{timestamp} WARNING: Process '{script}' is not running. Restarting it...")
                    subprocess.Popen(["python3", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        except Exception as e:
            print(f"{timestamp} ERROR: {e}. Restarting services...")
            kill_processes()
            restart_services()
            start_processes()

        time.sleep(3)

if __name__ == "__main__":
    print("Starting monitoring system...")  
    kill_processes()  # Ensure no duplicate processes exist
    restart_services()  # Restart network services
    restart_monitoring()  # Restart monitoring stack
    start_processes()  # Ensure required Python scripts are running
    check_system_health()  # Start system monitoring
