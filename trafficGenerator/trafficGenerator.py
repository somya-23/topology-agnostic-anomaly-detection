import subprocess
import random
import time
import signal
import sys

# Global dictionaries to keep track of processes
ping_pids = {}
iperf_client_pids = {}
iperf_server_pid = None
docker_ids = []
open5gs_container_id = None
previous_traffic = 0
current_traffic = None

result = subprocess.run("pwd", capture_output=True, text=True)
pwd = result.stdout.strip()
print(f'Current working directory is {pwd}')

def generate_random_sizes(total_ping_size, num_sizes, min_size=10000, max_size=60000):
    sizes = []
    sum_sizes = 0

    for _ in range(num_sizes - 1):
        size = random.randint(min_size, max_size)
        sizes.append(size)
        sum_sizes += size

    last_size = total_ping_size - sum_sizes
    if last_size > max_size:
        last_size = max_size
    elif last_size < min_size:
        last_size = min_size
    sizes.append(last_size)

    return sizes

def get_ping_target_ip(container_id):
    try:
        cmd = (
            f"docker exec -it {container_id} bash -c 'ip a show tun_srsue | "
            f"grep \"inet \" | awk \"{{print \\$2}}\" | cut -d\"/\" -f1'"
        )
        result = subprocess.check_output(cmd, shell=True, text=True).strip()
        if not result:
            raise ValueError("Empty result for IP address")
        ip_parts = result.split('.')
        if len(ip_parts) != 4 or not all(part.isdigit() for part in ip_parts):
            raise ValueError(f"Invalid IP address format: {result}")
        ip_int = (int(ip_parts[0]) << 24) + (int(ip_parts[1]) << 16) + (int(ip_parts[2]) << 8) + int(ip_parts[3])
        ip_int -= 1
        if ip_int < 0:
            raise ValueError(f"Resulting IP address is invalid after decrement: {result}")
        decremented_ip = f"{(ip_int >> 24) & 0xFF}.{(ip_int >> 16) & 0xFF}.{(ip_int >> 8) & 0xFF}.{ip_int & 0xFF}"
        # return decremented_ip
        return "10.45.1.1"
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error retrieving or processing IP address for container {container_id}: {e}")
        return None

def start_ping(container_id, size, target_ip):
    try:
        cmd = (
            f"docker exec {container_id} bash -c "
            f"'ping -i 0.1 -s {size} {target_ip} > /dev/null 2>&1 & echo $!'"
        )
        pid = subprocess.check_output(cmd, shell=True, text=True).strip()
        if not pid.isdigit():
            raise ValueError(f"Unexpected PID output: {pid}")
        return pid
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error starting ping in container {container_id}: {e}")
        return None

def stop_ping(container_id, pid):
    try:
        cmd = f"docker exec {container_id} bash -c 'kill -9 {pid}'"
        subprocess.run(cmd, shell=True, check=True)
        # print(f"Stopped ping in container {container_id} (PID: {pid})")
    except subprocess.CalledProcessError:
        print(f"Error stopping ping in container {container_id}")

def get_docker_ids():
    try:
        cmd = "docker ps --filter 'name=^/srsue' --format '{{.ID}}'"
        result = subprocess.check_output(cmd, shell=True, text=True)
        return result.strip().split('\n') if result.strip() else []
    except subprocess.CalledProcessError:
        print("Error retrieving Docker container IDs")
        return []

def setup_iperf_server(container_id):
    try:
        cmd = (
            f"docker exec {container_id} bash -c 'iperf3 -s > /dev/null 2>&1 & echo $!'"
        )
        pid = subprocess.check_output(cmd, shell=True, text=True).strip()
        # print(f"iperf3 server started in container {container_id} with PID {pid}")
        return pid
    except subprocess.CalledProcessError as e:
        print(f"Error starting iperf3 server in container {container_id}: {e}")
        return None

def start_iperf_client(container_id, target_ip):
    try:
        cmd = (
            f"docker exec {container_id} bash -c 'iperf3 -c {target_ip} -t 0 > /dev/null 2>&1 & echo $!'"
        )
        pid = subprocess.check_output(cmd, shell=True, text=True).strip()
        # print(f"iperf3 client started in container {container_id} targeting {target_ip} with PID {pid}")
        return pid
    except subprocess.CalledProcessError as e:
        print(f"Error starting iperf3 client in container {container_id}: {e}")
        return None

def stop_process(container_id, pid):
    try:
        cmd = f"docker exec {container_id} bash -c 'kill -9 {pid}'"
        subprocess.run(cmd, shell=True, check=True)
        # print(f"Process {pid} stopped in container {container_id}")
    except subprocess.CalledProcessError as e:
        print(f"Error stopping process {pid} in container {container_id}: {e}")

def cleanup(signum=None, frame=None):
    """Clean up all running processes."""
    # print("\nCleaning up...")
    if iperf_server_pid:
        stop_process(open5gs_container_id, iperf_server_pid)

    for container_id, pid in ping_pids.items():
        stop_ping(container_id, pid)

    for container_id, pid in iperf_client_pids.items():
        stop_process(container_id, pid)

    sys.exit(0)

# Register signal handlers for cleanup
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def main():
    global ping_pids, iperf_server_pid, docker_ids, open5gs_container_id

    docker_ids = get_docker_ids()
    if not docker_ids:
        print("No Docker containers found.")
        return

    open5gs_container_id = subprocess.check_output(
        "docker ps -q -f name=open5gs", shell=True, text=True
    ).strip()
    if not open5gs_container_id:
        print("No open5gs container found.")
        return

    iperf_server_pid = setup_iperf_server(open5gs_container_id)

    distribution = [
        11.0, 8.1, 5.6, 3.6, 2.7, 1.9, 3.0, 5.0, 7.1, 11.1, 11.2, 11.9,
        12.4, 12.3, 13.0, 13.1, 12.9, 12.7, 12.4, 12.2, 12.0, 13.0, 14.0, 15.0, 14.0
    ]
    distribution = [val / 16 for val in distribution]

    i=0
    
    for hour, current_distribution in enumerate(distribution, start=1):
        temp_size = current_distribution * 1_000_000_000
        total_ping_size = int(temp_size // 3600)

        previous_traffic = distribution[i-1]
        current_traffic = distribution[i]


        # Write the list to a file (one item per line)
        with open(f'{pwd}/trafficGenerator/traffic_distribution.txt', 'w') as file:
            for item in [previous_traffic, current_traffic]:
                file.write(str(item) + '\n')


        # print(f"Half Hour: {hour}")
        # print(f"Current Distribution: {current_distribution}")
        # print(f"Total Ping Size: {total_ping_size} bytes")

        num_containers = len(docker_ids)
        sizes = generate_random_sizes(total_ping_size, num_containers)

        start_time = time.time()
        end_time = start_time + 3600

        while time.time() < end_time:
            ping_pids = {}
            for container_id, size in zip(docker_ids, sizes):
                target_ip = get_ping_target_ip(container_id)
                if not target_ip:
                    continue

                pid = start_ping(container_id, size, target_ip)
                if pid:
                    ping_pids[container_id] = pid

                if container_id not in iperf_client_pids:
                    client_pid = start_iperf_client(container_id, target_ip)
                    if client_pid:
                        iperf_client_pids[container_id] = client_pid

            time.sleep(5)

            for container_id, pid in ping_pids.items():
                stop_ping(container_id, pid)

            sizes = generate_random_sizes(total_ping_size, num_containers)

        previous_traffic = current_traffic
        i=i+1
        current_traffic = distribution[i]
if __name__ == "__main__":
    while True:
        main()
