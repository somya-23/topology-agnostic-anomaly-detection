import subprocess
import time
from datetime import datetime
import sys
import subprocess


# Define the screen command to reattach to the 'RAN' session and send commands
commands = [
    # Kill existing processes
    
    "screen -r OTHERS -X stuff \"cd /home/newsl/workspace/O-RAN_srsRAN && ps aux | grep 'stresser.py' | grep -v grep | awk '{print $2}' | xargs kill -9\n\"",
    "screen -r OTHERS -X stuff \"cd /home/newsl/workspace/O-RAN_srsRAN && ps aux | grep 'trafficGenerator.py' | grep -v grep | awk '{print $2}' | xargs kill -9\n\"",
    "screen -r OTHERS -X stuff \"cd /home/newsl/workspace/O-RAN_srsRAN && ps aux | grep 'dataScrapper.py' | grep -v grep | awk '{print $2}' | xargs kill -9\n\"",

    # Stop running services using Ctrl+C
    # "screen -r RIC -X stuff $'\003'",  # Send Ctrl+C to stop RIC
    "screen -r RAN -X stuff $'\003'",  # Send Ctrl+C to stop RAN

    # Restart RIC and RAN
    # "screen -r RIC -X stuff \"cd /home/newsl/workspace/O-RAN_srsRAN/RIC/oran-sc-ric && docker compose up\n\"",
    "screen -r RAN -X stuff \"cd /home/newsl/workspace/O-RAN_srsRAN && docker compose up\n\"",

    # Restart Monitoring Services
    "screen -r MON -X stuff \"cd /home/newsl/workspace/O-RAN_srsRAN && docker rm -f prometheus cadvisor node-exporter\n\"",
    "screen -r MON -X stuff \"cd /home/newsl/workspace/O-RAN_srsRAN && docker pull prom/prometheus:latest && docker run -d --name=prometheus --network=oran-intel -p 9090:9090 -v=$PWD/setup/prometheus:/prometheus-data prom/prometheus:latest --config.file=/prometheus-data/prometheus.yml && docker pull gcr.io/cadvisor/cadvisor:latest && docker run --name=cadvisor --network=oran-intel --volume=/:/rootfs:ro --volume=/var/run:/var/run:rw --volume=/sys:/sys:ro --volume=/var/lib/docker/:/var/lib/docker:ro --publish=8080:8080 --detach=true gcr.io/cadvisor/cadvisor:latest && docker run -d --name=node-exporter --network=oran-intel -p 9100:9100 prom/node-exporter:latest\n\"",

    # Start Python scripts in the OTHERS screen session
    "screen -r OTHERS -X stuff $'\003'",  # Send Ctrl+C to stop OTHERS
    "screen -r OTHERS -X stuff \"cd /home/newsl/workspace/O-RAN_srsRAN && nohup python3 stresser/stresser.py > /dev/null 2>&1 &\n\"",
    "screen -r OTHERS -X stuff \"cd /home/newsl/workspace/O-RAN_srsRAN && nohup python3 trafficGenerator/trafficGenerator.py > /dev/null 2>&1 &\n\"",
    "screen -r OTHERS -X stuff \"cd /home/newsl/workspace/O-RAN_srsRAN && nohup python3 dataScrapper/dataScrapper.py > /dev/null 2>&1 &\n\"",
]

    
def monitor_containers():
    containers_to_check = ["srscu0", "srscu1", "srsdu2", "srsdu1", "srsdu0"]
    containers_to_check_logs = ["srsue0", "srsue1", "srsue2"]
    log_keyword = "Received RRC Release"

    while True:
        while True:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                # Verify containers are running using docker ps
                running_containers = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True).splitlines()
                # print(running_containers)
                for container_name in containers_to_check:
                    if container_name not in running_containers:
                        print(f"{timestamp} WARNING: Container '{container_name}' is not running.")
                        # Execute the commands in sequence
                        print("Running commands")
                        for command in commands:
                            result = subprocess.run(command, capture_output=True, text=True, shell=True)
                        return
                for container_name in containers_to_check_logs:
                    if container_name in running_containers:
                        try:
                            logs = subprocess.check_output(["docker", "logs", "--tail", "50", container_name], text=True)
                            if log_keyword in logs:
                                print(f"{timestamp} WARNING: Found '{log_keyword}' in logs of container '{container_name}'.")
                                # Execute the commands in sequence
                                print("Running commands")
                                for command in commands:
                                    result = subprocess.run(command, capture_output=True, text=True, shell=True)
                                return
                        except subprocess.CalledProcessError:
                            print(f"{timestamp} WARNING: Failed to retrieve logs for container '{container_name}'.")
                            # Execute the commands in sequence
                            print("Running commands")
                            for command in commands:
                                result = subprocess.run(command, capture_output=True, text=True, shell=True)
                            return
            except Exception as e:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} An error occurred: {e}")
                # Execute the commands in sequence
                print("Running commands")
                for command in commands:
                    result = subprocess.run(command, capture_output=True, text=True, shell=True)
                    if("RAN" in command):
                        sleep(3)
                return
            time.sleep(1)

if __name__ == "__main__":
    monitor_containers()
