import os
import random
import subprocess
import threading
import time
import numpy as np
import signal
import sys
import csv
import json

import csv
import threading
import time

import csv
import time

# Global list to track PIDs of stress processes
stress_pids = []
lock = threading.Lock()  # For thread-safe access to the stress_pids list
packetLoss_Containers = None
file = {}
save_thread = None

command = "docker ps -a --filter 'name=srscu' --filter 'name=srsdu' --format '{{.Names}}'"
container_ids = subprocess.check_output(command, shell=True).decode().strip().split('\n')


# Initialize the file dictionary
for cid in container_ids:
    file[cid] = [0, 0]


# def save_to_file():
#     """Function to save the 'file' dictionary to a CSV file every 45 seconds."""
#     while True:
#         with lock:
#             # Open the CSV file in write mode (this will overwrite the file)
#             with open('stresser/file_data.csv', 'w', newline='') as csvfile:
#                 # Define the fieldnames (dynamic columns based on container IDs)
#                 fieldnames = []

#                 # Dynamically create fieldnames for each container
#                 for cid in file.keys():
#                     fieldnames.append(f"{cid}_stressType")
#                     fieldnames.append(f"{cid}_stepStress")
                
#                 # Create the CSV writer
#                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
#                 # Write the header once (field names for each container)
#                 writer.writeheader()
                
#                 # Create a row with the current data
#                 row = {}
#                 for cid, (stress_type, step_stress) in file.items():
#                     row[f"{cid}_stressType"] = stress_type
#                     row[f"{cid}_stepStress"] = step_stress
                
#                 # Write the row into the CSV file
#                 writer.writerow(row)
#                 csvfile.flush()  # Ensure data is written immediately
            
#             # print("File saved to CSV.")

#         # Save every 45 seconds
#         time.sleep(45)

def save_to_file():
    """Save the 'file' dictionary to a CSV file every 45 seconds."""
    while True:
        with lock:
            # Open CSV in append mode
            with open('stresser/file_data.csv', 'a', newline='') as csvfile:
                fieldnames = [f"{cid}_stressType" for cid in file.keys()] + [f"{cid}_stepStress" for cid in file.keys()]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header only if file is empty
                if csvfile.tell() == 0:
                    writer.writeheader()

                # Write current state of stresses
                row = {}
                for cid, (stress_type, step_stress) in file.items():
                    row[f"{cid}_stressType"] = stress_type
                    row[f"{cid}_stepStress"] = step_stress

                writer.writerow(row)
        time.sleep(45)



# def ensure_stress_ng_installed(container_id):
#     """Ensure that stress-ng is installed in the container."""
#     try:
#         install_cmd = f"docker exec -it {container_id} bash -c 'apt update && apt install -y stress-ng'"
#         print(f"Installing stress-ng on container {container_id}: {install_cmd}")
#         os.system(install_cmd)
#     except Exception as e:
#         print(f"Error installing stress-ng on container {container_id}: {e}")

def get_container_name(container_id):
    # Run the docker inspect command to get the container details in JSON format
    command = f"docker inspect {container_id}"
    try:
        # Get the output from the command and decode it
        output = subprocess.check_output(command, shell=True).decode()
        # Convert the output (JSON) to a Python object (list of dictionaries)
        container_info = json.loads(output)
        
        # Extract the container name from the first (and only) element in the list
        container_name = container_info[0]['Name']
        
        # Remove the leading '/' from the container name
        container_name = container_name.lstrip('/')
        
        return container_name
    except subprocess.CalledProcessError as e:
        print(f"Error while fetching container name: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error while parsing JSON output: {e}")
        return None

def injectStress(container_id, typeOfStress, d, percentageOfStress, percentageOfStressEnd):
    """Inject stress into a Docker container based on the specified type."""
    global stress_pids
    # with lock:
        # file[container_id] = typeOfStress
    
    
    if (typeOfStress == 0 or (percentageOfStress==0 and percentageOfStressEnd==0)):
        # print(f"No stress applied to container {container_id} for {d}.")
        with lock:
            file[get_container_name(container_id)] = [0,0]
        time.sleep(d)
        return

    try:
        if typeOfStress == 1:  # CPU stress
            cpu_cores = 2
            remaining_time = d
            current_stress = percentageOfStress

            while remaining_time > 0:
                step_duration = random.randint(0, remaining_time)
                remaining_time -= step_duration
                step_stress = random.randint(current_stress, percentageOfStressEnd)

                with lock:
                    file[get_container_name(container_id)] = [typeOfStress, step_stress]
                stress_cmd = f"docker exec -it {container_id} bash -c 'stress-ng --cpu {cpu_cores} --cpu-load {step_stress} --timeout {step_duration}s'"
                # print(f"Applying CPU stress on container {container_id}: {stress_cmd}")
                proc = subprocess.Popen(stress_cmd, shell=True)
                with lock:
                    stress_pids.append(proc.pid)
                proc.wait()
                current_stress = step_stress

        elif typeOfStress == 2:  # Memory stress
            remaining_time = d
            current_stress = percentageOfStress

            while remaining_time > 0:
                step_duration = random.randint(0, remaining_time)
                remaining_time -= step_duration
                step_stress = random.randint(current_stress, percentageOfStressEnd)

                with lock:
                    file[get_container_name(container_id)] = [typeOfStress, step_stress]
                stress_cmd = f"docker exec -it {container_id} bash -c 'stress-ng --vm 1 --vm-bytes {step_stress}% --timeout {step_duration}s'"
                # print(f"Applying Memory stress on container {container_id}: {stress_cmd}")
                proc = subprocess.Popen(stress_cmd, shell=True)
                with lock:
                    stress_pids.append(proc.pid)
                proc.wait()
                current_stress = step_stress

        elif typeOfStress == 3:  # Packet loss
            remaining_time = d
            current_stress = percentageOfStress

            cleanup_cmd = f"docker exec -it {container_id} bash -c 'tc qdisc del dev eth0 root netem'"
            os.system(cleanup_cmd)

            while remaining_time > 0:
                step_duration = random.randint(0, remaining_time)
                remaining_time -= step_duration
                step_stress = random.randint(current_stress, percentageOfStressEnd)

                try:
                    tc_cmd = f"docker exec -it {container_id} bash -c 'tc qdisc add dev eth0 root netem loss {step_stress}%'"
                    # print(f"Applying Packet Loss on container {container_id}: {tc_cmd}")
                    proc = subprocess.Popen(tc_cmd, shell=True)
                except Exception as e:
                    print(e)
                    os.system(cleanup_cmd)
                    proc = subprocess.Popen(tc_cmd, shell=True)

                with lock:
                    file[get_container_name(container_id)] = [typeOfStress, step_stress]

                with lock:
                    stress_pids.append(proc.pid)
                time.sleep(step_duration)
                os.system(cleanup_cmd)  # Cleanup after each step
                current_stress = step_stress
            os.system("python3 pl_deleter.py")

        print(f'File is {file} stress duration is {d} and sttress type is {typeOfStress}')

    except subprocess.CalledProcessError as e:
        print(f"Subprocess error applying stress to container {container_id}: {e}")
    except Exception as e:
        print(f"Error applying stress to container {container_id}: {e}")


def cleanup_stress():
    """Stop all stress processes and cleanup."""
    global stress_pids
    print("\nCleaning up stress processes...")
    with lock:
        for pid in stress_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Terminated process with PID {pid}")
            except OSError as e:
                print(f"Error terminating process with PID {pid}: {e}")
        stress_pids.clear()
    sys.exit(0)


def signal_handler(sig, frame):
    """Handle interrupt signal to perform cleanup."""
    cleanup_stress()


def main():
    global stress_pids
    signal.signal(signal.SIGINT, signal_handler)  # Register the signal handler

    max_duration = 48 * 60 * 60  # 24 hours in seconds
    sum_duration = 0

    # max_duration = 10*60

    cmd = "docker ps --filter 'name=^srscu' --filter 'name=^srsdu' --format '{{.ID}}'"
    container_ids = subprocess.check_output(cmd, shell=True).decode().splitlines()
    # print(f"Fetched container IDs: {container_ids}")
    packetLoss_Containers = container_ids
    # for container_id in container_ids:
    #     ensure_stress_ng_installed(container_id)
    # Start the background thread to save the 'file' dictionary every 500ms
    global save_thread
    save_thread = threading.Thread(target=save_to_file)
    save_thread.daemon = True  # This makes sure the thread will exit when the main program exits
    save_thread.start()

    while sum_duration <= max_duration:
        duration = int(np.random.exponential(scale=20) + 1)
        duration = min(max(duration, 120), 300)
        duration = min(max(duration, 10), 30)
        sum_duration += duration

        threads = []
        type_probs = [0.25, 0.25, 0.25, 0.25]
        # type_probs = [1,0,0,0]
        percentage_of_stress_start=0
        percentage_of_stress_end=0
        type_of_stress = np.random.choice([0, 1, 2, 3], p=type_probs)
        for container_id in container_ids:   
            is_stress = np.random.choice([0, 1], p=[0.6,0.4])
            if is_stress == 0:
                percentage_of_stress_start = 0
                percentage_of_stress_end = 0
            elif type_of_stress == 1:
                percentage_of_stress_start = np.random.randint(40, 91)
                percentage_of_stress_end = np.random.randint(percentage_of_stress_start, 101)
            elif type_of_stress == 2:
                percentage_of_stress_start = np.random.randint(25, 36)
                percentage_of_stress_end = np.random.randint(percentage_of_stress_start, 61)
            elif is_stress==1 and type_of_stress == 3:
                percentage_of_stress_start = np.random.randint(1, 3)
                percentage_of_stress_end = np.random.randint(percentage_of_stress_start, 6)

            t = threading.Thread(
                target=injectStress,
                args=(container_id, type_of_stress, duration, percentage_of_stress_start, percentage_of_stress_end)
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()
