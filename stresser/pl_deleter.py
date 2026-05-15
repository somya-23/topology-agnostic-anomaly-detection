import subprocess
import os

# Run the Docker command to get the container IDs
command = "docker ps -a --filter 'name=srscu' --filter 'name=srsdu' -q"
container_ids = subprocess.check_output(command, shell=True).decode().strip().split('\n')
print(container_ids)
for cid in container_ids:
    cmd = f"docker exec -it {cid} bash -c 'tc qdisc del dev eth0 root netem' "
    os.system(cmd)