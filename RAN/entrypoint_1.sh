#!/bin/bash

# Find the interface that belongs to the oran-intel network (175.x.x.x)
IFACE=$(ip -o addr show | grep "175\." | awk '{print $2}' | head -1)
IFACE2=$(ip -o addr show | grep "171\." | awk '{print $2}' | head -1)


# Add the secondary IP address on the correct interface
ip addr add 175.53.1.12/16 dev ${IFACE}

ip link set dev ${IFACE} down
ip link set dev ${IFACE2} down

# Use temp names to avoid "File exists" when interfaces need swapping
ip link set dev ${IFACE} name tmp_oran
ip link set dev ${IFACE2} name tmp_f1
ip link set dev tmp_oran name eth0
ip link set dev tmp_f1 name eth1

ip link set dev eth0 up
ip link set dev eth1 up


# Execute the provided command (CMD from the Dockerfile)
exec "$@"

