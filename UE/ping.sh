#!/bin/bash

# Check if an argument (IP address) is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <IP_ADDRESS>"
  exit 1
fi

IP_ADDRESS=$1

# Start pinging the provided IP address
while true; do
  ping $IP_ADDRESS
  sleep 2
done
