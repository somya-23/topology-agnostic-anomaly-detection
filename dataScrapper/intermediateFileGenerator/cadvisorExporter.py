import json
import subprocess
import re

# Function to extract all "expr" values
def extract_expressions(data):
    expressions = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "expr":
                expressions.append(value)
            else:
                expressions.extend(extract_expressions(value))
    elif isinstance(data, list):
        for item in data:
            expressions.extend(extract_expressions(item))
    return expressions


# Function to fetch docker container names
def get_docker_container_names(password="darkangel"):
    try:
        # Prepare the command that echoes the password into sudo and runs docker ps
        command = f"docker ps --format '{{{{.Names}}}}'"
        
        # Run the command
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if the command executed successfully
        if result.returncode != 0:
            raise Exception(f"Error running docker ps: {result.stderr.decode('utf-8')}")
        
        # Decode the result and split it into lines (each container name on a separate line)
        container_names = result.stdout.decode('utf-8').splitlines()
        
        return list(set(container_names) - {'srsue4', 'srsue5','srsue3', 'srsue2', 'srsue1', 'srsue0', 'node-exporter', 'cadvisor', 'prometheus', 'grafana', 'influxdb', 'redis', 'metrics_server', 'ric_submgr', 'ric_e2term', 'python_xapp_runner', 'ric_dbaas', 'ric_appmgr', 'ric_rtmgr_sim'})
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# Function to fetch all network interfaces
def get_all_network_interfaces(password="darkangel"):
    try:
        # Prepare the command to get network interfaces
        command = f"ip a"
        
        # Run the command
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if the command executed successfully
        if result.returncode != 0:
            raise Exception(f"Error running ip a: {result.stderr.decode('utf-8')}")
        
        # Decode the result
        interfaces_output = result.stdout.decode('utf-8')
        
        # Extract the network interfaces (e.g., eth0, br-*, etc.)
        interfaces = re.findall(r'^\d+: (\S+):', interfaces_output, re.MULTILINE)
        
        # return interfaces
        return ["eth0", "eth1"]
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# Function to modify the query
def modify_query(query, container_name, interface):
    # Replace 'name' with the container name
    query = query.replace('name=~"$name"', f'name="{container_name}"')
    query = query.replace('interface="$interface"', f'interface="{interface}"')
    
    # Replace 'instance' with 'cadvisor:8080'
    query = query.replace('instance=~"$instance"', 'instance="cadvisor:8080"')
  
    # Remove 'image' part
    query = re.sub(r',image!=""', '', query)
    
    # Replace any interval with 1m
    query = re.sub(r'\[\$__rate_interval\]', '[10s]', query)
    
    pattern = r'without\s+\(dc,from,id,\$\{sum_without:csv\}\)'

    # Perform substitution
    query = re.sub(pattern, '', query)

    return query


# Main function to process queries from x.txt and save to cadvisor.txt
def process_queries(input_file='grafana_expressions2.txt', output_file='cadvisorQueries.txt'):
    # Get all docker container names
    container_names = get_docker_container_names()
    print("Docker Containers:", container_names)
    
    # Get all network interfaces
    network_interfaces = get_all_network_interfaces()
    print("Network Interfaces:", network_interfaces)

    # Read queries from input file
    with open(input_file, 'r') as f:
        queries = f.readlines()

    # Open the output file to write the modified queries
    with open(output_file, 'w') as f:
        print(len(network_interfaces))
        # For every container, only loop interfaces for queries that have an $interface placeholder.
        # Non-network queries are emitted once per container (no need to duplicate per interface).
        for container_name in container_names:
            for query in queries:
                query = query.strip()
                if "$interface" in query:
                    for interface in network_interfaces:
                        modified_query = modify_query(query, container_name, interface)
                        f.write(modified_query + '\n')
                else:
                    modified_query = modify_query(query, container_name, network_interfaces[0])
                    f.write(modified_query + '\n')

    print(f"Modified queries have been saved to {output_file}")


# Call the function to process queries
# process_queries()



def main():
    input_file = 'grafanaCadvisor.json'  # Replace with your actual file path
    output_file = 'Cadvisor.txt'
    final_file = '../promCadvisor.txt'
    
    # Load the Grafana JSON file
    with open(input_file, 'r') as f:
        grafana_data = json.load(f)
    
    # Extract all expressions from the JSON
    expr_list = extract_expressions(grafana_data)

    # Write to a CSV file
    with open(output_file, 'w', newline='') as f:
        f.writelines(f"{expr}\n" for expr in expr_list)
   
    process_queries(output_file,final_file)
    
    
main()