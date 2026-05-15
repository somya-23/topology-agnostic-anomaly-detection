import json

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


# Function to modify lines
def modify_line(line):
    # Apply the replacements
    line = line.replace('instance="$node"', 'instance="node-exporter:9100"')
    line = line.replace('job="$job"', 'job="node"')
    line = line.replace ('"$diskdevices"', '"nvme0n1"')            
    line = line.replace('[$__rate_interval]', '[10s]')
    # line = line.replace('expr', )
    return line



def main():
    # Load the Grafana JSON file
    input_file = 'grafananode.json' 
    mid_file='tmp.txt'    # Replace with your actual file path
    output_file = '../Node.txt'

    # Open and parse the JSON file
    with open(input_file, 'r') as f:
        grafana_data = json.load(f)
        expr_list = extract_expressions(grafana_data)
        
    # Write to a CSV file
    with open(mid_file, 'w', newline='') as f:
        # f.write("expr\n")  # Header
        f.writelines(f"{expr}\n" for expr in expr_list)

    # with open(mid_file, 'r') as mid:
    #     # finalLines = []
    #     lines = mid.readlines()
    #     for line in lines:
    #         if("fs" not in line):
    #             finalLines.append(line)

    print(f"Extracted expressions saved to '{output_file}'")
    # Read, modify, and write lines
    
    with open(output_file, 'w') as outfile, open(mid_file, 'r') as midfile:
        for line in midfile.readlines():
            modified_line = modify_line(line)
            # if(modified_line!='expr\n' or modified_line!='\n'):
            outfile.write(modified_line)


    print(f"Modified text saved to '{output_file}'")

main()
            
            
   
