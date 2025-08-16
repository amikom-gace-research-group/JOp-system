import csv
import sys
import os

def extract_numericals(s):
    lst = [word.split(' ')[-1] for word in s.split(',')]
    numerical_lst = []
    for item in lst:
        if '%' in item:
            numerical_lst.append(float(item.rstrip('%')))
        elif any(unit in item for unit in ['MB', 'mW']):
            numerical_lst.append(float(item[:-2]))
        elif 'bread/s' in item:
            numerical_lst.append(float(item.split('bread/s')[0]))  # Extract the numerical part before 'bread/s'
        elif 'bwrtn/s' in item:
            numerical_lst.append(float(item.split('bwrtn/s')[0]))  # Extract the numerical part before 'bwrtn/s'
        else:
            try:
                numerical_lst.append(float(item))
            except ValueError:
                numerical_lst.append(item)  # Keep the value unchanged if not convertible to float
    return numerical_lst

# Open the input text file
with open(sys.argv[3], 'r') as infile:
    modfile = 'a' if os.path.exists(sys.argv[4]) else 'w'
    with open(sys.argv[4], modfile, newline='') as outfile:
        # Specify the order of columns
        fieldnames = ["Resolution", "Concurrency Level", "FPS", "Mode", "CPU User (%)", "CPU System (%)", "Memory (MB)", "Power (mW)", "GPU Utilization (%)", "IO Read (b/s)", "IO Write (b/s)"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        if modfile == 'w':
            writer.writeheader()

        # Read each line from the input file
        for line in infile:
            line_data = extract_numericals(line)
            line_data.insert(0, sys.argv[1])
            line_data.insert(1, sys.argv[2])
            # Create a dictionary with fieldnames as keys and line_data as values
            data = dict(zip(fieldnames, line_data))
            
            # Write the data to the CSV file
            writer.writerow(data)
