import pandas as pd
import matplotlib.pyplot as plt
import ast
from collections import defaultdict

title = 'Efficientdetd0 QoS: tight-power'

def avg(value):
    return sum(value) / len(value)

# Load the .csv file
file_path = ['rewards_qlearning.csv', 'rewards_a2c.csv', 'rewards_reinforce.csv', 'rewards_sarsa.csv']
df_q = pd.read_csv(file_path[0])
df_q = df_q[df_q['xlabel'] == title]
df_a2c = pd.read_csv(file_path[1])
df_a2c = df_a2c[df_a2c['xlabel'] == title]
df_re = pd.read_csv(file_path[2])
df_re = df_re[df_re['xlabel'] == title]
df_sa = pd.read_csv(file_path[3])
df_sa = df_sa[df_sa['xlabel'] == title]

# Initialize defaultdict to store rewards
rewards = defaultdict(lambda: defaultdict(list))

# Initialize list to store labels
labels = []

# Iterate over dataframes
for label, df in [('q_learning', df_q), ('a2c', df_a2c), ('reinforce', df_re), ('sarsa', df_sa)]:
    labels.append(label)
    for index, row in df.iterrows():
        for d in ast.literal_eval(row['data']):
            for key, value in d.items():
                rewards[label][key].append(value)
        rewards[label]['Time Elapsed'].append(row['Time Elapsed'])

# If you want to calculate average rewards for each key, you can do so
average_value = defaultdict(dict)
for model, data in rewards.items():
    for key, values in data.items():
        if key != 'Time Elapsed':
            average_value[model][key] = avg(values)
        else:
            average_value[model][key] = values[0]  # Extract the single value for 'Time Elapsed'

# Define colors for each model
colors = {
    'q_learning': 'orange',
    'a2c': 'blue',
    'reinforce': 'purple',
    'sarsa': 'brown'
}

# Define x-offsets for sizing dots
offsets = {
    'q_learning': (200, 10),
    'a2c': (120, 10),
    'reinforce': (60, 10),
    'sarsa': (20, -50)
}

# Create scatter plots
plt.figure(figsize=(10, 5))
legend_labels = {}  # Dictionary to track labels

for model, values in average_value.items():
    x_label = values['Time Elapsed']
    i = 0
    for key, re_ti in values.items():
        if key != 'Time Elapsed':
            y_label = re_ti
            i = i + 10
            if model not in legend_labels:
                plt.scatter(x_label, y_label, label=model, color=colors[model], s=offsets[model][0])
                legend_labels[model] = True
            else:
                plt.scatter(x_label, y_label, color=colors[model], s=offsets[model][0])
            plt.annotate(f'{key}', (x_label, y_label), textcoords="offset points", xytext=(0,offsets[model][1]+i), ha='center')

# Set the title and labels
plt.title(f'Rewards for {title}')
plt.xlabel('Time Elapsed')
plt.ylabel('Reward')

# Add legend
plt.legend()

# Show plot
plt.tight_layout()
plt.show()