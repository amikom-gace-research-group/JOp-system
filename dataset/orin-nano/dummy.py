import os
import random
import pandas as pd

# Define the ranges for each parameter
power_modes = [
    '15W',
    '7W'
]
concurrency_levels = [1, 2]
resolutions = {
    '1080p': {'fps_factor': 0.8},
    '720p': {'fps_factor': 0.9},
    '480p': {'fps_factor': 1.0}
}

# Generate 50 additional data points
additional_data = []
for _ in range(50):
    power_mode = random.choice(power_modes)
    concurrency_level = random.choice(concurrency_levels)
    resolution = random.choice(list(resolutions.keys()))

    fps_factor = resolutions[resolution]['fps_factor']

    power_consumption = round(random.uniform(5000, 9000), 2)
    fps = round(random.uniform(3, 6) * fps_factor, 2)

    additional_data.append({
        "Model": "Dummy", "Power Mode" :power_mode, "Concurrency Level" :concurrency_level, 
        "Resolution" :resolution, "FPS" :fps, "Power": power_consumption
    })

data = pd.DataFrame(additional_data)
data.to_csv('dummy.csv')
if os.path.exists('dummy.csv'):
    data.to_csv('dummy.csv', mode='a', header=False)