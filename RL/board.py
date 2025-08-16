import pandas as pd
import configs
import json
import numpy as np
from scipy.stats import norm, sem
from abc import ABC, abstractmethod
from statistics import mean, median

# def avg(value):
#     if not value.empty:
#         return mean(value)

# # Function to calculate bootstrap confidence interval for the median
# def ci_median(data, confidence_level=0.95):
#     """
#     Calculate the confidence interval for the median of a dataset.

#     Parameters:
#     - data (list): The dataset to calculate the confidence interval for.
#     - confidence_level (float): The confidence level for the interval (default is 0.95).

#     Returns:
#     - lower_bound (float): The lower bound of the confidence interval.
#     - upper_bound (float): The upper bound of the confidence interval.
#     """
#     if data.any():
#         # Calculate the sample median
#         median = np.median(data)

#         # Calculate the standard error of the median
#         se_median = sem(data)

#         # Calculate the z-critical value for the desired confidence level
#         z_critical = norm.ppf(0.5 + confidence_level / 2)

#         # Calculate the confidence interval for the median
#         lower_bound = median - z_critical * se_median
#         upper_bound = median + z_critical * se_median

#         return lower_bound, upper_bound
#     else:
#         raise ValueError(f"Data input {data.any()}")

def read_JSON(filename):
    with open(filename, 'r') as f:
        return json.load(f)

class IBoard(ABC):

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def CONCURRENCY(self):
        pass

    @abstractmethod
    def set_cl_settings(self, new_cl):
        pass
    
    @abstractmethod
    def get_power_mode(self):
        pass
    
    @abstractmethod
    def run_inference(self):
        pass

class Jetson(IBoard):
    CONCURRENCY = [1, 2, 3]

    def __init__(self) -> None:
        super().__init__()
        self.client = LearnOffPolicy(configs.DEVICE_NAME)
        self.power_mode = None
    
    def set_cl_settings(self, new_cl):
        self.current_cl = str(new_cl)

    def get_power_mode(self):
        return self.power_mode
        
    def run_inference(self):
        return self.client.run_inference(self.current_cl, self.power_mode)

class LearnOffPolicy:
    def __init__(self, device_name):
        self.DATA = f"../dataset/{device_name}/od_perf/dataset_od_coco.csv"
        self.POWER_MODE = configs.PMODE_ID
        file_csv = pd.read_csv(self.DATA)
        file_csv = file_csv[file_csv["Model"] == configs.MODEL_NAME]
        self.collected_data = {}
        for pmode in self.POWER_MODE.keys():
            self.collected_data[pmode] = {}
            for cl in Jetson.CONCURRENCY:
                if not file_csv[(file_csv["Mode"] == pmode) & (file_csv["CL"] == cl)].empty:
                    data = {}
                    data["throughput"] = float(file_csv["Median of Throughput"][(file_csv["Mode"] == pmode) & (file_csv["CL"] == cl)].values.item())
                    data["power"] = float(file_csv["Median of Power"][(file_csv["Mode"] == pmode) & (file_csv["CL"] == cl)].values.item())
                    data["score"] = float(file_csv["Average of mAP"][(file_csv["Mode"] == pmode) & (file_csv["CL"] == cl)].values.item())
                    self.collected_data[pmode][str(cl)] = data

        # print(self.collected_data)
        # out_file = open("raw_collected_data.json", "w")
        # json.dump(self.collected_data, out_file, indent=4)
        # out_file.close()
    
    def run_inference(self, cl, pmode):
        data = {}
        if pmode in self.collected_data.keys() and cl in self.collected_data[pmode].keys():
            data = self.collected_data[pmode][cl]

        metrics = {}
        if data:
            metrics["throughput"] = data["throughput"]
            metrics["power"] = data["power"]
            metrics["score"] = data["score"]
            return metrics

if __name__ == "__main__":
    Jetson()