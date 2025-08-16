import configs
import sys
import numpy as np
import logging, csv
from time import time
from board import IBoard, Jetson
from configs import PMODE_ID, MODELS, ALPHAS, CONSTRAINTS, SPEC_ORDER, LOGGING_LEVEL

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S %p', stream=sys.stdout, level=LOGGING_LEVEL)

class Agent():
    def __init__(self, board: IBoard):
        self.client = board
        self.round_num = 0
        self.cl_index = len(board.CONCURRENCY)-1
        self.last_cl = {}
        self.best_cl = {}
    
    def get_cl_setting(self):
        cl = self.client.CONCURRENCY[self.cl_index]
        return np.array(cl)

    def calculate(self):
        reward = None
        measured_metrics = self.client.run_inference()
        if measured_metrics:
            power, throughput, score = measured_metrics['power'], measured_metrics['throughput'], measured_metrics['score']
            if (ALPHA == 0 and throughput < THROUGHPUT_TARGET) or (ALPHA == 1 and power > (POWER_BUDGET)) or (ALPHA == 2 and score < SCORE_THRESHOLD):
                reward = None
            elif (ALPHA == 0 and throughput > THROUGHPUT_TARGET) or (ALPHA == 1 and power < (POWER_BUDGET)) or (ALPHA == 2 and score > SCORE_THRESHOLD):
                reward = (1 - ALPHA) * (throughput / THROUGHPUT_TARGET) + ALPHA * (POWER_BUDGET / power) if ALPHA != 2 else (score / SCORE_THRESHOLD)
            elif ALPHA == None and not (power > (POWER_BUDGET) or throughput < THROUGHPUT_TARGET or score < SCORE_THRESHOLD):
                reward = 1/3 * (throughput / THROUGHPUT_TARGET) + 1/3 * (POWER_BUDGET / power) + 1/3 * (score / SCORE_THRESHOLD)

            if reward:
                self.last_cl[self.client.power_mode] = [str(self.get_cl_setting()), self.cl_index]
                if not self.client.power_mode in self.best_cl or reward > self.best_cl[self.client.power_mode][0]:
                    self.best_cl[self.client.power_mode] = [reward, self.cl_index, str(self.get_cl_setting())]
    
    def brute(self):
        self.start_time = time()
        logging.info("agent started to brute the better configuration")

        while self.round_num < configs.TOTAL_EPSIODES:
            self.round_num += 1
            for cl in range(len(self.client.CONCURRENCY)):
                for pmode in list(PMODE_ID.keys()):
                    self.cl_index = cl
                    self.client.set_cl_settings(self.get_cl_setting())
                    self.client.power_mode = pmode
                    self.calculate()
            
        self.stop_time = time()
        elapsed = round(self.stop_time - self.start_time, 5)
        logging.info(f"agent finished after {self.round_num} steps, {elapsed}s")


def main():
    global ALPHA, POWER_BUDGET, THROUGHPUT_TARGET, SCORE_THRESHOLD

    result = []
    for model in MODELS:
        for const in CONSTRAINTS:
            for alpha in ALPHAS:
                configs.MODEL_NAME = model["name"]
                POWER_BUDGET = SPEC_ORDER[const]["pb"]
                THROUGHPUT_TARGET = SPEC_ORDER[const]["throughputt"]
                SCORE_THRESHOLD = SPEC_ORDER[const]["scth"]
                ALPHA = alpha
                
                print(model["name"], ALPHA, const, POWER_BUDGET, THROUGHPUT_TARGET, SCORE_THRESHOLD)
                
                board_client = Jetson()
                agent = Agent(board_client)
                agent.brute()
                
                for eval_pmode in PMODE_ID.keys():
                    agent.client.power_mode = eval_pmode
                    configs.TOTAL_EPSIODES = 500
                    if eval_pmode in agent.best_cl.keys():
                        agent.cl_index = agent.best_cl[eval_pmode][1]
                        agent.brute()

                for pmode in PMODE_ID.keys():
                    if pmode in agent.last_cl.keys():
                        if agent.round_num == configs.TOTAL_EPSIODES:
                            row_low = agent.client.client.run_inference(agent.last_cl[pmode][0], pmode)
                            result.append({"model": model["name"], "pmode": pmode, "power_budget": POWER_BUDGET, "throughput_target": THROUGHPUT_TARGET, "score_threshold": SCORE_THRESHOLD, "alpha": ALPHA, "constraint_level": const,
                                "CL": str(agent.last_cl[pmode][0]), "throughput": row_low["throughput"], "power": row_low["power"], "score" :row_low["score"]})
                            
                            with open('results/result_bruteforce.csv', 'w', newline='') as output_file:
                                dict_writer = csv.DictWriter(output_file, result[0].keys())
                                if output_file.tell() == 0:  # Check if the file is empty
                                    dict_writer.writeheader()  # Write the header only if the file is empty
                                dict_writer.writerows(result)

if __name__ == "__main__":
    main()