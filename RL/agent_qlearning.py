import logging, sys, csv
from time import time
from qtable import Action, QTable, QTableVisualizer
import numpy as np
import pandas as pd
from board import IBoard, Jetson
import configs
from configs import EPISLON, MIN_EPSILON, RUNTIME_GUARD, CPU_DENVER_0, CPU_DENVER_1,\
                    CPU_DENVER_2, GPU_FREQ, CPU_ONLINE, EMC_FREQ, LOGGING_LEVEL, MODELS,\
                    ALPHAS, CONSTRAINTS, SPEC_ORDER
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S %p', stream=sys.stdout, level=LOGGING_LEVEL)

ALPHA, POWER_BUDGET, THROUGHPUT_TARGET, SCORE_THRESHOLD = 0, 0, 0, 0

print("Using Q-Learning Algorithm")

class Agent():
    def __init__(self, board: IBoard):
        self.round_num = 0
        self.client = board
        self.epsilon = [EPISLON for _ in range(len(self.client.CONCURRENCY))]
        self.states = np.array(np.meshgrid(self.client.CONCURRENCY, CPU_DENVER_0, CPU_DENVER_1, CPU_DENVER_2, CPU_ONLINE, GPU_FREQ, EMC_FREQ, indexing='ij')).T.reshape(-1,7)
        self.RL = QTable(self.client, self.states, Action)
        self.client.cl = 1
        self.cpu_freq1_index = len(CPU_DENVER_0)-1
        self.cpu_freq2_index = len(CPU_DENVER_1)-1
        self.cpu_freq3_index = len(CPU_DENVER_2)-1
        self.cpu_core_index = len(CPU_ONLINE)-1
        self.gpu_freq_index = len(GPU_FREQ)-1
        self.mem_freq_index = len(EMC_FREQ)-1
        self.current_state = np.array([])
        self.measured_data = {}
        self.last_sets = {}
        self.best_sets = {}
        self.oracle_data = np.array([{} for i in range(len(self.client.CONCURRENCY))])
        self.num_of_violations_training = np.array([0 for i in range(len(self.client.CONCURRENCY))])
        self.num_of_violations_evaluations = 0
        self.evaluation = False
        self.toggled = False
        self.eval_data = {'throughput': [], 'power': [], 'score':[]}
        self.reward_history = []

    def clean(self):
        self.round_num = 0
        self.measured_data = {}
    
    def get_current_state(self):
        self.cl_index = self.get_current_cl_index()
        current_state = np.array([self.cl_index, *self.get_sets()])
        return current_state

    def get_current_state_index(self):
        cl_index = self.get_current_cl_index()
        return [cl_index, self.cpu_core_index, self.cpu_freq1_index, self.cpu_freq2_index, self.cpu_freq3_index, self.gpu_freq_index, self.mem_freq_index]
    
    def get_sets(self):
        cpu_cores = CPU_ONLINE[self.cpu_core_index]
        cpu_freq1 = CPU_DENVER_0[self.cpu_freq1_index]
        cpu_freq2 = CPU_DENVER_1[self.cpu_freq2_index]
        cpu_freq3 = CPU_DENVER_2[self.cpu_freq3_index]
        gpu_freq = GPU_FREQ[self.gpu_freq_index]
        mem_freq = EMC_FREQ[self.mem_freq_index]
        return np.array([cpu_cores, cpu_freq1, cpu_freq2, cpu_freq3, gpu_freq, mem_freq])
    
    def get_current_cl_index(self):
        cl = self.client.get_cl()
        return [idx for idx, key in enumerate(self.client.CONCURRENCY) if key == cl][0]
    
    def get_best_oracle_data(self, cl):
        cl_index = [idx for idx, key in enumerate(self.client.CONCURRENCY) if key == cl][0]
        data = []
        for sets in self.oracle_data[cl_index].keys():
            measured = self.oracle_data[cl_index][sets]
            measured['settings'] = sets
            if ALPHA == 0 and measured['throughput'] > THROUGHPUT_TARGET:
                data.append(measured)
            elif ALPHA == 1 and measured['power'] < POWER_BUDGET:
                data.append(measured)
            elif ALPHA == 2 and measured['score'] > SCORE_THRESHOLD:
                data.append(measured)
            elif ALPHA == None and measured['throughput'] > THROUGHPUT_TARGET and measured['power'] < POWER_BUDGET and measured['score'] > SCORE_THRESHOLD:
                data.append(measured)
        if ALPHA != None:
            data = sorted(data, key=lambda d: d['throughput'] if ALPHA == 0 else (d['power'] if ALPHA == 1 else d['score']))
        if data:
            return data[0]
        else:
            return data
    
    def special_freq_adjustment(self):
        self.cpu_freq1_index = len(CPU_DENVER_0)-1
        self.cpu_freq2_index = len(CPU_DENVER_1)-1
        self.cpu_freq3_index = len(CPU_DENVER_2)-1
        self.cpu_core_index = len(CPU_ONLINE)-1
        self.gpu_freq_index = len(GPU_FREQ)-1
        self.mem_freq_index = len(EMC_FREQ)-1

    def set_client_settings(self, action):
        actions_list = list(Action)[action.value].name.split('_')
        
        if "INC_CPU_CORE" in actions_list:
            self.cpu_core_index = min(self.cpu_core_index + 1, len(CPU_ONLINE) - 1)
        if "DEC_CPU_CORE" in actions_list:
            self.cpu_core_index = max(0, self.cpu_core_index - 1)
        if "INC_CPU_FREQ1" in actions_list:
            self.cpu_freq1_index = min(self.cpu_freq1_index + 1, len(CPU_DENVER_0) - 1)
        if "DEC_CPU_FREQ1" in actions_list:
            self.cpu_freq1_index = max(0, self.cpu_freq1_index - 1)
        if "INC_CPU_FREQ2" in actions_list:
            self.cpu_freq2_index = min(self.cpu_freq2_index + 1, len(CPU_DENVER_1) - 1)
        if "DEC_CPU_FREQ2" in actions_list:
            self.cpu_freq2_index = max(0, self.cpu_freq2_index - 1)
        if "INC_CPU_FREQ3" in actions_list:
            self.cpu_freq3_index = min(self.cpu_freq3_index + 1, len(CPU_DENVER_2) - 1)
        if "DEC_CPU_FREQ3" in actions_list:
            self.cpu_freq3_index = max(0, self.cpu_freq3_index - 1)
        if "INC_GPU_FREQ" in actions_list:
            self.gpu_freq_index = min(self.gpu_freq_index + 1, len(GPU_FREQ) - 1)
        if "DEC_GPU_FREQ" in actions_list:
            self.gpu_freq_index = max(0, self.gpu_freq_index - 1)
        if "INC_MEM_FREQ" in actions_list:
            self.mem_freq_index = min(self.mem_freq_index + 1, len(EMC_FREQ) - 1)
        if "DEC_MEM_FREQ" in actions_list:
            self.mem_freq_index = max(0, self.mem_freq_index - 1)
        if action == None:
            self.special_freq_adjustment()
        self.set_settings(self.get_sets())
    
    def run_inference_and_collect_data(self):
        new_state = str(self.get_current_state())
        if new_state in self.measured_data.keys() and not self.evaluation:
            return self.measured_data[new_state]
        prev_t = time()
        measured = self.client.run_inference()
        end_t = time() - prev_t
        logging.info(f"OD Inference Time: {round(end_t, 2)} sec")
        if measured:
            if self.calculate_reward(measured) < 0:
                self.num_of_violations_training[self.get_current_cl_index()] += 1
            if not new_state in self.oracle_data[self.get_current_cl_index()].keys() \
                or self.oracle_data[self.get_current_cl_index()][str(self.get_sets())]['throughput'] < measured['throughput']:
                self.oracle_data[self.get_current_cl_index()][str(self.get_sets())] = measured
            self.measured_data[new_state] = measured
            return measured
    
    def calculate_reward(self, measured_metrics):
        guard = 0 if self.evaluation else RUNTIME_GUARD
        power, throughput, score = measured_metrics['power'], measured_metrics['throughput'], measured_metrics['score']
        if (ALPHA == 0 and throughput < THROUGHPUT_TARGET) or (ALPHA == 1 and power > (POWER_BUDGET)) or (ALPHA == 2 and score < SCORE_THRESHOLD):
            reward = -1
        elif (ALPHA == 0 and throughput > THROUGHPUT_TARGET) or (ALPHA == 1 and power < (POWER_BUDGET)) or (ALPHA == 2 and score > SCORE_THRESHOLD):
            reward = (1 - ALPHA) * (throughput / THROUGHPUT_TARGET) + ALPHA * (POWER_BUDGET / power) if ALPHA != 2 else (score / SCORE_THRESHOLD)
        elif ALPHA == None and not (power * (1 + guard) > (POWER_BUDGET) or throughput * (1 + guard) < THROUGHPUT_TARGET or score * (1 + guard) < SCORE_THRESHOLD):
            reward = 1/3 * (throughput / THROUGHPUT_TARGET) + 1/3 * (POWER_BUDGET / power) + 1/3 * (score / SCORE_THRESHOLD)
        else:
            reward = -1

        self.last_sets[self.client.cl] = [str(self.get_sets()), [self.cpu_core_index, self.cpu_freq1_index, self.cpu_freq2_index, self.cpu_freq3_index, self.gpu_freq_index, self.mem_freq_index]]
        if not self.client.cl in self.best_sets or reward > self.best_sets[self.client.cl][0]:
            self.best_sets[self.client.cl] = [reward, [self.cpu_core_index, self.cpu_freq1_index, self.cpu_freq2_index, self.cpu_freq3_index, self.gpu_freq_index, self.mem_freq_index]]
        return reward
    
    def capture_data(self, reward, measured_metrics, sets):
        self.reward_history.append({sets:reward})
        if reward < 0:
            for cl_index in range(self.get_current_cl_index(), len(self.client.CONCURRENCY)):
                self.RL.prohibited_states[cl_index][str(self.get_sets())] = True
        if self.evaluation == True:
            self.eval_data['throughput'].append(measured_metrics['throughput'])
            self.eval_data['power'].append(measured_metrics['power'])
            self.eval_data['score'].append(measured_metrics['score'])
            if reward < 0:
                self.toggled = True
                self.num_of_violations_evaluations += 1
            if self.toggled == True and reward > 0:
                self.toggled = False

    def train(self):
        self.start_time = time()
        logging.info("agent started learning optimization")

        while self.round_num < configs.TOTAL_EPSIODES:
            for cl in range(len(self.client.CONCURRENCY)):
                self.cl_index = cl
                self.client.set_settings(self.get_sets())
                logging.debug(f"round number: {self.round_num}, epsilon: {self.epsilon}")
                self.round_num += 1
                
                self.current_state = self.get_current_state()
                
                choosen_action = self.RL.get_action(self.current_state, self.epsilon[self.cl_index], self.get_current_state_index())
                logging.debug(f"new choosen action: {str(choosen_action)}")
                
                self.set_client_settings(choosen_action)
                if str(self.get_current_state()) in self.measured_data.keys():
                    logging.info("Trained")
                    break
                logging.debug(f"new sets setting: {self.get_sets()}")

                measured_metrics = self.run_inference_and_collect_data()
                if choosen_action == None or measured_metrics == None:
                    continue
                logging.debug(f"throughput: {measured_metrics['throughput']} - power: {measured_metrics['power']} - score: {measured_metrics['score']}")
                
                reward = self.calculate_reward(measured_metrics)
                logging.debug(f"reward: {reward}")
                
                new_state = np.array([self.current_state[0], self.get_sets()])
                new_state_value = self.RL.update_qlearning(self.current_state, choosen_action, new_state, reward)
                logging.debug(f"new state value: {new_state_value}")
                self.current_state = new_state
                
                self.capture_data(reward, measured_metrics, str(self.get_sets()))

                self.epsilon[self.cl_index] = max(self.epsilon[self.cl_index] * EPISLON, MIN_EPSILON)
                
        self.stop_time = time()
        elapsed = round(self.stop_time - self.start_time, 2)
        logging.info(f"agent finished measurement and learning optimization after {self.round_num} steps, {elapsed}s")
        # q_table_visualizer = QTableVisualizer(self.RL.table)
        # q_table_visualizer.visualize()
        return elapsed
    
def main():
    global ALPHA, POWER_BUDGET, THROUGHPUT_TARGET, SCORE_THRESHOLD

    prev_train = time()
    result = []
    oracle_res = []
    rewards = []
    for model in MODELS:
        for const in CONSTRAINTS:
            for alpha in ALPHAS:
                configs.MODEL_NAME = model["name"]
                POWER_BUDGET = SPEC_ORDER[const]["pb"]
                THROUGHPUT_TARGET = SPEC_ORDER[const]["throughputt"]
                SCORE_THRESHOLD = SPEC_ORDER[const]["scth"]
                ALPHA = alpha
                
                print(model["name"], ALPHA, const, POWER_BUDGET, THROUGHPUT_TARGET, SCORE_THRESHOLD)
               
                board_client = Jetson(True)

                if SCORE_THRESHOLD > board_client.client.file_score[configs.MODEL_NAME]:
                    print(f"Score does not meet Score Threshold")
                    continue

                agent = Agent(board_client)
                elapsed = agent.train()

                qos = "power" if ALPHA==1 else (("throughput_target" if ALPHA==0 else "score_threshold") if ALPHA else "default")
                rewards.append({"data": agent.reward_history, "xlabel": f"{configs.MODEL_NAME.title()} QoS: {const}-{qos}", "Time Elapsed": elapsed})    
                
                for cl in self.client.CONCURRENCY:
                    agent.round_num = 0
                    agent.evaluation = True
                    agent.epsilon = [MIN_EPSILON for _ in range(len(self.client.CONCURRENCY))]
                    agent.num_of_violations_evaluations = 0
                    agent.client.cl = cl
                    configs.TOTAL_EPSIODES = 5
                    if cl in agent.best_sets.keys():
                        agent.cl_index = agent.best_sets[cl][1]
                        agent.train()

                for clid, cl in enumerate(self.client.CONCURRENCY):
                    if cl in agent.last_sets.keys():
                        if agent.round_num == configs.TOTAL_EPSIODES and not (agent.RL.prohibited_states[clid][agent.last_sets[cl][0]] if agent.last_sets[cl][0] in agent.RL.prohibited_states[clid].keys() else False):
                            row_low = agent.client.client.run_inference(agent.last_sets[cl][0], cl)
                            result.append({"model": model["name"], "cl": cl, "power_budget": POWER_BUDGET, "throughput_target": THROUGHPUT_TARGET, "score_threshold": SCORE_THRESHOLD, "alpha": ALPHA, "constraint_level": const,
                                "settings": str(agent.last_sets[cl][0]), "throughput": row_low["throughput"], "power": row_low["power"], "score" :row_low["score"], "violations": agent.num_of_violations_evaluations})
                            oracle_data = agent.get_best_oracle_data(cl)
                            if oracle_data:
                                oracle_res.append({"model": model["name"], "cl": cl, "power_budget": POWER_BUDGET, "throughput_target": THROUGHPUT_TARGET, "score_threshold": SCORE_THRESHOLD, "alpha": ALPHA, "constraint_level": const,
                                "setting": oracle_data['settings'], "throughput": oracle_data["throughput"], "power": oracle_data["power"], "score" :oracle_data["score"],})

                            with open('results/result_qlearning.csv', 'w', newline='') as output_file:
                                dict_writer = csv.DictWriter(output_file, result[0].keys())
                                if output_file.tell() == 0:  # Check if the file is empty
                                    dict_writer.writeheader()  # Write the header only if the file is empty
                                dict_writer.writerows(result)

                            if oracle_res:
                                with open('results/oracle_qlearning.csv', 'w', newline='') as output_file:
                                    dict_writer = csv.DictWriter(output_file, oracle_res[0].keys())
                                    if output_file.tell() == 0:  # Check if the file is empty
                                        dict_writer.writeheader()  # Write the header only if the file is empty
                                    dict_writer.writerows(oracle_res)
    
    end_train = time() - prev_train
    print("Total Time Elapsed for Training RL", round(end_train, 2), "sec")

    df_reward = pd.DataFrame(rewards)
    df_reward.to_csv("results/rewards_qlearning.csv", index=False)

if __name__ == "__main__":
    main()
