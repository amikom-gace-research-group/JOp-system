import logging, sys, csv
from time import time
from a2c import Action, A2C
import numpy as np
import pandas as pd
from board import IBoard, Jetson
import configs
from configs import EPISLON, MIN_EPSILON, RUNTIME_GUARD, PMODE_ID,\
                    LOGGING_LEVEL, MODELS, ALPHAS, CONSTRAINTS, SPEC_ORDER
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S %p', stream=sys.stdout, level=LOGGING_LEVEL)

ALPHA, POWER_BUDGET, THROUGHPUT_TARGET, SCORE_THRESHOLD = 0, 0, 0, 0

print("Using A2C Algorithm")

class Agent():
    def __init__(self, board: IBoard):
        self.round_num = 0
        self.epsilon = [EPISLON for _ in range(len(PMODE_ID))]
        self.client = board
        pmode_list = [i for i in range(len(PMODE_ID))]
        self.states = np.array(np.meshgrid(pmode_list, board.CONCURRENCY, indexing='ij')).T.reshape(-1,2)
        self.RL = A2C(len(self.states[0]), len(Action))
        self.cl_index = 0
        self.current_state = np.array([])
        self.measured_data = {}
        self.last_cl = {}
        self.best_cl = {}
        self.oracle_data = np.array([{} for i in range(len(PMODE_ID))])
        self.num_of_violations_training = np.array([0 for i in range(len(PMODE_ID))])
        self.num_of_violations_evaluations = 0
        self.client.power_mode = list(PMODE_ID.keys())[-1]
        self.evaluation = False
        self.toggled = False
        self.eval_data = {'throughput': [], 'power': [], 'score':[]}
        self.reward_history = []

    def clean(self):
        self.round_num = 0
        self.measured_data = {}
    
    def get_current_state(self):
        self.pmode_index = self.get_current_pmode_index()
        current_state = np.array([self.pmode_index, self.get_cl_setting()])
        return current_state

    def get_current_state_index(self):
        pmode_index = self.get_current_pmode_index()
        return [pmode_index, self.cl_index]
    
    def get_cl_setting(self):
        cl = self.client.CONCURRENCY[self.cl_index]
        return np.array(cl)
    
    def get_current_pmode_index(self):
        pmode = self.client.get_power_mode()
        return [idx for idx, key in enumerate(list(PMODE_ID.items())) if key[0] == pmode][0]
    
    def get_best_oracle_data(self, powermode):
        pmode_index = [idx for idx, key in enumerate(list(PMODE_ID.items())) if key[0] == powermode][0]
        data = []
        for pmode in self.oracle_data[pmode_index].keys():
            measured = self.oracle_data[pmode_index][pmode]
            measured['pmode'] = pmode
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
    
    def special_cl_adjustment(self):
        self.cl_index = 0
    
    def set_client_settings(self, action):
        cl_size = len(self.client.CONCURRENCY)

        if action == Action.INC_CL:
            self.cl_index = min(self.cl_index + 1, cl_size - 1)
        elif action == Action.DEC_CL:
            self.cl_index = max(0, self.cl_index - 1)
        elif action == Action.DO_NOTHING:
            pass
        elif action == None:
            self.special_cl_adjustment()
        self.client.set_cl_settings(self.get_cl_setting())
    
    def run_inference_and_collect_data(self):
        new_state = str(self.get_current_state())
        if new_state in self.measured_data.keys() and not self.evaluation:
            return self.measured_data[new_state]
        measured = self.client.run_inference()
        if measured:
            if self.calculate_reward(measured) < 0:
                self.num_of_violations_training[self.get_current_pmode_index()] += 1
            if not new_state in self.oracle_data[self.get_current_pmode_index()].keys() \
                or self.oracle_data[self.get_current_pmode_index()][str(self.get_cl_setting())]['throughput'] < measured['throughput']:
                self.oracle_data[self.get_current_pmode_index()][str(self.get_cl_setting())] = measured
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
        self.last_cl[self.client.power_mode] = [str(self.get_cl_setting()), self.cl_index]
        if not self.client.power_mode in self.best_cl or reward > self.best_cl[self.client.power_mode][0]:
            self.best_cl[self.client.power_mode] = [reward, self.cl_index, str(self.get_cl_setting())]
        return reward
    
    def set_power_mode(self):
        if not self.evaluation:
            for index, pmode in enumerate(list(PMODE_ID.keys())[1:]):
                if self.round_num > (configs.TOTAL_EPSIODES * (index+1)) // len(PMODE_ID):
                    self.client.power_mode = pmode
            if self.round_num < (configs.TOTAL_EPSIODES) // len(PMODE_ID):
                self.client.power_mode = list(PMODE_ID.keys())[0]
        if not str(self.get_current_state) in self.measured_data.keys():
            self.set_client_settings(Action.DO_NOTHING)
            self.run_inference_and_collect_data()
    
    def capture_data(self, reward, measured_metrics, cl_setting):
        self.reward_history.append({cl_setting:reward})
        if reward < 0:
            for pmode_index in range(self.get_current_pmode_index(), len(PMODE_ID)):
                self.RL.prohibited_states[pmode_index][str(self.get_cl_setting())] = True
        if self.evaluation == True:
            self.eval_data['throughput'].append(measured_metrics['throughput'])
            self.eval_data['power'].append(measured_metrics['power'])
            self.eval_data['score'].append(measured_metrics['score'])
            if reward < 0:
                self.toggled = True
                self.num_of_violations_evaluations += 1
            if self.toggled == True and reward > 0:
                self.toggled = False
    
    def check_episode_done(self):
        """
        Check if the current episode should be considered done.
        Returns:
            bool: True if the episode is done, False otherwise.
        """
        if self.current_step >= len(self.states):
            self.current_step = 0
            return 1
        else:
            self.current_step += 1
            return 0
    
    def train(self):
        self.start_time = time()
        logging.info("agent started learning optimization")

        while self.round_num < configs.TOTAL_EPSIODES:
            logging.debug(f"round number: {self.round_num}, epsilon: {self.epsilon}")
            self.round_num += 1
            states = []
            actions = []
            new_values = []
            rewards = []
            dones = []
            self.current_step = 0
            for _ in range(len(self.states)):
                self.current_state = self.get_current_state()
                self.set_power_mode()
                logging.debug(f"current state: {self.current_state}")

                choosen_action = self.RL.get_action(self.current_state, self.epsilon[self.pmode_index])
                logging.debug(f"new choosen action: {str(choosen_action)}")
                if choosen_action:
                    states.append(self.current_state)
                    actions.append(choosen_action.value)
                else:
                    self.current_step += 1
                    continue
                self.set_client_settings(choosen_action)
                logging.debug(f"new cl setting: {self.get_cl_setting()}")

                measured_metrics = self.run_inference_and_collect_data()
                if measured_metrics == None:
                    continue
                logging.debug(f"throughput: {measured_metrics['throughput']} - power: {measured_metrics['power']} - score: {measured_metrics['score']}")

                reward = self.calculate_reward(measured_metrics)
                logging.debug(f"reward: {reward}")
                rewards.append(reward)
                new_state = np.array([self.current_state[0], self.get_cl_setting()])
                new_values.append(new_state)
                self.current_state = new_state

                # Check if the episode is done (e.g., based on a condition)
                done = self.check_episode_done()
                dones.append(done)

                self.capture_data(reward, measured_metrics, str(self.get_cl_setting()))
                self.epsilon[self.pmode_index] = max(self.epsilon[self.pmode_index] * EPISLON, MIN_EPSILON)
            if len(states) == len(actions) == len(new_values) == len(rewards) == len(dones):
                self.RL.update(states, actions, rewards, new_values, dones)
            
        self.stop_time = time()
        elapsed = round(self.stop_time - self.start_time, 2)
        logging.info(f"agent finished learning optimization after {self.round_num} steps, {elapsed}s")
        return elapsed
    
def main():
    global ALPHA, POWER_BUDGET, THROUGHPUT_TARGET, SCORE_THRESHOLD

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
                
                board_client = Jetson()
                agent = Agent(board_client)
                elapsed = agent.train()

                qos = "power" if ALPHA==1 else (("throughput_target" if ALPHA==0 else "score_threshold") if ALPHA else "default")
                rewards.append({"data": agent.reward_history, "xlabel": f"{configs.MODEL_NAME.title()} QoS: {const}-{qos}", "Time Elapsed": elapsed})    
                
                for eval_pmode in PMODE_ID.keys():
                    agent.round_num = 0
                    agent.evaluation = True
                    agent.epsilon = [MIN_EPSILON for _ in range(len(PMODE_ID))]
                    agent.num_of_violations_evaluations = 0
                    agent.client.power_mode = eval_pmode
                    configs.TOTAL_EPSIODES = 500
                    if eval_pmode in agent.best_cl.keys():
                        agent.cl_index = agent.best_cl[eval_pmode][1]
                        agent.train()

                for pi, pmode in enumerate(PMODE_ID.keys()):
                    if pmode in agent.last_cl.keys():
                        if agent.round_num == configs.TOTAL_EPSIODES and not (agent.RL.prohibited_states[pi][agent.last_cl[pmode][0]] if agent.last_cl[pmode][0] in agent.RL.prohibited_states[pi].keys() else False):
                            row_low = agent.client.client.run_inference(agent.last_cl[pmode][0], pmode)
                            result.append({"model": model["name"], "pmode": pmode, "power_budget": POWER_BUDGET, "throughput_target": THROUGHPUT_TARGET, "score_threshold": SCORE_THRESHOLD, "alpha": ALPHA, "constraint_level": const,
                                "CL": str(agent.last_cl[pmode][0]), "throughput": row_low["throughput"], "power": row_low["power"], "score" :row_low["score"], "violations": agent.num_of_violations_evaluations})
                            oracle_data = agent.get_best_oracle_data(pmode)
                            if oracle_data:
                                oracle_res.append({"model": model["name"], "pmode": pmode, "power_budget": POWER_BUDGET, "throughput_target": THROUGHPUT_TARGET, "score_threshold": SCORE_THRESHOLD, "alpha": ALPHA, "constraint_level": const,
                                "CL": oracle_data['pmode'], "throughput": oracle_data["throughput"], "power": oracle_data["power"], "score" :oracle_data["score"],})

                            with open('results/result_a2c.csv', 'w', newline='') as output_file:
                                dict_writer = csv.DictWriter(output_file, result[0].keys())
                                if output_file.tell() == 0:  # Check if the file is empty
                                    dict_writer.writeheader()  # Write the header only if the file is empty
                                dict_writer.writerows(result)

                            if oracle_res:
                                with open('results/oracle_a2c.csv', 'w', newline='') as output_file:
                                    dict_writer = csv.DictWriter(output_file, oracle_res[0].keys())
                                    if output_file.tell() == 0:  # Check if the file is empty
                                        dict_writer.writeheader()  # Write the header only if the file is empty
                                    dict_writer.writerows(oracle_res)

    df_reward = pd.DataFrame(rewards)
    df_reward.to_csv("results/rewards_a2c.csv", index=False)

if __name__ == "__main__":
    main()