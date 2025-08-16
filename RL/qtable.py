import numpy as np
import itertools
from enum import Enum
from configs import MU, GAMMA, CPU_DENVER_0, CPU_DENVER_1, CPU_DENVER_2, GPU_FREQ, EMC_FREQ, CPU_ONLINE

from board import IBoard

# Define all possible individual actions
cpu_core_actions = ["INC_CPU_CORE", "DEC_CPU_CORE", "DO_NOTHING_CPU_CORE"]
cpu_freq1_actions = ["INC_CPU_FREQ1", "DEC_CPU_FREQ1", "DO_NOTHING_CPU_FREQ1"]
cpu_freq2_actions = ["INC_CPU_FREQ2", "DEC_CPU_FREQ2", "DO_NOTHING_CPU_FREQ2"]
cpu_freq3_actions = ["INC_CPU_FREQ3", "DEC_CPU_FREQ3", "DO_NOTHING_CPU_FREQ3"]
gpu_freq_actions = ["INC_GPU_FREQ", "DEC_GPU_FREQ", "DO_NOTHING_GPU_FREQ"]
mem_freq_actions = ["INC_MEM_FREQ", "DEC_MEM_FREQ", "DO_NOTHING_MEM_FREQ"]

# Generate all combinations of these actions
all_combinations = list(itertools.product(cpu_core_actions, cpu_freq1_actions, cpu_freq2_actions,
                                          cpu_freq3_actions, gpu_freq_actions, mem_freq_actions))

# Remove duplicates (if any)
unique_combinations = list(set(all_combinations))
print("Unique Combinations", len(unique_combinations))

action_dict = {"_".join(combo): i for i, combo in enumerate(unique_combinations)}

# Dynamically create the Enum class with the given actions
Action = Enum('Action', action_dict)

class QTable():
    def __init__(self, client: IBoard, states, actions):
        self.client = client
        self.mu = MU
        self.gamma = GAMMA
        self.states = states
        self.actions = actions
        self.init_table()
        self.init_prohibited_states()
    
    def init_table(self):
        self.table = {}
        for state in self.states:
            self.table[str(state)] = np.zeros(len(self.actions))
    
    def init_prohibited_states(self):
        self.prohibited_states = []
        for _ in range(len(self.client.CONCURRENCY)):
            self.prohibited_states.append({})
    
    def get_next_state(self, current_state_index, action):
        next_state_index = current_state_index[1:]
        err = 0
        
        # Map each component of the action to the appropriate state update
        actions_list = list(Action)[action.value].name.split('_')
        
        if "INC_CPU_CORE" in actions_list:
            next_state_index[0] = min(next_state_index[0] + 1, len(CPU_ONLINE) - 1)
        if "DEC_CPU_CORE" in actions_list:
            next_state_index[0] = max(0, next_state_index[0] - 1)
        if "INC_CPU_FREQ1" in actions_list:
            next_state_index[1] = min(next_state_index[1] + 1, len(CPU_DENVER_0) - 1)
        if "DEC_CPU_FREQ1" in actions_list:
            next_state_index[1] = max(0, next_state_index[1] - 1)
        if "INC_CPU_FREQ2" in actions_list:
            next_state_index[2] = min(next_state_index[2] + 1, len(CPU_DENVER_1) - 1)
        if "DEC_CPU_FREQ2" in actions_list:
            next_state_index[2] = max(0, next_state_index[2] - 1)
        if "INC_CPU_FREQ3" in actions_list:
            next_state_index[3] = min(next_state_index[3] + 1, len(CPU_DENVER_2) - 1)
        if "DEC_CPU_FREQ3" in actions_list:
            next_state_index[3] = max(0, next_state_index[3] - 1)
        if "INC_GPU_FREQ" in actions_list:
            next_state_index[4] = min(next_state_index[4] + 1, len(GPU_FREQ) - 1)
        if "DEC_GPU_FREQ" in actions_list:
            next_state_index[4] = max(0, next_state_index[4] - 1)
        if "INC_MEM_FREQ" in actions_list:
            next_state_index[5] = min(next_state_index[5] + 1, len(EMC_FREQ) - 1)
        if "DEC_MEM_FREQ" in actions_list:
            next_state_index[5] = max(0, next_state_index[5] - 1)

        # Check for invalid actions that don't change the state
        if next_state_index == current_state_index:
            err = 1
        
        return next_state_index, err
    
    def available_actions(self, current_state_index: np.array):
        available_actions = []
        for action_index, action in enumerate(Action):
            _, err = self.get_next_state(current_state_index, action)
            if err == 1:
                continue
            available_actions.append(action_index)
        return available_actions
    
    def get_action(self, current_state: np.array, epsilon: float, current_state_index):
        if not (self.prohibited_states[current_state[0]][str(current_state[1:])] if str(current_state[1:]) in self.prohibited_states[current_state[0]].keys() else False):
            rand_num = np.random.rand()
            if rand_num < epsilon:
                available_actions = self.available_actions(current_state_index)
                if len(available_actions) == 0:
                    return None
                action = np.random.choice(available_actions)
            else:
                action = self.get_largest_q_action(str(current_state))
            return Action(action)
    
    def get_largest_q_action(self, state):
        return np.argmax(self.table[state])
    
    def update_sarsa(self, state, action, new_state, new_action, reward):
        action = action.value
        new_action = new_action.value
        state, new_state = str(state), str(new_state)
        
        qsa = self.table[state][action]
        qnewsa = self.table[new_state][new_action]
        
        # SARSA update rule (on-policy)
        self.table[state][action] = qsa + self.mu * (reward + self.gamma * qnewsa - qsa)
        
        return self.table[state][action]

    
    def update_qlearning(self, state, action, new_state, reward):
        action = action.value
        state, new_state = str(state), str(new_state)
        
        new_state_action = self.get_largest_q_action(new_state)
        qsa = self.table[state][action]

        qnewsa = self.table[new_state][new_state_action]

        # Q-learning update rule (off-policy)
        self.table[state][action] = (1 - self.mu) * qsa + self.mu * (reward + self.gamma * qnewsa)
        return self.table[state][action]
    
class QTableVisualizer:
    def __init__(self, q_table):
        self.q_table = q_table

    def visualize(self):
        for state, actions in self.q_table.items():
            print(f"State: {state}")
            print("Actions:")
            for action, q_value in enumerate(actions):
                print(f"  Action {Action(action).name}: Q-value = {q_value}")
            print()
