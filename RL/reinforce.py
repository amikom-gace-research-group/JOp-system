import torch
import torch.nn as nn
from enum import Enum
import numpy as np
import torch.optim as optim
from configs import PMODE_ID, MU

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

class Action(Enum):
    INC_CL, DEC_CL, DO_NOTHING = range(3)

class REINFORCE:
    def __init__(self, state_dim, action_dim):
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=MU)
        self.init_prohibited_states()
    
    def init_prohibited_states(self):
        self.prohibited_states = []
        for _ in range(len(PMODE_ID)):
            self.prohibited_states.append({})
    
    def get_action(self, state, epsilon):
        if not (self.prohibited_states[state[0]][str(state[1:])] if str(state[1:]) in self.prohibited_states[state[0]].keys() else False):
            state = torch.FloatTensor(state)
            action_probs = self.policy_network(state)
            if np.random.rand() < epsilon:
                action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())
            else:
                action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[action])
            return Action(action), log_prob

    def update(self, log_probs, rewards):
        policy_loss = 0.0
        for log_prob, reward in zip(log_probs, rewards):
            policy_loss += -log_prob * reward
        policy_loss = torch.tensor(policy_loss, requires_grad=True)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

