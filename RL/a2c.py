import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from enum import Enum
import torch.optim as optim
from configs import PMODE_ID, MU, GAMMA

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        actor_output = self.actor(state)
        critic_output = self.critic(state)
        return actor_output, critic_output

class Action(Enum):
    INC_CL, DEC_CL, DO_NOTHING = range(3)

class A2C:
    def __init__(self, state_dim, action_dim):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=MU)
        self.init_prohibited_states()
    
    def init_prohibited_states(self):
        self.prohibited_states = [{} for _ in range(len(PMODE_ID))]

    def get_action(self, state, epsilon):
        if not (self.prohibited_states[state[0]][str(state[1:])] if str(state[1:]) in self.prohibited_states[state[0]].keys() else False):
            state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            action_probs, _ = self.actor_critic(state)
            if np.random.rand() < epsilon:
                action = np.random.choice(len(action_probs.squeeze()), p=action_probs.squeeze().detach().numpy())
                return Action(action)
            else:
                distribution = torch.distributions.Categorical(action_probs)
                action = distribution.sample()
                return Action(action.item())

    def update(self, states, actions, rewards, next_states, dones):
        self.optimizer.zero_grad()

        # Convert lists to tensors
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        dones = torch.IntTensor(np.array(rewards))

        # Get current predictions
        _, values = self.actor_critic(states)
        _, next_values = self.actor_critic(next_states)
        
        # Calculate TD Target and advantages
        td_target = rewards + GAMMA * next_values * (1 - dones)
        advantages = td_target - values
        
        # Calculate log probabilities
        action_probs, _ = self.actor_critic(states)
        log_probs = torch.log(action_probs.gather(1, actions.view(-1, 1)))
        
        # Calculate losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = (advantages ** 2).mean()
        
        # Total loss
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()
