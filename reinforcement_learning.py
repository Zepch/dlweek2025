# reinforcement_learning.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        # Filter out entries with None for next_state
        valid_batch = [i for i in minibatch if i[3] is not None]
        
        if len(valid_batch) == 0:
            return  # Skip this batch if no valid transitions
        
        states = torch.FloatTensor([i[0] for i in valid_batch])
        actions = torch.LongTensor([i[1] for i in valid_batch])
        rewards = torch.FloatTensor([i[2] for i in valid_batch])
        next_states = torch.FloatTensor([i[3] for i in valid_batch])
        dones = torch.FloatTensor([i[4] for i in valid_batch])
        
        # Q(s_t, a)
        state_action_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # V(s_{t+1}) for all next states
        next_state_values = self.target_model(next_states).max(1)[0].detach()
        
        # Expected Q values
        expected_state_action_values = (next_state_values * self.gamma) * (1 - dones) + rewards
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay