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

class TradingEnvironment:
    def __init__(self, df, initial_balance=10000, transaction_fee=0.001):
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long position
        self.portfolio_value = [self.initial_balance]
        
        return self._get_state()
    
    def _get_state(self):
        # Return the current state (features + position + balance)
        if self.current_step >= len(self.df):
            return None
            
        # Get current row of features
        features = self.df.iloc[self.current_step].values
        
        # Add position and balance to state
        position_balance = np.array([self.position, self.balance / self.initial_balance])
        state = np.concatenate([features, position_balance])
        
        return state
        
    def step(self, action):
        # 0 = hold, 1 = buy, 2 = sell
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Calculate reward based on action
        reward = 0
        
        if action == 1 and self.position == 0:  # Buy
            shares = self.balance / current_price
            cost = shares * current_price * (1 + self.transaction_fee)
            if cost <= self.balance:
                self.position = 1
                self.balance -= cost
                reward = 0  # Neutral reward for opening position
            else:
                reward = -1  # Penalty for invalid action
                
        elif action == 2 and self.position == 1:  # Sell
            next_price = self.df.iloc[min(self.current_step + 1, len(self.df) - 1)]['Close']
            price_change = (next_price - current_price) / current_price
            reward = -price_change  # Negative reward for selling before price increase
            self.position = 0
            self.balance += current_price * (1 - self.transaction_fee)
            
        elif self.position == 1:  # Hold with position
            next_price = self.df.iloc[min(self.current_step + 1, len(self.df) - 1)]['Close']
            price_change = (next_price - current_price) / current_price
            reward = price_change
        
        # Update step and check if done
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Calculate portfolio value
        portfolio_value = self.balance
        if self.position == 1:
            portfolio_value += current_price
        self.portfolio_value.append(portfolio_value)
        
        # Return next state, reward, done flag, and info
        next_state = self._get_state() if not done else None
        info = {'portfolio_value': portfolio_value}
        
        return next_state, reward, done, info