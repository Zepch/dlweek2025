# enhanced_rl_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class DuelQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelQNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        advantage = self.advantage_stream(features)
        value = self.value_stream(features)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        return value + advantage - advantage.mean(1, keepdim=True)

class EnhancedDQNAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, gamma=0.95, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        
        # Model, target model and optimizer
        self.model = DuelQNetwork(state_size, action_size)
        self.target_model = DuelQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.update_target_model()  # Copy weights to target model
        
        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Other parameters
        self.batch_size = 64
        self.update_frequency = 4
        self.step_counter = 0
        self.n_step_buffer = deque(maxlen=3)  # N-step buffer for N-step bootstrapping
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        # Store n-step transition
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Process n-step returns
        if len(self.n_step_buffer) == 3:
            state, action, _, _, _ = self.n_step_buffer[0]
            
            # Calculate n-step reward
            n_step_reward = 0
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:
                    break
            
            # Get the furthest next state and done flag
            _, _, _, next_state, done = self.n_step_buffer[-1]
            
            # Store the processed transition
            self.memory.append((state, action, n_step_reward, next_state, done))
            
    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
            
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state)
            return torch.argmax(act_values[0]).item()
            
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        # Sample a minibatch
        minibatch = random.sample(self.memory, batch_size)
        
        # Filter out None next_states
        valid_batch = [i for i in minibatch if i[3] is not None]
        if len(valid_batch) == 0:
            return
        
        # Convert to tensors
        states = torch.FloatTensor([i[0] for i in valid_batch])
        actions = torch.LongTensor([i[1] for i in valid_batch])
        rewards = torch.FloatTensor([i[2] for i in valid_batch])
        next_states = torch.FloatTensor([i[3] for i in valid_batch])
        dones = torch.FloatTensor([i[4] for i in valid_batch])
        
        # Double DQN update
        # Select actions from main model
        next_q_values = self.model(next_states)
        max_actions = next_q_values.max(1)[1].unsqueeze(1)
        
        # Evaluate those actions with target network
        next_q_values_target = self.target_model(next_states)
        next_q_values = next_q_values_target.gather(1, max_actions).squeeze(1)
        
        # Expected Q values
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Get current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)  # Gradient clipping
        self.optimizer.step()
        
        # Update target network periodically
        self.step_counter += 1
        if self.step_counter % self.update_frequency == 0:
            self.update_target_model()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
