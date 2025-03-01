# enhanced_rl.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
import random
import matplotlib.pyplot as plt

class PrioritizedReplayBuffer:
    """A prioritized replay buffer for more efficient learning"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # determines how much prioritization is used
        self.beta = beta    # importance sampling weight
        self.beta_increment = beta_increment
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        # Add with max priority on first entry
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        if self.size < batch_size:
            return [], [], [], []
            
        # Calculate sampling probabilities
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        states = torch.FloatTensor([s[0] for s in samples if s[3] is not None])
        actions = torch.LongTensor([s[1] for s in samples if s[3] is not None])
        rewards = torch.FloatTensor([s[2] for s in samples if s[3] is not None])
        next_states = torch.FloatTensor([s[3] for s in samples if s[3] is not None])
        dones = torch.FloatTensor([s[4] for s in samples if s[3] is not None])
        
        return states, actions, rewards, next_states, dones, indices, torch.FloatTensor(weights)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # small constant to avoid zero priority

def enhance_rl_agent():
    print("Adding advanced RL features to your trading system...")
    print("1. Creating enhanced DuelDQNAgent class")
    
    with open("enhanced_rl_agent.py", "w", encoding="utf-8") as f:
        f.write("""# enhanced_rl_agent.py
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
""")
    
    print("2. Creating adaptive risk management module")
    with open("risk_management.py", "w", encoding="utf-8") as f:
        f.write("""# risk_management.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AdaptiveRiskManager:
    def __init__(self, initial_capital, max_drawdown_threshold=0.2, volatility_lookback=20):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_capital = initial_capital
        self.max_drawdown_threshold = max_drawdown_threshold
        self.volatility_lookback = volatility_lookback
        self.position_sizes = []
        self.drawdowns = []
        
    def kelly_position_size(self, win_rate, win_loss_ratio, volatility, balance):
       
        if win_rate <= 0 or win_loss_ratio <= 0:
            return 0.01  # Default to small position
            
        # Standard Kelly formula
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Adjust for volatility (lower size when volatility is high)
        kelly_pct = kelly_pct * (1 / (1 + volatility))
        
        # Cap the position size
        kelly_pct = min(kelly_pct, 0.25)  # Maximum 25% of capital
        
        # If drawdown is greater than threshold, reduce position size
        current_drawdown = 1 - (balance / self.max_capital)
        if current_drawdown > self.max_drawdown_threshold:
            kelly_pct *= (1 - (current_drawdown / (self.max_drawdown_threshold * 2)))
            kelly_pct = max(kelly_pct, 0.01)  # Minimum position size
        
        return kelly_pct
        
    def update_capital(self, new_capital):
        self.current_capital = new_capital
        if new_capital > self.max_capital:
            self.max_capital = new_capital
        
        # Calculate and track drawdown
        drawdown = 1 - (new_capital / self.max_capital)
        self.drawdowns.append(drawdown)
        
    def calculate_volatility(self, returns, window=None):
        if window is None:
            window = self.volatility_lookback
            
        if len(returns) < window:
            return 0.01  # Default low volatility if not enough data
            
        # Use exponentially weighted standard deviation for more weight on recent data
        return returns[-window:].ewm(span=window//2).std().iloc[-1]
        
    def suggest_position_size(self, win_rate, avg_win, avg_loss, returns):
        win_loss_ratio = avg_win / max(avg_loss, 0.001)  # Avoid division by zero
        volatility = self.calculate_volatility(pd.Series(returns))
        
        position_size = self.kelly_position_size(
            win_rate, 
            win_loss_ratio,
            volatility,
            self.current_capital
        )
        
        self.position_sizes.append(position_size)
        return position_size
        
    def get_stop_loss(self, entry_price, position_type, volatility=None):
        
        if volatility is None:
            volatility = 0.02  # Default volatility
            
        # ATR-inspired stop loss - wider stops in volatile markets
        atr_multiple = 2.0
        stop_distance = entry_price * volatility * atr_multiple
        
        if position_type == 'long':
            return entry_price - stop_distance
        else:  # short position
            return entry_price + stop_distance
""")
    
    print("3. Creating regime detection module for adaptive strategies")
    with open("regime_detection.py", "w", encoding="utf-8") as f:
        f.write("""# regime_detection.py
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class MarketRegimeDetector:
    def __init__(self, n_regimes=2, lookback=120):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.model = None
        self.scaler = StandardScaler()
        
    def extract_features(self, df):
        features = pd.DataFrame(index=df.index)
        
        # Volatility features
        features['volatility'] = df['Close'].pct_change().rolling(window=20).std()
        
        # Trend features
        features['trend'] = df['Close'].pct_change(20)
        
        # Volume features
        if 'Volume' in df.columns:
            features['vol_change'] = df['Volume'].pct_change()
        
        # Correlation features
        if 'High' in df.columns and 'Low' in df.columns:
            features['high_low_ratio'] = df['High'] / df['Low']
        
        # Mean reversion features
        if 'MA_20' in df.columns and 'Close' in df.columns:
            features['ma_distance'] = (df['Close'] - df['MA_20']) / df['MA_20']
        
        return features.dropna()
        
    def fit(self, df):
        features = self.extract_features(df)
        if len(features) < self.lookback:
            print("Warning: Not enough data for regime detection")
            return None
        
        # Standardize features
        X = self.scaler.fit_transform(features.values)
        
        # Fit Gaussian Mixture Model
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42
        )
        self.model.fit(X)
        
        # Get regime labels
        regimes = self.model.predict(X)
        
        # Add regime labels to dataframe
        df_with_regimes = df.copy()
        df_with_regimes['market_regime'] = pd.Series(regimes, index=features.index)
        
        return df_with_regimes
        
    def predict_regime(self, df):
        if self.model is None:
            return None
            
        features = self.extract_features(df)
        if len(features) == 0:
            return None
            
        # Use most recent data point
        latest_features = features.iloc[-1:].values
        X = self.scaler.transform(latest_features)
        
        # Get regime and probability
        regime = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        
        return {
            'regime': regime,
            'confidence': probs[regime],
            'all_probs': probs
        }
""")
    
    print("✅ Enhanced RL modules created successfully!")
    print("You can now import these modules to improve your trading system's performance.")

if __name__ == "__main__":
    enhance_rl_agent()