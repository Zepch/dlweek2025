# hybrid_strategy.py
import numpy as np
import pandas as pd
from models.lstm import build_lstm_model
from models.transformer import build_transformer_model
from models.reinforcement import DQNAgent

class HybridTradingStrategy:
    def __init__(self, config):
        self.config = config
        self.lstm_model = None
        self.transformer_model = None
        self.rl_agent = None
        self.position = 0  # 0: no position, 1: long, -1: short
        
    def initialize_models(self, input_shape):
        # Initialize LSTM model
        self.lstm_model = build_lstm_model(input_shape)
        
        # Initialize Transformer model
        self.transformer_model = build_transformer_model(input_shape)
        
        # Initialize RL agent with state_size corresponding to features + predictions
        state_size = input_shape[1] + 2  # +2 for LSTM and Transformer predictions
        action_size = 3  # buy, hold, sell
        self.rl_agent = DQNAgent(state_size, action_size)
        
    def train_models(self, X_train, y_train, epochs=50, batch_size=32):
        # Train LSTM
        self.lstm_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        # Train Transformer
        self.transformer_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        # Train RL agent through environment interaction (done in backtest)
        
    def predict(self, X):
        """
        Generate predictions from both models
        """
        lstm_pred = self.lstm_model.predict(X)
        transformer_pred = self.transformer_model.predict(X)
        
        # Combine predictions (simple average for now)
        # This can be enhanced with adaptive weighting based on recent performance
        ensemble_pred = (lstm_pred + transformer_pred) / 2
        
        return ensemble_pred, lstm_pred, transformer_pred
        
    def get_action(self, state, current_price, next_pred):
        """
        Determine trading action based on ML predictions and RL policy
        
        Returns:
        action: 0 (sell), 1 (hold), 2 (buy)
        """
        # Construct state for RL agent
        rl_state = np.append(state.flatten(), [next_pred[0][0], next_pred[1][0]])
        rl_state = rl_state.reshape(1, -1)
        
        # Get action from RL agent
        action = self.rl_agent.act(rl_state)
        
        return action
        
    def execute_trade(self, action, price, capital, shares_held):
        """
        Execute trading action
        """
        if action == 0:  # Sell
            if shares_held > 0:  # If we have shares, sell them
                capital += shares_held * price
                shares_held = 0
                self.position = 0
                
        elif action == 2:  # Buy
            if shares_held == 0:  # If we don't have shares, buy them
                shares_to_buy = capital // price
                capital -= shares_to_buy * price
                shares_held = shares_to_buy
                self.position = 1
                
        # action == 1 is hold, do nothing
        
        return capital, shares_held