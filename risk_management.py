# risk_management.py
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
