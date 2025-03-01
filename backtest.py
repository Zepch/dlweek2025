# backtest.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SimpleBacktester:
    def __init__(self, data, initial_capital=10000):
        self.data = data
        print(self.data)
        self.initial_capital = initial_capital
        self.position = 0
        self.capital = initial_capital
        self.returns = []
        self.equity = []
        self.trades = []
    
    def run(self, signal_column):
        """
        Run backtest based on signal column
        signal_column values: 1 (buy), -1 (sell), 0 (hold)
        """
        self.capital = self.initial_capital
        self.position = 0
        self.returns = []
        self.equity = [self.initial_capital]
        self.trades = []
        
        for i in range(1, len(self.data)):
            signal = self.data[signal_column].iloc[i]
            current_price = self.data['Close'].iloc[i]
            prev_price = self.data['Close'].iloc[i-1]
            daily_return = (current_price - prev_price) / prev_price
            
            # Update position based on signal
            if signal == 1 and self.position == 0:  # Buy signal
                self.position = 1
                self.trades.append(('BUY', self.data.index[i], current_price))
            elif signal == -1 and self.position == 1:  # Sell signal
                self.position = 0
                self.trades.append(('SELL', self.data.index[i], current_price))
            
            # Update portfolio value
            if self.position == 1:
                self.capital *= (1 + daily_return)
                
            self.equity.append(self.capital)
            self.returns.append(daily_return if self.position == 1 else 0)
        
        # Calculate performance metrics
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        self.total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        self.returns_series = pd.Series(self.returns)
        
        # Annualized return
        self.annual_return = (1 + self.total_return/100) ** (252/len(self.returns)) - 1
        
        # Sharpe ratio
        risk_free_rate = 0.02  # Assuming 2% annual risk-free rate
        daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = [r - daily_risk_free for r in self.returns]
        if np.std(excess_returns) != 0:
            self.sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        else:
            self.sharpe_ratio = 0
        
        # Maximum drawdown
        equity_series = pd.Series(self.equity)
        drawdown = equity_series / equity_series.cummax() - 1
        self.max_drawdown = drawdown.min() * 100
    
    def plot_results(self):
        """Plot equity curve and drawdown"""
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(self.data.index[1:], self.equity[1:])
        plt.title('Equity Curve')
        plt.grid(True)
        
        # Plot drawdown
        equity_series = pd.Series(self.equity[1:], index=self.data.index[1:])
        drawdown = equity_series / equity_series.cummax() - 1
        
        plt.subplot(2, 1, 2)
        plt.fill_between(self.data.index[1:], drawdown, 0, color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def print_summary(self):
        """Print performance summary"""
        print(f"Total Return: {self.total_return:.2f}%")
        print(f"Annualized Return: {self.annual_return*100:.2f}%")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {self.max_drawdown:.2f}%")
        print(f"Total Trades: {len(self.trades)}")