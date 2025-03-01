# backtest_engine.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategies.hybrid_strategy import HybridTradingStrategy
from utils.data_loader import fetch_market_data, add_technical_indicators
from utils.feature_engineering import normalize_features, create_sequences

class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.strategy = None
        self.data = None
        self.train_data = None
        self.test_data = None
        
    def load_data(self, ticker, start_date, end_date, train_ratio=0.8):
        """
        Load and prepare data for backtesting
        """
        # Fetch market data
        self.data = fetch_market_data(ticker, start_date, end_date)
        
        # Add technical indicators
        self.data = add_technical_indicators(self.data)
        
        # Split into train and test sets
        split_idx = int(len(self.data) * train_ratio)
        self.train_data = self.data.iloc[:split_idx]
        self.test_data = self.data.iloc[split_idx:]
        
        return self.data, self.train_data, self.test_data
        
    def prepare_features(self, sequence_length=60):
        """
        Prepare features for model training and backtesting
        """
        # Select features - removed BB_Upper, BB_Middle, BB_Lower
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                        'SMA20', 'SMA50', 'RSI', 'MACD', 'Signal']
        
        # Normalize features
        train_norm, scaler = normalize_features(self.train_data, feature_cols)
        test_norm = pd.DataFrame(scaler.transform(self.test_data[feature_cols]), 
                                columns=feature_cols, 
                                index=self.test_data.index)
        
        # Create sequences for LSTM/Transformer
        X_train, y_train = create_sequences(train_norm, 'Close', sequence_length)
        X_test, y_test = create_sequences(test_norm, 'Close', sequence_length)
        
        return X_train, y_train, X_test, y_test, scaler
        
    def initialize_strategy(self, X_train, y_train):
        """
        Initialize and train the strategy
        """
        self.strategy = HybridTradingStrategy(self.config)
        self.strategy.initialize_models(input_shape=X_train[0].shape)
        self.strategy.train_models(X_train, y_train, 
                                   epochs=self.config.get('epochs', 50),
                                   batch_size=self.config.get('batch_size', 32))
        
    def run_backtest(self, X_test, y_test, initial_capital=100000.0):
        """
        Run backtest on test data
        """
        capital = initial_capital
        shares_held = 0
        positions = []
        portfolio_values = []
        
        # Get actual prices for calculating portfolio value
        test_prices = self.test_data['Close'].iloc[len(self.test_data) - len(y_test):].values
        
        # Run backtest
        for i in range(len(X_test)):
            current_state = X_test[i]
            current_price = test_prices[i]
            
            # Get predictions
            ensemble_pred, lstm_pred, transformer_pred = self.strategy.predict(
                current_state.reshape(1, *current_state.shape)
            )
            
            # Determine action
            action = self.strategy.get_action(current_state, current_price, 
                                             [lstm_pred, transformer_pred])
            
            # Execute trade
            capital, shares_held = self.strategy.execute_trade(
                action, current_price, capital, shares_held
            )
            
            # Track positions and portfolio value
            positions.append(self.strategy.position)
            portfolio_value = capital + shares_held * current_price
            portfolio_values.append(portfolio_value)
            
            # Update RL agent memory
            if i < len(X_test) - 1:
                next_state = X_test[i+1]
                next_state_for_rl = np.append(
                    next_state.flatten(), 
                    [lstm_pred[0][0], transformer_pred[0][0]]
                ).reshape(1, -1)
                
                # Calculate reward based on portfolio change
                next_price = test_prices[i+1]
                next_portfolio = capital + shares_held * next_price
                reward = (next_portfolio - portfolio_value) / portfolio_value
                
                # Remember this experience
                state_for_rl = np.append(
                    current_state.flatten(), 
                    [lstm_pred[0][0], transformer_pred[0][0]]
                ).reshape(1, -1)
                
                self.strategy.rl_agent.remember(
                    state_for_rl, 
                    action, 
                    reward, 
                    next_state_for_rl, 
                    i == len(X_test) - 2  # done flag
                )
                
                # Train RL agent on batch
                if len(self.strategy.rl_agent.memory) > 32:
                    self.strategy.rl_agent.replay(32)
        
        # Calculate performance metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(portfolio_values)
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        
        results = {
            'portfolio_values': portfolio_values,
            'positions': positions,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': capital,
            'final_shares': shares_held,
            'final_portfolio_value': portfolio_values[-1]
        }
        
        return results
    
    def plot_results(self, results):
        """
        Plot backtest results
        """
        plt.figure(figsize=(14, 8))
        
        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(results['portfolio_values'])
        plt.title('Portfolio Value Over Time')
        plt.ylabel('Value ($)')
        plt.grid(True)
        
        # Plot positions
        plt.subplot(2, 1, 2)
        plt.plot(results['positions'])
        plt.title('Positions Over Time (1: Long, 0: None, -1: Short)')
        plt.ylabel('Position')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Final Portfolio Value: ${results['final_portfolio_value']:.2f}")