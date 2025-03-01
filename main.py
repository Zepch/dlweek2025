# main.py
from data_collection import fetch_market_data, create_features
from data_preprocessing import FeatureProcessor
from advanced_models import ModelTrainer, HybridModel
from sentiment_analysis import SentimentAnalyzer
from reinforcement_learning import DQNAgent, TradingEnvironment
from backtest import SimpleBacktester
from train import train_models
from data_quality import DataQualityCheck

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime, timedelta
import os
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns

def main():
    # Define parameters
    symbol = 'AAPL'
    start_date = '2018-01-01'  # Extended training period
    end_date = '2023-01-01'
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 1. Fetch market data and basic analysis
    print("Fetching market data...")
    data = fetch_market_data(symbol, start_date, end_date)
    
    print(f"Processing data for {symbol}...")
    processed_data = create_features(data[symbol[0]])
        
    # 2. Train ML models for prediction
    print("\n==== FETCHING AND TRAINING PREDICTIVE MODELS ====")
    model_results = {}

    print(f"\nTraining models for {symbol}...")
    model_results[symbol] = train_models(symbol, start_date, end_date, 
                                        lookback=63, forecast_horizon=5, epochs=1)
    
    # 2. Create Reinforcement Learning agent
    print("\n==== TRAINING RL TRADING AGENT ====")
    # Use the first symbol for RL demonstration
    df = processed_data.copy()
    
    # Prepare environment
    state_size = len(df.columns) + 2  # Features + position + balance
    action_size = 3  # hold, buy, sell
    
    # Initialize agent
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    env = TradingEnvironment(df=df)
    
    # Training parameters
    batch_size = 32
    num_episodes = 100
    
    # Training loop
    episode_rewards = []
    portfolio_values = []
    
    print(f"Training RL agent for {num_episodes} episodes...")
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            if done:
                agent.remember(state, action, reward, next_state, done)
                total_reward += reward
                portfolio_values.append(info['portfolio_value'])
                break
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        
        # Train the agent using experiences
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_model()
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total reward: {total_reward:.4f}, Portfolio value: {env.portfolio_value[-1]:.2f}")
    
    # Plot RL training progress
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(range(len(episode_rewards)), episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(2, 1, 2)
    plt.plot(range(len(portfolio_values)), portfolio_values)
    plt.title('Portfolio Value')
    plt.xlabel('Episode')
    plt.ylabel('Value ($)')
    
    plt.tight_layout()
    plt.savefig('results/rl_training.png')
    plt.show()
    
    # 3. Generate combined trading signals
    print("\n==== GENERATING COMBINED SIGNALS ====")
    
    # Load ML model signals
    ml_signals = pd.read_csv(f'models/{symbol}_signals.csv')
    ml_signals['Date'] = pd.to_datetime(ml_signals['Date'])
    ml_signals.set_index('Date', inplace=True)
    
    # Generate RL signals on the same test data
    test_data = processed_data.loc[ml_signals.index]

    # Use the trained agent to generate signals
    rl_signals = []

    env = TradingEnvironment(df=test_data)
    state = env.reset()
    done = False

    while not done:
        action = agent.act(state)  # 0=hold, 1=buy, 2=sell
        next_state, _, done, _ = env.step(action)
        
        # Convert to signal format (-1 for sell, 0 for hold, 1 for buy)
        if action == 1:  # Buy
            signal = 1
        elif action == 2:  # Sell
            signal = -1
        else:  # Hold
            signal = 0
            
        rl_signals.append(signal)
        state = next_state

    # Make sure rl_signals is the same length as the test data
    while len(rl_signals) < len(test_data):
        rl_signals.append(0)  # Append hold signals if needed
        
    # If there are too many signals, trim the list
    if len(rl_signals) > len(test_data):
        rl_signals = rl_signals[:len(test_data)]

    # Now assign the signals
    test_data['RL_Signal'] = rl_signals
    
    # Combine ML and RL signals
    test_data['ML_Signal'] = ml_signals['Signal'].values
    test_data['RL_Signal'] = rl_signals
    
    # Ensemble approach - weighted combination
    test_data['Combined_Signal'] = np.where(
        test_data['ML_Signal'] == test_data['RL_Signal'],
        test_data['ML_Signal'],  # If both agree, use that signal
        test_data['ML_Signal'] * 0.7 + test_data['RL_Signal'] * 0.3  # Weighted average if they disagree
    )
    
    # Discretize combined signal
    test_data['Final_Signal'] = np.sign(test_data['Combined_Signal'])
    
    # 4. Backtest the strategy
    print("\n==== BACKTESTING STRATEGY ====")
    
    # Initialize backtester
    backtester = SimpleBacktester(data=test_data, initial_capital=10000)
    
    # Run backtest with ML signals
    print("\nML Model Signals Backtest:")
    backtester.run(signal_column='ML_Signal')
    backtester.print_summary()
    backtester.plot_results()
    
    # Run backtest with RL signals
    print("\nRL Agent Signals Backtest:")
    backtester.run(signal_column='RL_Signal')
    backtester.print_summary()
    backtester.plot_results()
    
    # Run backtest with combined signals
    print("\nCombined Strategy Backtest:")
    backtester.run(signal_column='Final_Signal')
    backtester.print_summary()
    backtester.plot_results()
    
    # 5. Save final model ensemble
    print("\n==== SAVING FINAL MODEL ====")
    
    # Save the combined model and results
    final_results = {
        'ml_models': model_results,
        'rl_agent': agent,
        'test_data': test_data,
        'backtest_results': {
            'final_equity': backtester.equity,
            'total_return': backtester.total_return,
            'sharpe_ratio': backtester.sharpe_ratio,
            'max_drawdown': backtester.max_drawdown
        }
    }
    
    # Save summary results
    summary_df = pd.DataFrame({
        'Metric': ['Total Return (%)', 'Annual Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)'],
        'Value': [
            backtester.total_return,
            backtester.annual_return * 100,
            backtester.sharpe_ratio,
            backtester.max_drawdown
        ]
    })
    
    summary_df.to_csv('results/strategy_performance.csv', index=False)
    print("\nStrategy performance summary saved to 'results/strategy_performance.csv'")
    print("\nAI Trading System Implementation Complete!")

if __name__ == "__main__":
    main()