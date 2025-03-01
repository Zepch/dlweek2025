# main.py
import argparse
from backtesting.backtest_engine import BacktestEngine
from utils.data_loader import fetch_market_data
import json


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='AI Trading Strategy')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2018-01-01', help='Backtest start date')
    parser.add_argument('--end_date', type=str, default='2023-01-01', help='Backtest end date')
    parser.add_argument('--initial_capital', type=float, default=100000.0, help='Initial capital for backtesting')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize backtest engine
    engine = BacktestEngine(config)
    
    # Load and prepare data
    print(f"Loading data for {args.ticker} from {args.start_date} to {args.end_date}")
    engine.load_data(args.ticker, args.start_date, args.end_date)
    
    # Prepare features
    X_train, y_train, X_test, y_test, _ = engine.prepare_features(
        sequence_length=config.get('sequence_length', 60)
    )
    
    # Initialize and train strategy
    print("Initializing and training strategy models...")
    engine.initialize_strategy(X_train, y_train)
    
    # Run backtest
    print("Running backtest...")
    results = engine.run_backtest(X_test, y_test, initial_capital=args.initial_capital)
    
    # Plot results
    engine.plot_results(results)

if __name__ == "__main__":
    main()