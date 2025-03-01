# train.py
from data_collection import fetch_market_data, create_features
from data_preprocessing import FeatureProcessor
from advanced_models import ModelTrainer
from sentiment_analysis import SentimentAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

def calculate_risk_metrics(signals):
    """Calculate comprehensive risk metrics"""
    
    daily_returns = signals['Strategy_Return']
    risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
    
    daily_stats = {
        'mean_return': daily_returns.mean(),
        'std_return': daily_returns.std(),
        'annualized_return': daily_returns.mean() * 252,
        'annualized_vol': daily_returns.std() * np.sqrt(252),
        'sharpe_ratio': (daily_returns.mean() - risk_free_rate) / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else np.nan,
        'total_trades': len(signals[signals['Signal'] != 0]),
        'buy_trades': len(signals[signals['Signal'] == 1]),
        'sell_trades': len(signals[signals['Signal'] == -1]),
        'hold_trades': len(signals[signals['Signal'] == 0])
    }
    
    return daily_stats

def train_models(symbol, start_date, end_date, lookback=126, forecast_horizon=5, epochs=50):
    """
    End-to-end training pipeline for multiple models
    """
    # 1. Fetch market data
    print(f"Fetching data for {symbol}...")
    data_dict = fetch_market_data([symbol], start_date, end_date)
    if symbol not in data_dict:
        print(f"Error: Could not fetch data for {symbol}")
        return None
    
    # 2. Create technical features
    print("Creating technical features...")
    df = create_features(data_dict[symbol])
    
    # 4. Prepare features and target for ML
    print("Preparing features for machine learning...")
    processor = FeatureProcessor(scaling_method='standard')
    X, y, dates = processor.prepare_features(df, lookback=lookback, forecast_horizon=forecast_horizon)
    
    # 5. Split into train/test
    X_train, X_test, y_train, y_test, dates_train, dates_test = processor.create_train_test_split(X, y, dates)
    
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Check for NaN values in the entire dataset
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        print(f"Warning: Training data contains {np.isnan(X_train).sum()} NaN values in X_train and {np.isnan(y_train).sum()} NaN values in y_train")
        
        # Handle NaN values in y_train (if any)
        if np.isnan(y_train).any():
            y_train = np.nan_to_num(y_train, nan=0.0)
            print("Replaced NaN values in y_train with 0.0")
    
    # 6. Train multiple models
    models = {}
    model_types = ['gru', 'transformer', 'random_forest', 'xgboost']
    # model_types = ['gru']
    
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        input_dim = X_train.shape[2]  # Number of features
        
        trainer = ModelTrainer(model_type=model_type)
        trainer.build_model(input_dim=input_dim)
        
        # For neural models, use a specified epoch count; for others, pass None
        trainer.train(X_train, y_train, epochs=epochs if model_type in ['gru', 'transformer'] else None)
        
        # Evaluate
        metrics = trainer.evaluate(X_test, y_test)
        print(f"{model_type} model metrics: {metrics}")
        
        # Save model and predictions
        predictions = trainer.predict(X_test)
        print(f"Predictions shape for {model_type}: {predictions.shape}")
        models[model_type] = {
            'trainer': trainer,
            'metrics': metrics,
            'predictions': predictions
        }
    
    # 7. Generate trading signals based on ensemble of models
    print("\n=== Generating Trading Signals ===")

    # Initialize array for ensemble predictions
    ensemble_preds = np.zeros((len(y_test)))
    
    # Calculate weighted average using non-negative R² scores to ignore poor models
    print("\nEnsemble Weights based on R² scores:")
    total_weight = 0
    for model_type, model_info in models.items():
        r2_value = abs(model_info['metrics']['r2'])
        # Set weight to 0 if negative; otherwise use the r2 value
        weight = r2_value
        total_weight += weight
        print(f"{model_type}: R² = {r2_value:.4f}, weight = {weight:.4f}")
        
        pred = model_info['predictions']
        pred = pred.flatten()
        ensemble_preds += (weight * pred)/total_weight
    
    print("\nPrediction shapes:")
    print(f"Ensemble predictions shape: {ensemble_preds.shape}")
    print(f"Target shape: {y_test.shape}")

    # Create signals DataFrame with expanded metrics
    signals = pd.DataFrame({
        'Date': dates_test,
        'Current_Close': df.loc[dates_test, 'Close'].values,
        'Actual_Return': y_test,  # First day's return
        'Predicted_Return': ensemble_preds,  # First day's prediction
    })
    
    # Generate trading signals with thresholds
    threshold = 0.02
    signals['Signal'] = np.select(
        [
            ensemble_preds[:] > threshold,     # buy signal
            ensemble_preds[:] < -threshold,    # sell signal
        ],
        [1, -1],                    
        default=0                    # Hold signal
    )
    print(signals)
    # Calculate strategy returns
    signals['Strategy_Return'] = signals['Signal'].shift(1) * signals['Actual_Return']
    signals.loc[signals['Signal'].shift(1) == 0, 'Strategy_Return'] = 0
    signals['Cumulative_Return'] = (1 + signals['Strategy_Return']).cumprod()

    # Print detailed trading summary
    print("\nTrading Strategy Analysis:")
    metrics = calculate_risk_metrics(signals)
    print(f"\nTrading Activity:")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Buy Signals: {metrics['buy_trades']} ({metrics['buy_trades']/len(signals)*100:.1f}%)")
    print(f"Sell Signals: {metrics['sell_trades']} ({metrics['sell_trades']/len(signals)*100:.1f}%)")
    print(f"Hold Signals: {metrics['hold_trades']} ({metrics['hold_trades']/len(signals)*100:.1f}%)")
    
    print(f"\nPerformance Metrics:")
    print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
    print(f"Annualized Volatility: {metrics['annualized_vol']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

    # Save results
    os.makedirs('models', exist_ok=True)
    joblib.dump(processor, f'models/{symbol}_processor.pkl')
    
    for model_type, model_info in models.items():
        if model_type in ['random_forest', 'gradient_boosting']:
            joblib.dump(model_info['trainer'].model, f'models/{symbol}_{model_type}.pkl')
    
    signals.to_csv(f'models/{symbol}_signals.csv')
    
    # Plot predictions vs actual returns
    plt.figure(figsize=(12, 6))
    plt.plot(signals['Date'], signals['Actual_Return'], label='Actual Returns', alpha=0.7)
    plt.plot(signals['Date'], signals['Predicted_Return'], label='Predicted Returns', color='red', linestyle='--')
    plt.title(f'{symbol} Actual vs Predicted Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{symbol}_predictions.png')
    plt.close()  # Close to prevent display in notebooks
    
    return {
        'models': models,
        'signals': signals,
        'processor': processor
    }
