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

def train_models(symbol, start_date, end_date, lookback=60, forecast_horizon=5):
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
    
    # 3. Add sentiment features (if available)
    try:
        sentiment = SentimentAnalyzer()
        news_data = sentiment.fetch_news(symbol, start_date, end_date)
        sentiment_df = sentiment.analyze_sentiment(news_data)
        
        # Convert date to datetime for merging and normalize timezone
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.tz_localize(None)
        
        # Merge with price data
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  # Normalize timezone
        
        # Use concat instead of merge
        sentiment_df.set_index('date', inplace=True)
        df.set_index('Date', inplace=True)
        df = pd.concat([df, sentiment_df], axis=1)
        
        # Fill missing sentiment values
        df[['sentiment_mean', 'sentiment_std', 'sentiment_count']] = df[['sentiment_mean', 'sentiment_std', 'sentiment_count']].fillna(method='ffill')
        
    except Exception as e:
        print(f"Skipping sentiment analysis due to: {str(e)}")
    
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
    model_types = ['lstm', 'transformer', 'random_forest', 'gradient_boosting']
    
    for model_type in model_types:
        print(f"Training {model_type} model...")
        input_dim = X_train.shape[2]  # Number of features
        
        trainer = ModelTrainer(model_type=model_type)
        trainer.build_model(input_dim=input_dim)
        
        trainer.train(X_train, y_train, epochs=50 if model_type in ['lstm', 'transformer'] else None)
        
        # Evaluate
        metrics = trainer.evaluate(X_test, y_test)
        print(f"{model_type} model metrics: {metrics}")
        
        # Save model and predictions
        models[model_type] = {
            'trainer': trainer,
            'metrics': metrics,
            'predictions': trainer.predict(X_test)
        }
    
    # 7. Generate trading signals based on ensemble of models
    ensemble_preds = np.zeros(len(y_test))
    for model_type, model_info in models.items():
        # Apply different weights based on model performance
        # Here we use a simple approach, but this could be optimized
        weight = 1.0
        if 'direction_accuracy' in model_info['metrics']:
            weight = model_info['metrics']['direction_accuracy']
        elif 'accuracy' in model_info['metrics']:
            weight = model_info['metrics']['accuracy']
        
        preds = model_info['predictions'].flatten()
        ensemble_preds += weight * preds
    
    # Normalize weights
    ensemble_preds = ensemble_preds / sum(weight for model_type, model_info in models.items())
    
    # Convert predictions to trading signals
    signals = pd.DataFrame({
        'Date': dates_test,
        'Actual': y_test,
        'Predicted': ensemble_preds,
        'Signal': np.where(ensemble_preds > 0, 1, -1)  # 1 for buy, -1 for sell
    })
    
    # Save results
    os.makedirs('models', exist_ok=True)
    joblib.dump(processor, f'models/{symbol}_processor.pkl')
    
    for model_type, model_info in models.items():
        if model_type in ['random_forest', 'gradient_boosting']:
            joblib.dump(model_info['trainer'].model, f'models/{symbol}_{model_type}.pkl')
    
    signals.to_csv(f'models/{symbol}_signals.csv')
    
        # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(signals['Date'], signals['Actual'], label='Actual Returns', alpha=0.7)
    plt.plot(signals['Date'], signals['Predicted'], label='Predicted Returns', color='red', linestyle='--')
    plt.title(f'{symbol} Actual vs Predicted Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{symbol}_predictions.png')
    plt.close()  # Close to prevent display in notebooksreturn {
        'models': models,
        'signals': signals,
        'processor': processor
    }