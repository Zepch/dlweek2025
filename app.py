from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import json
import os
from train import train_models
from data_collection import fetch_market_data, create_features
from reinforcement_learning import DQNAgent, TradingEnvironment
from advanced_models import ModelTrainer
import plotly.express as px
import plotly
import joblib

app = Flask(__name__)

# Global storage for training logs
training_logs = {
    'gru': {'loss': [], 'val_loss': []},
    'transformer': {'loss': [], 'val_loss': []},
    'random_forest': {'training_progress': []},
    'xgboost': {'training_progress': []}
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_symbols')
def get_symbols():
    # Example symbols or read from a file
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    return jsonify(symbols)

@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.json
    symbol = data.get('symbol', 'AAPL')
    model_type = data.get('model_type', 'random_forest')
    
    # Training parameters
    start_date = data.get('start_date', '2018-01-01')
    end_date = data.get('end_date', '2023-01-01')
    
    try:
        # Simulated training - in production this would call your actual training
        results = train_models(symbol, start_date, end_date, 
                             lookback=63, forecast_horizon=5, epochs=1)
        
        # For demo purposes - return sample metrics
        metrics = {
            'model_type': model_type,
            'symbol': symbol,
            'r2': float(results['models'][model_type]['metrics']['r2']),
            'rmse': float(results['models'][model_type]['metrics']['rmse']),
            'direction_accuracy': float(results['models'][model_type]['metrics']['direction_accuracy']),
            'training_complete': True
        }
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_performance/<symbol>')
def get_performance(symbol):
    try:
        # Load signals data
        signals = pd.read_csv(f'models/{symbol}_signals.csv')
        signals['Date'] = pd.to_datetime(signals['Date'])
        
        # Format for JSON response
        performance_data = {
            'dates': signals['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'cumulative_returns': signals['Cumulative_Return'].tolist(),
            'strategy_returns': signals['Strategy_Return'].tolist(),
            'actual_returns': signals['Actual_Return'].tolist(),
            'predicted_returns': signals['Predicted_Return'].tolist(),
        }
        
        # Calculate metrics
        from train import calculate_risk_metrics
        metrics = calculate_risk_metrics(signals)
        metrics = {k: float(v) if isinstance(v, (np.float64, np.float32, np.int64)) else v 
                  for k, v in metrics.items()}
        
        return jsonify({
            'performance_data': performance_data,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_training_progress/<model_type>')
def get_training_progress(model_type):
    # In a real implementation, this would return actual training progress
    # For now, we'll simulate some training progress
    return jsonify(training_logs[model_type])

if __name__ == '__main__':
    app.run(debug=True)