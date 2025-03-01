import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import torch
import joblib
from data_collection import fetch_market_data, create_features
from data_preprocessing import FeatureProcessor
from advanced_models import ModelTrainer, GRUModel, TransformerAttentionModel
from reinforcement_learning import DQNAgent, TradingEnvironment
from backtest import SimpleBacktester
import xgboost as xgb
from collections import deque
import torch.serialization
from torch.serialization import safe_globals


def load_models():
    """Load all trained models"""
    print("Loading models...")
    try:
        # Load traditional ML models
        random_forest = joblib.load('models/random_forest_model.joblib')
        xgboost_model = xgb.Booster()
        xgboost_model.load_model('models/xgboost_model.json')
        
        gru_model = GRUModel(13, 256, 5, 1,
                              bidirectional=True, dropout=0.1).to('cuda')
        transformer_model = TransformerAttentionModel(
                input_dim=13,
                d_model=256,
                nhead=4,
                num_layers=5,
                output_dim=1,
                dropout=0.1
            ).to('cuda')
        
        # Option 1: Use safe_globals context manager to allow both required classes
        with safe_globals(['numpy.core.multiarray._reconstruct', 'collections.deque']):
            gru_state_dict = torch.load('models/gru_model')
            gru_model.load_state_dict(gru_state_dict)
            
            transformer_state_dict = torch.load('models/transformer_model')
            transformer_model.load_state_dict(transformer_state_dict)
        
        # Load RL agent
        rl_agent = DQNAgent(15,3)
        rl_agent = rl_agent.load_models()
        
        models = {
            'random_forest': random_forest,
            'xgboost': xgboost_model,
            'gru': gru_model,
            'transformer': transformer_model,
            'rl_agent': rl_agent
        }
        print("Models loaded successfully")
        return models
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise





def process_data_and_predict(symbol, eval_start, eval_end, test_start, test_end):
    """Process data and generate predictions"""
    try:
        print(f"Processing request for {symbol}...")
        
        # Convert input dates to timezone-aware datetime
        def convert_to_ny_time(date_str):
            date = pd.to_datetime(date_str)
            ny_date = date.tz_localize('America/New_York').strftime('%Y-%m-%d %H:%M:%S-05:00')
            return ny_date
        
        # Convert all dates
        eval_start_tz = convert_to_ny_time(eval_start)
        eval_end_tz = convert_to_ny_time(eval_end)
        test_start_tz = convert_to_ny_time(test_start)
        test_end_tz = convert_to_ny_time(test_end)
        
        # Fetch data
        print("Fetching market data...")
        data = fetch_market_data(symbol, eval_start, test_end)
        if not data or symbol not in data:
            raise ValueError(f"No data found for symbol {symbol}")
            
        print("Creating features...")
        processed_data = create_features(data[symbol])
        
        # Find closest available trading days
        def find_closest_date(target_date, dates_index, direction='forward'):
            if direction == 'forward':
                available_dates = dates_index[dates_index >= target_date]
                return available_dates[0] if len(available_dates) > 0 else None
            else:
                available_dates = dates_index[dates_index <= target_date]
                return available_dates[-1] if len(available_dates) > 0 else None
        
        # Adjust dates to nearest trading days
        eval_start_adj = find_closest_date(eval_start_tz, processed_data.index, 'forward')
        eval_end_adj = find_closest_date(eval_end_tz, processed_data.index, 'backward')
        test_start_adj = find_closest_date(test_start_tz, processed_data.index, 'forward')
        test_end_adj = find_closest_date(test_end_tz, processed_data.index, 'backward')
        
        if not all([eval_start_adj, eval_end_adj, test_start_adj, test_end_adj]):
            raise ValueError("One or more dates are outside the available data range")
        
        print(f"Adjusted evaluation period: {eval_start_adj} to {eval_end_adj}")
        print(f"Adjusted test period: {test_start_adj} to {test_end_adj}")
        
        # Split data using adjusted dates
        eval_data = processed_data[eval_start_adj:eval_end_adj]
        test_data = processed_data[test_start_adj:test_end_adj]
        
        if eval_data.empty or test_data.empty:
            raise ValueError("No data available for the specified date ranges")
            
        print(f"Evaluation data shape: {eval_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Load models with error handling
        try:
            models = load_models()
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}")
        
        # Generate predictions for each model
        predictions = {}
        r2_scores = {}
        
        for model_name, model in models.items():
            if model_name != 'rl_agent':
                trainer = ModelTrainer(model_type=model_name)
                trainer.model = model
                preds = trainer.predict(eval_data)
                metrics = trainer.evaluate(eval_data, eval_data['target'])
                predictions[model_name] = preds
                r2_scores[model_name] = metrics['r2']
        
        # Combine ML predictions based on R2 scores
        total_weight = sum(max(0, r2) for r2 in r2_scores.values())
        combined_preds = np.zeros(len(test_data))
        
        for model_name, preds in predictions.items():
            weight = max(0, r2_scores[model_name]) / total_weight
            combined_preds += weight * preds
            
        # Generate RL predictions
        rl_agent = models['rl_agent']
        env = TradingEnvironment(df=test_data)
        rl_signals = []
        state = env.reset()
        done = False
        
        while not done:
            action = rl_agent.act(state)
            next_state, _, done, _ = env.step(action)
            rl_signals.append(action - 1)  # Convert to [-1, 0, 1]
            state = next_state
            
        # Combine ML and RL signals
        final_signals = 0.7 * np.sign(combined_preds) + 0.3 * np.array(rl_signals)
        
        # Backtest the strategy
        backtester = SimpleBacktester(test_data)
        backtester.run(final_signals)
        
        # Create performance plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot cumulative returns
        ax1.plot(backtester.equity)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Portfolio Value ($)')
        
        # Plot drawdown
        ax2.plot(backtester.drawdown)
        ax2.set_title('Drawdown Over Time')
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Drawdown (%)')
        
        plt.tight_layout()
        
        # Create performance summary
        summary = f"""
        Performance Summary:
        Total Return: {backtester.total_return:.2f}%
        Annual Return: {backtester.annual_return:.2f}%
        Sharpe Ratio: {backtester.sharpe_ratio:.2f}
        Maximum Drawdown: {backtester.max_drawdown:.2f}%
        """
        
        return fig, summary
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")  # Debug print
        return None, f"Error: {str(e)}"

# Modified interface creation
def create_gradio_interface():
    with gr.Blocks() as app:
        gr.Markdown("# AI Trading Model Interface")
        
        with gr.Row():
            symbol = gr.Textbox(
                label="Stock Symbol", 
                value="AAPL",
                placeholder="Enter stock symbol (e.g., AAPL)"
            )
        
        with gr.Row():
            eval_start = gr.Textbox(
                label="Evaluation Start Date", 
                value="2018-01-01",
                placeholder="YYYY-MM-DD"
            )
            eval_end = gr.Textbox(
                label="Evaluation End Date", 
                value="2021-12-31",
                placeholder="YYYY-MM-DD"
            )
        
        with gr.Row():
            test_start = gr.Textbox(
                label="Test Start Date", 
                value="2022-01-01",
                placeholder="YYYY-MM-DD"
            )
            test_end = gr.Textbox(
                label="Test End Date", 
                value="2023-12-31",
                placeholder="YYYY-MM-DD"
            )
        
        with gr.Row():
            submit_btn = gr.Button("Generate Predictions")
            
        with gr.Row():
            error_output = gr.Textbox(label="Status/Error Messages")
        
        with gr.Row():
            plot_output = gr.Plot(label="Performance Charts")
            metrics_output = gr.Textbox(label="Performance Metrics")
        
        def wrapped_predict(*args):
            try:
                fig, summary = process_data_and_predict(*args)
                return "", fig, summary
            except Exception as e:
                return f"Error: {str(e)}", None, None
        
        submit_btn.click(
            fn=wrapped_predict,
            inputs=[symbol, eval_start, eval_end, test_start, test_end],
            outputs=[error_output, plot_output, metrics_output]
        )
        
    return app

if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False  # Set to True only if you want to create a public URL
    )