# main.py
from data_collection import fetch_market_data, create_features
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def main():
    # Define parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    
    # Fetch market data
    print("Fetching market data...")
    data = fetch_market_data(symbols, start_date, end_date)
    
    # Process data for the first symbol
    if symbols[0] in data:
        print(f"Processing data for {symbols[0]}...")
        processed_data = create_features(data[symbols[0]])
        
        # Display the first few rows
        print(processed_data.head())
        
        # Plot some basic features
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(processed_data.index, processed_data['Close'], label='Close Price')
        plt.plot(processed_data.index, processed_data['MA_20'], label='20-day MA')
        plt.legend()
        plt.title(f'{symbols[0]} Price and Moving Average')
        
        plt.subplot(2, 1, 2)
        plt.plot(processed_data.index, processed_data['RSI'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='-')
        plt.axhline(y=30, color='g', linestyle='-')
        plt.legend()
        plt.title(f'{symbols[0]} RSI')
        
        plt.tight_layout()
        plt.savefig('initial_analysis.png')
        plt.show()
        
        print("Basic analysis completed!")
    
if __name__ == "__main__":
    main()