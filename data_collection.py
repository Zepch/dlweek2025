# data_collection.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_market_data(symbols, start_date, end_date, interval='1d'):
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            # Basic data cleaning
            df = df.dropna()
            data[symbol] = df
            print(f"Successfully downloaded data for {symbol}")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
    
    return data

def create_features(df):
    """
    Generate technical indicators and features
    """
    # Copy the dataframe to avoid modifying original
    data = df.copy()
    # data = data.drop(columns=['Dividends', 'Stock Splits'])
    
    # Technical indicators
    # Moving averages
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    
    # MACD
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility
    data['Volatility'] = data['Close'].rolling(window=20).std()
    
    # Return columns
    data['Daily_Return'] = data['Close'].pct_change()
    
    # Drop NaN values resulting from calculations
    data = data.dropna()
    
    return data