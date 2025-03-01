# data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_market_data(ticker, start_date, end_date, interval='1d'):
    """
    Fetch historical market data from Yahoo Finance
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data

# Modified add_technical_indicators function
def add_technical_indicators(df):
    """
    Add technical indicators like RSI, MACD without Bollinger Bands
    """
    # Calculate moving averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Removed Bollinger Bands calculation
    
    return df