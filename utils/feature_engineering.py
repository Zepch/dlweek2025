# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_features(df, columns):
    """
    Normalize features to [0,1] range
    """
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

def create_sequences(df, target_col, sequence_length=60):
    """
    Create sequences for time series prediction
    """
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:i+sequence_length].values)
        y.append(df.iloc[i+sequence_length][target_col])
    return np.array(X), np.array(y)

def add_sentiment_features(df, sentiment_data):
    """
    Add market sentiment data from external sources
    """
    # Merge sentiment data based on date
    df = df.merge(sentiment_data, left_index=True, right_index=True, how='left')
    # Fill missing sentiment values
    df['sentiment_score'] = df['sentiment_score'].fillna(0)
    return df