# regime_detection.py
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class MarketRegimeDetector:
    def __init__(self, n_regimes=2, lookback=120):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.model = None
        self.scaler = StandardScaler()
        
    def extract_features(self, df):
        features = pd.DataFrame(index=df.index)
        
        # Volatility features
        features['volatility'] = df['Close'].pct_change().rolling(window=20).std()
        
        # Trend features
        features['trend'] = df['Close'].pct_change(20)
        
        # Volume features
        if 'Volume' in df.columns:
            features['vol_change'] = df['Volume'].pct_change()
        
        # Correlation features
        if 'High' in df.columns and 'Low' in df.columns:
            features['high_low_ratio'] = df['High'] / df['Low']
        
        # Mean reversion features
        if 'MA_20' in df.columns and 'Close' in df.columns:
            features['ma_distance'] = (df['Close'] - df['MA_20']) / df['MA_20']
        
        return features.dropna()
        
    def fit(self, df):
        features = self.extract_features(df)
        if len(features) < self.lookback:
            print("Warning: Not enough data for regime detection")
            return None
        
        # Standardize features
        X = self.scaler.fit_transform(features.values)
        
        # Fit Gaussian Mixture Model
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42
        )
        self.model.fit(X)
        
        # Get regime labels
        regimes = self.model.predict(X)
        
        # Add regime labels to dataframe
        df_with_regimes = df.copy()
        df_with_regimes['market_regime'] = pd.Series(regimes, index=features.index)
        
        return df_with_regimes
        
    def predict_regime(self, df):
        if self.model is None:
            return None
            
        features = self.extract_features(df)
        if len(features) == 0:
            return None
            
        # Use most recent data point
        latest_features = features.iloc[-1:].values
        X = self.scaler.transform(latest_features)
        
        # Get regime and probability
        regime = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        
        return {
            'regime': regime,
            'confidence': probs[regime],
            'all_probs': probs
        }
