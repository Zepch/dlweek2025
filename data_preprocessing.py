# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

class FeatureProcessor:
    def __init__(self, scaling_method='minmax'):
        self.scaling_method = scaling_method
        self.feature_scaler = MinMaxScaler() if scaling_method == 'minmax' else StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.pca = None
        
    def prepare_features(self, df, lookback=60, forecast_horizon=5, pca_components=None):
        """
        Convert time series data to supervised learning problem with lookback window
        """
        # Select features - exclude non-numerical columns
        feature_columns = []
        for col in df.columns:
            # Only include columns that can be converted to float
            if pd.api.types.is_numeric_dtype(df[col]) and col not in ['Volume']:
                feature_columns.append(col)
        
        print(f"Using features: {feature_columns}")
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(df[feature_columns])
        scaled_df = pd.DataFrame(scaled_features, columns=feature_columns, index=df.index)
        
        # Apply PCA if specified
        if pca_components:
            self.pca = PCA(n_components=pca_components)
            pca_result = self.pca.fit_transform(scaled_features)
            scaled_df = pd.DataFrame(pca_result, 
                                    columns=[f'PC{i+1}' for i in range(pca_components)],
                                    index=df.index)
            feature_columns = scaled_df.columns
            
        # Target variable - future returns
        scaled_df['Target'] = df['Close'].pct_change(forecast_horizon).shift(-forecast_horizon)
        
        # Create sequences
        X, y, dates = [], [], []
        for i in range(lookback, len(scaled_df) - forecast_horizon):
            X.append(scaled_df[feature_columns].iloc[i-lookback:i].values)
            y.append(scaled_df['Target'].iloc[i])
            dates.append(scaled_df.index[i])
            
        return np.array(X), np.array(y), dates
    
    def create_train_test_split(self, X, y, dates, test_ratio=0.2):
        """
        Split data into train and test sets while maintaining temporal order
        """
        split_idx = int(len(X) * (1 - test_ratio))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_train, dates_test = dates[:split_idx], dates[split_idx:]
        
        return X_train, X_test, y_train, y_test, dates_train, dates_test
    
    def feature_importance(self, model, feature_names):
        """
        Calculate feature importance for tree-based models
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in indices],
                'Importance': importances[indices]
            })
            
            return importance_df
        else:
            return None