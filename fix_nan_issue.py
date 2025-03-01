# fix_nan_issue.py
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd

def apply_fixes():
    """Apply fixes to the existing code to handle common errors"""
    print("Applying fixes to the codebase...")
    
    # Fix the advanced_models.py file
    with open('advanced_models.py', 'r') as file:
        content = file.read()
    
    # Add imputer to the ModelTrainer class
    updated_content = content.replace(
        'def __init__(self, model_type=\'lstm\', device=\'cpu\'):',
        'def __init__(self, model_type=\'lstm\', device=\'cpu\'):\n        self.imputer = SimpleImputer(strategy=\'mean\')'
    )
    
    # Update the train method to handle NaN values
    updated_content = updated_content.replace(
        '        else:  # For sklearn models\n            # For classification, convert continuous targets to binary (up/down)\n            y_binary = (y_train > 0).astype(int)\n            self.model.fit(X_train.reshape(X_train.shape[0], -1), y_binary)',
        '        else:  # For sklearn models\n            # For classification, convert continuous targets to binary (up/down)\n            y_binary = (y_train > 0).astype(int)\n            \n            # Reshape the data\n            X_reshaped = X_train.reshape(X_train.shape[0], -1)\n            \n            # Check for and handle NaN values\n            if np.isnan(X_reshaped).any():\n                print(f"Detected {np.isnan(X_reshaped).sum()} NaN values in training data. Imputing...")\n                X_reshaped = self.imputer.fit_transform(X_reshaped)\n            \n            self.model.fit(X_reshaped, y_binary)'
    )
    
    # Update the predict method to handle NaN values
    updated_content = updated_content.replace(
        '            return self.model.predict_proba(X.reshape(X.shape[0], -1))[:, 1]',
        '            X_reshaped = X.reshape(X.shape[0], -1)\n            \n            # Handle NaN values in prediction data\n            if np.isnan(X_reshaped).any():\n                X_reshaped = self.imputer.transform(X_reshaped)\n                \n            return self.model.predict_proba(X_reshaped)[:, 1]'
    )
    
    # Add import for SimpleImputer
    updated_content = updated_content.replace(
        'import torch.optim as optim',
        'import torch.optim as optim\nfrom sklearn.impute import SimpleImputer'
    )
    
    with open('advanced_models.py', 'w') as file:
        file.write(updated_content)
    
    # Fix data_preprocessing.py to handle timestamps better
    with open('data_preprocessing.py', 'r') as file:
        content = file.read()
    
    # Update prepare_features method to be more robust
    updated_content = content.replace(
        '        # Select features - exclude non-numerical columns\n        feature_columns = []\n        for col in df.columns:\n            # Only include columns that can be converted to float\n            if pd.api.types.is_numeric_dtype(df[col]) and col not in [\'Volume\']:\n                feature_columns.append(col)',
        '        # Select features - exclude non-numerical columns\n        feature_columns = []\n        for col in df.columns:\n            # Only include columns that can be converted to float\n            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_dtype(df[col]) and col not in [\'Volume\']:\n                feature_columns.append(col)\n        \n        print(f"Using features: {feature_columns}")'
    )
    
    with open('data_preprocessing.py', 'w') as file:
        file.write(updated_content)
    
    print("Fixes applied successfully!")

if __name__ == "__main__":
    apply_fixes()