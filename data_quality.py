# data_quality.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer

class DataQualityCheck:
    """
    Class to check and fix data quality issues
    """
    def __init__(self):
        self.issues_found = []
        
    def check_dataframe(self, df, name="Dataset"):
        """Check a pandas dataframe for common issues"""
        self.issues_found = []
        report = {}
        
        # Check for NaN values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            self.issues_found.append(f"NaN values found: {nan_count}")
            report["nan_count"] = nan_count
            report["nan_columns"] = df.columns[df.isna().any()].tolist()
            
        # Check for infinite values
        inf_mask = np.isinf(df.select_dtypes(include=np.number))
        inf_count = inf_mask.sum().sum()
        if inf_count > 0:
            self.issues_found.append(f"Infinite values found: {inf_count}")
            report["inf_count"] = inf_count
            
        # Check for duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            self.issues_found.append(f"Duplicate rows found: {dup_count}")
            report["duplicate_count"] = dup_count
            
        # Check for high correlation features
        if df.select_dtypes(include=np.number).shape[1] > 1:
            try:
                corr_matrix = df.select_dtypes(include=np.number).corr()
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.95:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                
                if high_corr_pairs:
                    self.issues_found.append(f"High correlation pairs found: {len(high_corr_pairs)}")
                    report["high_corr_pairs"] = high_corr_pairs
            except Exception as e:
                self.issues_found.append(f"Error calculating correlations: {str(e)}")
        
        # Print summary
        if self.issues_found:
            print(f"\n===== Data Quality Report for {name} =====")
            for issue in self.issues_found:
                print(f"⚠️ {issue}")
        else:
            print(f"✅ No data quality issues found in {name}")
            
        return report
    
    def fix_dataframe(self, df, impute_method='mean', drop_duplicates=True, inplace=False):
        """Fix common data quality issues in a dataframe"""
        if not inplace:
            df = df.copy()
            
        # Fix NaN values
        if 'nan_count' in self.check_dataframe(df, name="Dataset to fix"):
            print(f"Imputing missing values using {impute_method} method...")
            
            num_cols = df.select_dtypes(include=np.number).columns
            
            if impute_method == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif impute_method == 'median':
                imputer = SimpleImputer(strategy='median')
            elif impute_method == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            else:
                imputer = SimpleImputer(strategy='mean')
                
            df[num_cols] = imputer.fit_transform(df[num_cols])
        
        # Fix duplicates
        if drop_duplicates and df.duplicated().sum() > 0:
            df = df.drop_duplicates()
            print(f"Dropped {df.duplicated().sum()} duplicate rows")
            
        return df
    
    def check_arrays(self, X, y=None, names=("X", "y")):
        """Check numpy arrays for common issues"""
        issues = {}
        
        # Check X array
        x_nan_count = np.isnan(X).sum()
        if x_nan_count > 0:
            print(f"⚠️ {names[0]} contains {x_nan_count} NaN values")
            issues[f"{names[0]}_nan_count"] = x_nan_count
            
        x_inf_count = np.isinf(X).sum()
        if x_inf_count > 0:
            print(f"⚠️ {names[0]} contains {x_inf_count} infinite values")
            issues[f"{names[0]}_inf_count"] = x_inf_count
        
        # Check y array if provided
        if y is not None:
            y_nan_count = np.isnan(y).sum()
            if y_nan_count > 0:
                print(f"⚠️ {names[1]} contains {y_nan_count} NaN values")
                issues[f"{names[1]}_nan_count"] = y_nan_count
                
            y_inf_count = np.isinf(y).sum() if np.issubdtype(y.dtype, np.number) else 0
            if y_inf_count > 0:
                print(f"⚠️ {names[1]} contains {y_inf_count} infinite values")
                issues[f"{names[1]}_inf_count"] = y_inf_count
        
        if not issues:
            print(f"✅ No issues found in the arrays")
            
        return issues
    
    def fix_arrays(self, X, y=None, impute_strategy='mean'):
        """Fix common data quality issues in numpy arrays"""
        # Fix X array
        if np.isnan(X).any() or np.isinf(X).any():
            print(f"Fixing NaN/Inf values in X using {impute_strategy}...")
            
            original_shape = X.shape
            X_reshaped = X.reshape(X.shape[0], -1)
            
            if impute_strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif impute_strategy == 'median':
                imputer = SimpleImputer(strategy='median')
            elif impute_strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            else:
                imputer = SimpleImputer(strategy='mean')
                
            X_fixed = imputer.fit_transform(X_reshaped)
            X = X_fixed.reshape(original_shape)
            
        # Fix y array if provided
        if y is not None and (np.isnan(y).any() or (np.issubdtype(y.dtype, np.number) and np.isinf(y).any())):
            print("Fixing NaN/Inf values in y...")
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if y is not None:
            return X, y
        else:
            return X
    
    def visualize_missing_data(self, df):
        """Visualize missing data in a dataframe"""
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title('Missing Data Visualization')
        plt.tight_layout()
        plt.savefig('results/missing_data_visualization.png')
        plt.show()