import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression  # Add this import at the top

##############################################
# GRU Model (Advanced Alternative to LSTM)  #
##############################################
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bidirectional=True, dropout=0.1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0  # dropout works only if num_layers > 1
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        out, _ = self.gru(x)
        # Use the representation from the last time step
        out = self.fc(out[:, -1, :])
        return out

######################################################
# Positional Encoding for Transformer Models         #
######################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

######################################################
# Self-Attention Block for Transformer Models         #
######################################################
class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention sublayer
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        # Feedforward sublayer
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

######################################################
# Transformer Model with Attention Blocks            #
######################################################
class TransformerAttentionModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, dropout=0.1):
        super(TransformerAttentionModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.attention_blocks = nn.ModuleList(
            [SelfAttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        for block in self.attention_blocks:
            x = block(x)
        # Use the representation from the last time step
        out = self.fc(x[:, -1, :])
        return out

##############################################
# Hybrid Ensemble Model                      #
##############################################
class HybridModel:
    """
    Ensemble of multiple models for improved performance.
    """
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
    
    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        weighted_preds = np.average(predictions, axis=0, weights=self.weights)
        return weighted_preds

##############################################
# Model Trainer                              #
##############################################
class ModelTrainer:
    def __init__(self, model_type='gru', device='cpu'):
        """
        model_type: 'gru', 'transformer_attention', 'random_forest', 'gradient_boosting', 'xgboost', 'linear'
        """
        self.imputer = SimpleImputer(strategy='mean')
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() and device=='cuda' else 'cpu')
        self.model = None
        
    def build_model(self, input_dim, hidden_dim=256, num_layers=5, output_dim=1):
        if self.model_type == 'gru':
            self.model = GRUModel(input_dim, hidden_dim, num_layers, output_dim,
                                  bidirectional=True, dropout=0.1).to(self.device)
        elif self.model_type == 'transformer':
            self.model = TransformerAttentionModel(
                input_dim=input_dim,
                d_model=hidden_dim,
                nhead=4,
                num_layers=num_layers,
                output_dim=output_dim,
                dropout=0.1
            ).to(self.device)
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=300,
                max_depth=200,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=200,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
    
    def train(self, X_train, y_train, batch_size=32, epochs=10, learning_rate=0.001):
        if self.model_type in ['gru', 'transformer']:
            # Convert numpy arrays to PyTorch tensors
            X_tensor = torch.FloatTensor(X_train).to(self.device)
            y_tensor = torch.FloatTensor(y_train).to(self.device).view(-1, 1)
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            best_loss = float('inf')
            patience_counter = 0
            max_patience = 10
            self.model.train()
            
            for epoch in range(epochs):
                total_loss = 0
                for X_batch, y_batch in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    # Gradient clipping to avoid exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(dataloader)
                scheduler.step(avg_loss)

                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
        else:
            # For non-neural network models, reshape and impute if needed
            X_reshaped = X_train.reshape(X_train.shape[0], -1)
            if np.isnan(X_reshaped).any():
                X_reshaped = self.imputer.fit_transform(X_reshaped)
            self.model.fit(X_reshaped, y_train)
    
    def predict(self, X):
        if self.model_type in ['gru', 'transformer']:
            self.model.eval()
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                predictions = self.model(X_tensor).cpu().numpy()
            return predictions
        else:
            X_reshaped = X.reshape(X.shape[0], -1)
            if np.isnan(X_reshaped).any():
                X_reshaped = self.imputer.transform(X_reshaped)
            return self.model.predict(X_reshaped)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(predictions.flatten(), y_test)[0, 1]
        
        # Calculate errors and their variance
        errors = predictions - y_test
        prediction_variance = np.var(errors)
        direction_accuracy = np.mean((predictions > 0) == (y_test > 0))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'correlation': correlation,
            'variance': prediction_variance,
            'direction_accuracy': direction_accuracy
        }
