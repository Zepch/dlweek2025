# advanced_models.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        x = self.input_projection(x)
        
        # Create a mask to prevent attention to padding
        x = self.transformer_encoder(x)
        
        # Take the representation from the last time step
        output = self.output_layer(x[:, -1, :])
        return output

class HybridModel:
    """
    Ensemble of multiple models for improved performance
    """
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
    
    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        weighted_preds = np.average(predictions, axis=0, weights=self.weights)
        return weighted_preds

class ModelTrainer:
    def __init__(self, model_type='lstm', device='cpu'):
        self.imputer = SimpleImputer(strategy='mean')
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() and device=='cuda' else 'cpu')
        self.model = None
        
    def build_model(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        if self.model_type == 'lstm':
            self.model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(self.device)
        elif self.model_type == 'transformer':
            self.model = TransformerModel(
                input_dim=input_dim, 
                d_model=hidden_dim,
                nhead=4,  # Number of attention heads
                num_layers=num_layers,
                dim_feedforward=hidden_dim*4,
                output_dim=output_dim
            ).to(self.device)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
    
    def train(self, X_train, y_train, batch_size=32, epochs=50, learning_rate=0.001):
        if self.model_type in ['lstm', 'transformer']:
            # Convert numpy arrays to PyTorch tensors
            X_tensor = torch.FloatTensor(X_train).to(self.device)
            y_tensor = torch.FloatTensor(y_train).to(self.device).reshape(-1, 1)
            
            # Create DataLoader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Add learning rate scheduler for adaptive learning rate
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            # Initialize tracking variables
            best_loss = float('inf')
            patience_counter = 0
            max_patience = 10  # Early stopping threshold
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                nan_detected = False
                
                for X_batch, y_batch in dataloader:
                    # Check for NaN values in input
                    if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                        print(f"Warning: NaN values in input batch. Skipping.")
                        continue
                    
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    
                    # Check for NaN in outputs
                    if torch.isnan(outputs).any():
                        print(f"Warning: NaN values in model output during epoch {epoch+1}. Attempting recovery.")
                        nan_detected = True
                        break
                    
                    loss = criterion(outputs, y_batch)
                    
                    # Check for NaN loss
                    if torch.isnan(loss).item():
                        print(f"Warning: NaN loss in epoch {epoch+1}. Attempting recovery.")
                        nan_detected = True
                        break
                    
                    loss.backward()
                    
                    # Apply gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    total_loss += loss.item()
                
                # Handle NaN detection
                if nan_detected:
                    print("NaN values detected. Reinitializing weights and reducing learning rate.")
                    # Reinitialize the model
                    if self.model_type == 'lstm':
                        input_dim = self.model.lstm.input_size
                        hidden_dim = self.model.lstm.hidden_size
                        num_layers = self.model.lstm.num_layers
                        self.model = LSTMModel(input_dim, hidden_dim, num_layers, 1).to(self.device)
                    elif self.model_type == 'transformer':
                        # Recreate transformer with more stable initialization
                        d_model = self.model.input_projection.out_features
                        input_dim = self.model.input_projection.in_features
                        self.model = TransformerModel(
                            input_dim=input_dim,
                            d_model=d_model,
                            nhead=4,
                            num_layers=2,
                            dim_feedforward=d_model*4,
                            output_dim=1,
                            dropout=0.1
                        ).to(self.device)
                    
                    # Reset optimizer with lower learning rate
                    learning_rate *= 0.5
                    print(f"New learning rate: {learning_rate}")
                    optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
                    continue
                
                # Print statistics
                avg_loss = total_loss / len(dataloader)
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
                
                # Update scheduler
                scheduler.step(avg_loss)
                
                # Early stopping check
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
        
        else:  # For sklearn models
            # For classification, convert continuous targets to binary (up/down)
            y_binary = (y_train > 0).astype(int)
            
            # Reshape the data
            X_reshaped = X_train.reshape(X_train.shape[0], -1)
            
            # Check for and handle NaN values
            if np.isnan(X_reshaped).any():
                print(f"Detected {np.isnan(X_reshaped).sum()} NaN values in training data. Imputing...")
                X_reshaped = self.imputer.fit_transform(X_reshaped)
            
            self.model.fit(X_reshaped, y_binary)
            
    def predict(self, X):
        if self.model_type in ['lstm', 'transformer']:
            self.model.eval()
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                predictions = self.model(X_tensor).cpu().numpy()
            return predictions
        else:
            X_reshaped = X.reshape(X.shape[0], -1)
            
            # Handle NaN values in prediction data
            if np.isnan(X_reshaped).any():
                X_reshaped = self.imputer.transform(X_reshaped)
                
            return self.model.predict_proba(X_reshaped)[:, 1]
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        
        if self.model_type in ['lstm', 'transformer']:
            # Regression metrics
            mse = ((predictions - y_test.reshape(-1, 1)) ** 2).mean()
            rmse = np.sqrt(mse)
            return {
                'mse': mse,
                'rmse': rmse,
                'direction_accuracy': np.mean((predictions > 0) == (y_test > 0))
            }
        else:
            # Classification metrics
            y_binary = (y_test > 0).astype(int)
            y_pred = (predictions > 0.5).astype(int)
            
            return {
                'accuracy': accuracy_score(y_binary, y_pred),
                'precision': precision_score(y_binary, y_pred),
                'recall': recall_score(y_binary, y_pred),
                'f1': f1_score(y_binary, y_pred)
            }

# For importing the LSTM model
from model import LSTMModel