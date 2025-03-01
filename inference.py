import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
import os

def test_model(model, test_loader, criterion, device, scaler=None):
    """Test the model on test data and return metrics"""
    model.eval()
    test_loss = 0.0
    price_mae = 0.0
    vol_mae = 0.0
    correct_decisions = 0
    total_samples = 0
    
    # For inverse transformation if scaler is provided
    close_idx = scaler.feature_names_in_.tolist().index('Close') if scaler else None
    
    with torch.no_grad():
        for batch in test_loader:
            X_batch, y_price_batch, y_vol_batch, y_dec_batch = [b.to(device) for b in batch]
            
            # Forward pass
            price_pred, vol_pred, decision_logits = model(X_batch)
            
            # Calculate loss
            loss = criterion((price_pred, vol_pred, decision_logits), 
                            (y_price_batch, y_vol_batch, y_dec_batch))
            test_loss += loss.item()
            
            # Calculate MAEs
            price_mae += F.l1_loss(price_pred.squeeze(), y_price_batch).item()
            vol_mae += F.l1_loss(vol_pred.squeeze(), y_vol_batch).item()
            
            # Calculate decision accuracy
            _, predicted_decisions = torch.max(decision_logits, 1)
            correct_decisions += (predicted_decisions == y_dec_batch).sum().item()
            total_samples += y_dec_batch.size(0)
            
    avg_loss = test_loss / len(test_loader)
    avg_price_mae = price_mae / len(test_loader)
    avg_vol_mae = vol_mae / len(test_loader)
    accuracy = correct_decisions / total_samples
    
    print(f"\nTest Results:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Price MAE: {avg_price_mae:.4f}")
    print(f"Volatility MAE: {avg_vol_mae:.4f}")
    print(f"Decision Accuracy: {accuracy:.2%}")
    
    if scaler:
        # Example inverse transformation for interpretation
        dummy_input = np.zeros((1, len(scaler.feature_names_in_)))
        dummy_input[0, close_idx] = 0.5  # Mid-range value
        example_price = scaler.inverse_transform(dummy_input)[0, close_idx]
        print(f"\nMAE Interpretation: {avg_price_mae:.4f} in scaled terms")
        print(f"Approximately ${avg_price_mae * example_price:.2f} in original price terms")

def inference_example(model, device, test_loader, scaler, num_examples=3):
    """Show example predictions with inverse scaling"""
    model.eval()
    feature_names = scaler.feature_names_in_
    close_idx = list(feature_names).index('Close')
    
    for i, (X, y_price, y_vol, y_dec) in enumerate(test_loader):
        if i >= num_examples:
            break
            
        X = X.to(device)
        with torch.no_grad():
            price_pred, vol_pred, dec_logits = model(X)
        
        # Inverse scaling
        dummy = np.zeros((1, len(feature_names)))
        dummy[0, close_idx] = price_pred[0].item()
        inv_price = scaler.inverse_transform(dummy)[0, close_idx]
        
        actual_dummy = np.zeros((1, len(feature_names)))
        actual_dummy[0, close_idx] = y_price[0].item()
        actual_price = scaler.inverse_transform(actual_dummy)[0, close_idx]
        
        # Decision probabilities
        probs = F.softmax(dec_logits[0], dim=0).cpu().numpy()
        
        print(f"\nExample {i+1}:")
        print(f"Predicted Price: ${inv_price:.2f}")
        print(f"Actual Price:    ${actual_price:.2f}")
        print(f"Price Error:     ${abs(inv_price - actual_price):.2f}")
        print(f"Volatility Pred: {vol_pred[0].item():.4f}")
        print(f"Decision Probs - Sell: {probs[0]:.2%}, Hold: {probs[1]:.2%}, Buy: {probs[2]:.2%}")

if __name__ == "__main__":
    # ... [previous training code] ...
    
    # After training and loading best model
    model.load_state_dict(torch.load(model_save_path))
    
    # Comprehensive testing
    test_model(model, test_loader, criterion, device, scaler)
    
    # Show some concrete examples
    inference_example(model, device, test_loader, scaler)
    
    # Single sequence inference example
    test_sample, _, _, _ = next(iter(test_loader))
    price_pred, vol_pred, dec_probs = inference(model, test_sample[0], device, model_save_path, scaler)
    print("\nSingle Sequence Prediction:")
    print(f"Predicted Price: {price_pred:.2f}")
    print(f"Predicted Volatility: {vol_pred:.4f}")
    print(f"Decision Probabilities: {dec_probs}")