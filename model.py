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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#####################
# DATA PREPARATION  #
#####################

def get_stock_data(tickers, period='5y'):
    """Fetch historical data for multiple stocks using yfinance."""
    dfs = []
    
    if isinstance(tickers, str):
        tickers = [tickers]  # Handle single ticker input
    
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        df['Ticker'] = ticker  # Add identifier column
        dfs.append(df)
    
    # Combine all dataframes and sort
    combined_df = pd.concat(dfs).sort_index()
    return combined_df

def create_features(df):
    """Compute technical indicators for multiple stocks."""
    # Ensure Date column is datetime type
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Group by ticker to prevent data leakage between stocks
    grouped = df.groupby('Ticker', group_keys=False)
    
    def calculate_features(group):
        """Calculate features for a single stock's data"""
        # Price-based features
        group['SMA_20'] = group['Close'].rolling(window=20).mean()
        group['SMA_50'] = group['Close'].rolling(window=50).mean()
        group['EMA_12'] = group['Close'].ewm(span=12, adjust=False).mean()
        group['EMA_26'] = group['Close'].ewm(span=26, adjust=False).mean()
        
        # Momentum indicators
        rsi = RSIIndicator(group['Close'])
        group['RSI'] = rsi.rsi()
        
        macd = MACD(group['Close'])
        group['MACD'] = macd.macd()
        group['MACD_signal'] = macd.macd_signal()
        group['MACD_hist'] = macd.macd_diff()
        
        # Volatility features
        bb = BollingerBands(group['Close'])
        group['BB_upper'] = bb.bollinger_hband()
        group['BB_middle'] = bb.bollinger_mavg()
        group['BB_lower'] = bb.bollinger_lband()
        group['BB_width'] = (group['BB_upper'] - group['BB_lower']) / group['BB_middle']
        
        # Volume features
        group['Volume_MA_20'] = group['Volume'].rolling(window=20).mean()
        group['Volume_Change'] = group['Volume'].pct_change()
        
        # Return-based features
        group['Returns'] = group['Close'].pct_change()
        group['Volatility'] = group['Returns'].rolling(window=20).std() * np.sqrt(252)
        group['Log_Returns'] = np.log1p(group['Returns'])
        
        # Price transformations
        group['Close_Open_Ratio'] = group['Close'] / group['Open']
        group['High_Low_Spread'] = group['High'] - group['Low']
        
        # Drop original columns we don't need as features
        group = group.drop(columns=['Dividends', 'Stock Splits'])
        
        return group.dropna()
    
    # Apply feature engineering to each stock group
    df = grouped.apply(calculate_features)
    
    # Add temporal features using Date column instead of index
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    
    # Add cyclical encoding for temporal features
    df['Day_sin'] = np.sin(2 * np.pi * df['Day_of_Week']/7)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day_of_Week']/7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
    
    return df

def prepare_market_data(df_market):
    """
    Scales the features and computes three targets:
      - y_price: next day's 'Close' (regression target)
      - y_decision: 3-class target (0: Sell, 1: Hold, 2: Buy)
      - y_volatility: computed from original prices
    Returns: X_data, y_price, y_decision, y_volatility and the fitted scaler.
    """
    # Store ticker information and date
    tickers = df_market['Ticker']
    dates = df_market['Date']
    
    # Select only numeric features for scaling
    numeric_cols = df_market.select_dtypes(include=np.number).columns.tolist()
    df_numeric = df_market[numeric_cols]
    
    # Initialize and fit scaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # Convert to tensors
    X_data = torch.FloatTensor(scaled_data[:-1])
    close_idx = numeric_cols.index('Close')
    y_price = torch.FloatTensor(scaled_data[1:, close_idx])
    
    # Compute decision targets
    original_close = df_numeric['Close'].values
    diff = original_close[1:] - original_close[:-1]
    threshold = np.std(diff) * 0.5  # Dynamic threshold based on price volatility
    decision = np.where(diff > threshold, 2, np.where(diff < -threshold, 0, 1))
    y_decision = torch.LongTensor(decision)
    
    # Compute volatility
    returns = np.diff(original_close) / original_close[:-1]
    volatility = pd.Series(returns).rolling(window=20).std().values * np.sqrt(252)
    y_volatility = torch.FloatTensor(volatility)
    
    return X_data, y_price, y_decision, y_volatility, scaler, tickers, dates

def create_sequences(X_data, y_price, y_decision, y_volatility, window_size):
    """
    Converts 2D feature tensor into sequences (sliding window).
    Returns:
       X_seq: [num_samples, window_size, num_features]
       y_price_seq, y_decision_seq, y_vol_seq: targets at the end of each sequence.
    """
    X_seq, y_price_seq, y_dec_seq, y_vol_seq = [], [], [], []
    for i in range(len(X_data) - window_size):
        X_seq.append(X_data[i:i+window_size])
        y_price_seq.append(y_price[i+window_size])
        y_dec_seq.append(y_decision[i+window_size])
        y_vol_seq.append(y_volatility[i+window_size])
    return torch.stack(X_seq), torch.stack(y_price_seq), torch.stack(y_dec_seq), torch.stack(y_vol_seq)

#########################
# ENHANCED MODEL ARCHITECTURE    #
#########################

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, lstm_outputs):
        batch_size = lstm_outputs.size(0)
        
        # Split into multiple heads
        Q = self.query(lstm_outputs).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        K = self.key(lstm_outputs).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        V = self.value(lstm_outputs).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        
        # Compute attention
        energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)
        context = torch.matmul(attention, V)
        
        # Combine heads
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads*self.head_dim)
        return self.out(context), attention

class EnhancedTradingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, 
                 dropout=0.4, seq_len=30, num_classes=3):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim//2, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_dim)
        
        # Feature processing
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim*2)
        self.dropout = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        
        # Prediction heads
        self.price_head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_dim, 1)
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out)
        
        # Combine LSTM and attention outputs
        context = attn_out[:, -1] + lstm_out[:, -1, :]
        
        # Process features
        x = self.bn1(context)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.bn2(x)
        
        # Generate predictions
        price_pred = self.price_head(x)
        volatility_pred = self.volatility_head(x)
        decision_logits = self.decision_head(x)
        
        return price_pred, volatility_pred, decision_logits

class EnhancedCompositeLoss(nn.Module):
    def __init__(self, price_weight=1.0, vol_weight=0.5, decision_weight=0.7,
                 label_smoothing=0.1):
        super().__init__()
        self.price_loss = nn.HuberLoss(delta=1.0)
        self.vol_loss = nn.HuberLoss(delta=0.5)
        self.decision_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.weights = torch.tensor([price_weight, vol_weight, decision_weight])

    def forward(self, predictions, targets):
        price_pred, vol_pred, decision_logits = predictions
        price_true, vol_true, decision_true = targets
        
        price_loss = self.price_loss(price_pred.squeeze(), price_true)
        vol_loss = self.vol_loss(vol_pred.squeeze(), vol_true)
        decision_loss = self.decision_loss(decision_logits, decision_true)
        
        total_loss = (self.weights[0] * price_loss +
                     self.weights[1] * vol_loss +
                     self.weights[2] * decision_loss)
        
        return total_loss

#########################
# IMPROVED TRAINING     #
#########################

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path):
    best_val_loss = float('inf')
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            X_batch, y_price_batch, y_vol_batch, y_dec_batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
            
            predictions = model(X_batch)
            loss = criterion(predictions, (y_price_batch, y_vol_batch, y_dec_batch))
            
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                X_batch, y_price_batch, y_vol_batch, y_dec_batch = [b.to(device) for b in batch]
                predictions = model(X_batch)
                loss = criterion(predictions, (y_price_batch, y_vol_batch, y_dec_batch))
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        
        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), model_save_path)
            saved_str = " (saved)"
        else:
            saved_str = ""
            
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}{saved_str}")
    
    print("Training complete.")

#########################
# MAIN EXECUTION        #
#########################

if __name__ == "__main__":
    # Get data for multiple stocks
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'INTC', 'AMD', 'IBM']
    df = get_stock_data(tickers, period='5y').reset_index()

    # Create features and preserve date information
    df_market = create_features(df)  # Convert index to 'Date' column
    
    # Prepare data
    X_data, y_price, y_decision, y_volatility, scaler, tickers, dates = prepare_market_data(df_market)
    
    # Create sequences
    window_size = 63  # 3-month trading window
    X_seq, y_price_seq, y_dec_seq, y_vol_seq = create_sequences(X_data, y_price, y_decision, y_volatility, window_size)
    
    # Verify shapes
    print(f"Data shapes - X: {X_seq.shape}, y_price: {y_price_seq.shape}, "
          f"y_decision: {y_dec_seq.shape}, y_volatility: {y_vol_seq.shape}")
    
    # Split into train/val/test (70/15/15)
    total = len(X_seq)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    
    dataset = TensorDataset(X_seq, y_price_seq, y_vol_seq, y_dec_seq)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, total - train_size - val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Initialize model
    input_dim = X_seq.shape[2]
    model = EnhancedTradingModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.4,
        seq_len=window_size
    ).to(device)
    
    # Training setup
    criterion = EnhancedCompositeLoss(
        price_weight=1.0,
        vol_weight=0.5,
        decision_weight=0.7,
        label_smoothing=0.1
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs = 1000
    model_save_path = "best_trading_model.pth"
    
    # Train with early stopping
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path)
    
    # Final evaluation on test set
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            X_batch, y_price_batch, y_vol_batch, y_dec_batch = [b.to(device) for b in batch]
            predictions = model(X_batch)
            loss = criterion(predictions, (y_price_batch, y_vol_batch, y_dec_batch))
            test_loss += loss.item()
    print(f"\nFinal Test Loss: {test_loss/len(test_loader):.4f}")