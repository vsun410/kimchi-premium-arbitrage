#!/usr/bin/env python3
"""
Encoder-Decoder LSTM with Attention for Kimchi Premium Prediction
Task #15: ED-LSTM Model Implementation
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.utils.logger import logger


class AttentionLayer(nn.Module):
    """
    Attention mechanism for sequence-to-sequence models
    """
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        
    def forward(
        self,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of attention layer
        
        Args:
            hidden: Current decoder hidden state [batch_size, hidden_size]
            encoder_outputs: All encoder outputs [batch_size, seq_len, hidden_size]
            
        Returns:
            context: Weighted context vector
            attention_weights: Attention weights
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Repeat hidden state for all time steps
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Calculate attention energies
        energy = torch.tanh(self.attention(torch.cat([hidden, encoder_outputs], dim=2)))
        energy = energy.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        
        # Calculate attention weights
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch_size, 1, hidden_size]
        attention_weights = torch.bmm(v, energy).squeeze(1)  # [batch_size, seq_len]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights to encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # [batch_size, hidden_size]
        
        return context, attention_weights


class EncoderLSTM(nn.Module):
    """
    Encoder LSTM for feature extraction
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of encoder
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            hidden: Initial hidden state
            
        Returns:
            outputs: All encoder outputs
            hidden: Final hidden state
        """
        outputs, hidden = self.lstm(x, hidden)
        outputs = self.dropout(outputs)
        return outputs, hidden


class DecoderLSTM(nn.Module):
    """
    Decoder LSTM with attention for prediction
    """
    
    def __init__(
        self,
        output_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(hidden_size)
            lstm_input_size = hidden_size + output_size
        else:
            lstm_input_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_size, output_size)
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of decoder
        
        Args:
            x: Input tensor [batch_size, 1, output_size]
            hidden: Hidden state from encoder
            encoder_outputs: All encoder outputs for attention
            
        Returns:
            output: Predicted output
            hidden: Updated hidden state
            attention_weights: Attention weights (if using attention)
        """
        attention_weights = None
        
        if self.use_attention and encoder_outputs is not None:
            # Apply attention
            context, attention_weights = self.attention(hidden[0][-1], encoder_outputs)
            # Concatenate context with input
            x = torch.cat([x, context.unsqueeze(1)], dim=2)
        
        output, hidden = self.lstm(x, hidden)
        output = self.dropout(output)
        output = self.output_projection(output)
        
        return output, hidden, attention_weights


class EDLSTMModel(nn.Module):
    """
    Complete Encoder-Decoder LSTM model with attention
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        output_size: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
        decoder_length: int = 24  # Predict next 24 hours
    ):
        super(EDLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.decoder_length = decoder_length
        self.use_attention = use_attention
        
        # Encoder
        self.encoder = EncoderLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = DecoderLSTM(
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention
        )
        
        logger.info(
            f"ED-LSTM initialized: input={input_size}, hidden={hidden_size}, "
            f"layers={num_layers}, attention={use_attention}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of ED-LSTM
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            target: Target tensor for teacher forcing [batch_size, decoder_len, output_size]
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Predicted outputs
            attention_weights: Attention weights over time
        """
        batch_size = x.size(0)
        device = x.device
        
        # Encode
        encoder_outputs, hidden = self.encoder(x)
        
        # Initialize decoder input (can be learned or zeros)
        decoder_input = torch.zeros(batch_size, 1, self.output_size).to(device)
        
        # Store decoder outputs
        outputs = []
        attention_weights = []
        
        # Decode
        for t in range(self.decoder_length):
            output, hidden, attn = self.decoder(
                decoder_input, hidden, 
                encoder_outputs if self.use_attention else None
            )
            
            outputs.append(output)
            if attn is not None:
                attention_weights.append(attn)
            
            # Teacher forcing
            if target is not None and np.random.random() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1, :]
            else:
                decoder_input = output
        
        # Stack outputs
        outputs = torch.cat(outputs, dim=1)
        
        if attention_weights:
            attention_weights = torch.stack(attention_weights, dim=1)
        else:
            attention_weights = None
        
        return outputs, attention_weights


class LSTMTrainer:
    """
    Training pipeline for ED-LSTM model
    """
    
    def __init__(
        self,
        model: EDLSTMModel,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"Trainer initialized on {device}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        teacher_forcing_ratio: float = 0.5
    ) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            teacher_forcing_ratio: Teacher forcing ratio
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = self.model(inputs, targets, teacher_forcing_ratio)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass (no teacher forcing)
                outputs, _ = self.model(inputs, targets, teacher_forcing_ratio=0)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        early_stopping_patience: int = 10,
        save_dir: str = "models/"
    ) -> Dict:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save best model
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for epoch in range(n_epochs):
            # Decay teacher forcing ratio
            teacher_forcing_ratio = max(0.5 * (0.99 ** epoch), 0.1)
            
            # Train
            train_loss = self.train_epoch(train_loader, teacher_forcing_ratio)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{n_epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(os.path.join(save_dir, "best_model.pth"))
                logger.info(f"Best model saved (val_loss: {val_loss:.6f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def predict(
        self,
        inputs: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions
        
        Args:
            inputs: Input tensor
            return_attention: Whether to return attention weights
            
        Returns:
            predictions: Predicted values
            attention_weights: Attention weights (if requested)
        """
        self.model.eval()
        
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs, attention = self.model(inputs, teacher_forcing_ratio=0)
            
            predictions = outputs.cpu().numpy()
            
            if return_attention and attention is not None:
                attention = attention.cpu().numpy()
            else:
                attention = None
        
        return predictions, attention
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'output_size': self.model.output_size,
                'num_layers': self.model.num_layers,
                'decoder_length': self.model.decoder_length,
                'use_attention': self.model.use_attention
            }
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {filepath}")


def create_sequences(
    data: np.ndarray,
    seq_length: int = 168,  # 7 days * 24 hours
    pred_length: int = 24,  # Predict next 24 hours
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction
    
    Args:
        data: Input data [n_samples, n_features]
        seq_length: Length of input sequence
        pred_length: Length of prediction sequence
        stride: Stride for sliding window
        
    Returns:
        X: Input sequences
        y: Target sequences
    """
    X, y = [], []
    
    for i in range(0, len(data) - seq_length - pred_length + 1, stride):
        X.append(data[i:i + seq_length])
        # Predict only the target column (e.g., kimchi premium)
        y.append(data[i + seq_length:i + seq_length + pred_length, 0])
    
    return np.array(X), np.array(y)


def prepare_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare data loaders for training
    
    Args:
        X: Input sequences
        y: Target sequences
        batch_size: Batch size
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
    """
    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # Split data
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(-1)  # Add feature dimension
    
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).unsqueeze(-1)
    
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(-1)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(
        f"Data loaders created: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )
    
    return train_loader, val_loader, test_loader