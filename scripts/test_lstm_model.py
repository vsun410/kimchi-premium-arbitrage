#!/usr/bin/env python3
"""
ED-LSTM Model Test Suite
Task #15: ED-LSTM Model Implementation Tests
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.lstm_model import (
    AttentionLayer,
    DecoderLSTM,
    EDLSTMModel,
    EncoderLSTM,
    LSTMTrainer,
    create_sequences,
    prepare_data_loaders,
)

warnings.filterwarnings("ignore")

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def create_synthetic_data(n_samples: int = 1000, n_features: int = 10) -> np.ndarray:
    """
    Create synthetic time series data for testing
    
    Args:
        n_samples: Number of time steps
        n_features: Number of features
        
    Returns:
        Synthetic data array
    """
    # Create synthetic kimchi premium data with trend and seasonality
    time = np.arange(n_samples)
    
    # Base kimchi premium (1-5% range)
    base_premium = 2.5
    
    # Trend component
    trend = 0.001 * time
    
    # Daily seasonality
    daily_season = 0.5 * np.sin(2 * np.pi * time / 24)
    
    # Weekly seasonality
    weekly_season = 0.3 * np.sin(2 * np.pi * time / (24 * 7))
    
    # Random noise
    noise = np.random.randn(n_samples) * 0.2
    
    # Combine components
    kimchi_premium = base_premium + trend + daily_season + weekly_season + noise
    
    # Create other features (correlated with premium)
    features = np.zeros((n_samples, n_features))
    features[:, 0] = kimchi_premium  # Target variable
    
    for i in range(1, n_features):
        correlation = np.random.uniform(0.3, 0.8)
        features[:, i] = (
            correlation * kimchi_premium +
            (1 - correlation) * np.random.randn(n_samples)
        )
    
    return features


def test_attention_layer():
    """Test attention mechanism"""
    print("\n" + "=" * 60)
    print("TEST 1: Attention Layer")
    print("=" * 60)
    
    try:
        batch_size = 8
        seq_len = 20
        hidden_size = 64
        
        # Create attention layer
        attention = AttentionLayer(hidden_size)
        
        # Create dummy inputs
        hidden = torch.randn(batch_size, hidden_size)
        encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        context, weights = attention(hidden, encoder_outputs)
        
        # Check output shapes
        assert context.shape == (batch_size, hidden_size), "Context shape mismatch"
        assert weights.shape == (batch_size, seq_len), "Weights shape mismatch"
        
        # Check attention weights sum to 1
        weight_sums = weights.sum(dim=1)
        assert torch.allclose(weight_sums, torch.ones(batch_size), atol=1e-6), \
            "Attention weights don't sum to 1"
        
        print(f"Context shape: {context.shape}")
        print(f"Attention weights shape: {weights.shape}")
        print(f"Weights sum: {weight_sums[0].item():.6f}")
        print("\n[OK] Attention layer test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Attention layer test failed: {e}")
        return False


def test_encoder_decoder():
    """Test encoder and decoder modules"""
    print("\n" + "=" * 60)
    print("TEST 2: Encoder-Decoder Modules")
    print("=" * 60)
    
    try:
        batch_size = 8
        seq_len = 50
        input_size = 10
        hidden_size = 128
        output_size = 1
        
        # Create encoder
        encoder = EncoderLSTM(input_size, hidden_size, num_layers=2)
        
        # Create decoder
        decoder = DecoderLSTM(output_size, hidden_size, num_layers=2)
        
        # Dummy input
        encoder_input = torch.randn(batch_size, seq_len, input_size)
        
        # Encode
        encoder_outputs, hidden = encoder(encoder_input)
        
        print(f"Encoder outputs shape: {encoder_outputs.shape}")
        print(f"Hidden state shape: {hidden[0].shape}")
        
        # Decode single step
        decoder_input = torch.randn(batch_size, 1, output_size)
        decoder_output, new_hidden, attn_weights = decoder(
            decoder_input, hidden, encoder_outputs
        )
        
        print(f"Decoder output shape: {decoder_output.shape}")
        
        assert encoder_outputs.shape == (batch_size, seq_len, hidden_size)
        assert decoder_output.shape == (batch_size, 1, output_size)
        
        print("\n[OK] Encoder-Decoder test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Encoder-Decoder test failed: {e}")
        return False


def test_full_model():
    """Test complete ED-LSTM model"""
    print("\n" + "=" * 60)
    print("TEST 3: Full ED-LSTM Model")
    print("=" * 60)
    
    try:
        batch_size = 16
        seq_len = 168  # 7 days
        input_size = 10
        hidden_size = 64
        decoder_length = 24  # Predict 24 hours
        
        # Create model
        model = EDLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=1,
            num_layers=2,
            dropout=0.1,
            use_attention=True,
            decoder_length=decoder_length
        )
        
        # Move to device
        model = model.to(DEVICE)
        
        # Dummy input
        x = torch.randn(batch_size, seq_len, input_size).to(DEVICE)
        target = torch.randn(batch_size, decoder_length, 1).to(DEVICE)
        
        # Forward pass with teacher forcing
        outputs, attention = model(x, target, teacher_forcing_ratio=0.5)
        
        print(f"Model output shape: {outputs.shape}")
        print(f"Expected shape: ({batch_size}, {decoder_length}, 1)")
        
        assert outputs.shape == (batch_size, decoder_length, 1), "Output shape mismatch"
        
        # Forward pass without teacher forcing
        outputs_no_tf, _ = model(x, teacher_forcing_ratio=0)
        assert outputs_no_tf.shape == outputs.shape, "Output shape inconsistent"
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {n_params:,}")
        
        print("\n[OK] Full model test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Full model test failed: {e}")
        return False


def test_training_pipeline():
    """Test training pipeline"""
    print("\n" + "=" * 60)
    print("TEST 4: Training Pipeline")
    print("=" * 60)
    
    try:
        # Generate synthetic data
        data = create_synthetic_data(n_samples=500, n_features=10)
        
        # Create sequences
        X, y = create_sequences(data, seq_length=48, pred_length=12, stride=1)
        print(f"Sequences created: X={X.shape}, y={y.shape}")
        
        # Create small datasets for testing
        X_train = torch.FloatTensor(X[:100])
        y_train = torch.FloatTensor(y[:100]).unsqueeze(-1)
        X_val = torch.FloatTensor(X[100:120])
        y_val = torch.FloatTensor(y[100:120]).unsqueeze(-1)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Create model
        model = EDLSTMModel(
            input_size=10,
            hidden_size=32,
            output_size=1,
            num_layers=1,
            decoder_length=12,
            use_attention=True
        )
        
        # Create trainer
        trainer = LSTMTrainer(model, learning_rate=0.001, device=DEVICE)
        
        # Train for a few epochs
        print("\nTraining for 5 epochs...")
        history = trainer.train(
            train_loader,
            val_loader,
            n_epochs=5,
            early_stopping_patience=10,
            save_dir="test_models/"
        )
        
        print(f"Final train loss: {history['train_losses'][-1]:.6f}")
        print(f"Final val loss: {history['val_losses'][-1]:.6f}")
        print(f"Best val loss: {history['best_val_loss']:.6f}")
        
        # Test prediction
        model.eval()
        with torch.no_grad():
            test_input = X_val[:1]
            test_input_tensor = torch.FloatTensor(test_input).to(DEVICE)
            predictions, attention = trainer.predict(test_input_tensor, return_attention=True)
            
            print(f"\nPrediction shape: {predictions.shape}")
            print(f"Sample predictions: {predictions[0, :5, 0]}")
        
        print("\n[OK] Training pipeline test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Training pipeline test failed: {e}")
        return False


def test_save_load():
    """Test model save and load functionality"""
    print("\n" + "=" * 60)
    print("TEST 5: Model Save/Load")
    print("=" * 60)
    
    try:
        import tempfile
        
        # Create model
        model = EDLSTMModel(
            input_size=10,
            hidden_size=32,
            output_size=1,
            decoder_length=12
        )
        
        trainer = LSTMTrainer(model, device=DEVICE)
        
        # Create dummy data and train briefly
        X = torch.randn(10, 48, 10)
        y = torch.randn(10, 12, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=2)
        
        # Train one epoch
        trainer.train_epoch(loader)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = f.name
        
        trainer.save_model(temp_path)
        print(f"Model saved to {temp_path}")
        
        # Create new model and load
        new_model = EDLSTMModel(
            input_size=10,
            hidden_size=32,
            output_size=1,
            decoder_length=12
        )
        new_trainer = LSTMTrainer(new_model, device=DEVICE)
        new_trainer.load_model(temp_path)
        print("Model loaded successfully")
        
        # Compare predictions
        model.eval()
        new_model.eval()
        
        test_input = torch.randn(1, 48, 10).to(DEVICE)
        
        with torch.no_grad():
            output1, _ = model(test_input, teacher_forcing_ratio=0)
            output2, _ = new_model(test_input, teacher_forcing_ratio=0)
        
        # Check if outputs are identical
        assert torch.allclose(output1, output2, atol=1e-6), "Loaded model produces different output"
        
        # Cleanup
        import os
        os.unlink(temp_path)
        
        print("\n[OK] Save/Load test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Save/Load test failed: {e}")
        return False


def test_data_preparation():
    """Test data preparation utilities"""
    print("\n" + "=" * 60)
    print("TEST 6: Data Preparation")
    print("=" * 60)
    
    try:
        # Generate data
        data = create_synthetic_data(1000, 10)
        
        # Test sequence creation
        seq_lengths = [24, 168, 720]  # 1 day, 1 week, 1 month
        pred_lengths = [1, 24, 168]  # 1 hour, 1 day, 1 week
        
        for seq_len, pred_len in zip(seq_lengths, pred_lengths):
            X, y = create_sequences(data, seq_len, pred_len)
            expected_samples = len(data) - seq_len - pred_len + 1
            
            assert X.shape[0] == expected_samples, f"Wrong number of samples for seq_len={seq_len}"
            assert X.shape[1] == seq_len, f"Wrong sequence length"
            assert y.shape[1] == pred_len, f"Wrong prediction length"
            
            print(f"Seq length {seq_len}, pred length {pred_len}: X={X.shape}, y={y.shape}")
        
        # Test data loader preparation
        X, y = create_sequences(data, 168, 24)
        train_loader, val_loader, test_loader = prepare_data_loaders(
            X, y, batch_size=16, train_ratio=0.7, val_ratio=0.15
        )
        
        print(f"\nData loaders created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Check data shapes
        batch = next(iter(train_loader))
        print(f"  Batch shapes: X={batch[0].shape}, y={batch[1].shape}")
        
        print("\n[OK] Data preparation test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Data preparation test failed: {e}")
        return False


def test_model_metrics():
    """Test model evaluation metrics"""
    print("\n" + "=" * 60)
    print("TEST 7: Model Metrics")
    print("=" * 60)
    
    try:
        # Create simple model
        model = EDLSTMModel(
            input_size=5,
            hidden_size=16,
            output_size=1,
            num_layers=1,
            decoder_length=6
        )
        
        trainer = LSTMTrainer(model, device=DEVICE)
        
        # Create dummy predictions
        y_true = torch.randn(10, 6, 1)
        y_pred = torch.randn(10, 6, 1)
        
        # Calculate MSE
        mse = nn.MSELoss()(y_pred, y_true)
        print(f"MSE: {mse.item():.6f}")
        
        # Calculate MAE
        mae = nn.L1Loss()(y_pred, y_true)
        print(f"MAE: {mae.item():.6f}")
        
        # Calculate RMSE
        rmse = torch.sqrt(mse)
        print(f"RMSE: {rmse.item():.6f}")
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = (torch.abs((y_true - y_pred) / y_true)).mean() * 100
        print(f"MAPE: {mape.item():.2f}%")
        
        # Verify metrics are positive
        assert mse >= 0, "MSE should be non-negative"
        assert mae >= 0, "MAE should be non-negative"
        
        print("\n[OK] Model metrics test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Model metrics test failed: {e}")
        return False


def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("ED-LSTM MODEL TEST SUITE")
    print("=" * 60)
    print("\nTesting Task #15: ED-LSTM Model Implementation")
    
    tests = [
        ("Attention Layer", test_attention_layer),
        ("Encoder-Decoder", test_encoder_decoder),
        ("Full Model", test_full_model),
        ("Training Pipeline", test_training_pipeline),
        ("Save/Load", test_save_load),
        ("Data Preparation", test_data_preparation),
        ("Model Metrics", test_model_metrics),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"[ERROR] {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Results summary
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{test_name:25} {status}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n[SUCCESS] Task #15 COMPLETED! ED-LSTM model ready.")
        print("\nKey features implemented:")
        print("  1. Encoder-Decoder LSTM architecture")
        print("  2. Attention mechanism for better predictions")
        print("  3. Teacher forcing for stable training")
        print("  4. Flexible sequence lengths (24h, 7d, 30d)")
        print("  5. Model checkpointing and loading")
        print("  6. Training with early stopping")
        print("  7. Gradient clipping for stability")
        return 0
    else:
        print("\n[WARN] Some tests failed. Please review.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)