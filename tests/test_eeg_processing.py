"""Test suite for EEG signal processing project."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import EEGNet, TCN, EEGTransformer, create_model
from data import EEGDataset, create_data_loaders
from losses import FocalLoss, WeightedCrossEntropyLoss, create_loss_function
from metrics import compute_classification_metrics, compute_calibration_metrics
from utils import set_seed, get_device, normalize_eeg_signal, apply_bandpass_filter


class TestModels:
    """Test model architectures."""
    
    def test_eegnet_forward(self):
        """Test EEGNet forward pass."""
        model = EEGNet(channels=14, seq_len=128, num_classes=2)
        x = torch.randn(2, 14, 128)
        output = model(x)
        assert output.shape == (2, 2)
    
    def test_tcn_forward(self):
        """Test TCN forward pass."""
        model = TCN(channels=14, seq_len=128, num_classes=2)
        x = torch.randn(2, 14, 128)
        output = model(x)
        assert output.shape == (2, 2)
    
    def test_transformer_forward(self):
        """Test EEG Transformer forward pass."""
        model = EEGTransformer(channels=14, seq_len=128, num_classes=2)
        x = torch.randn(2, 14, 128)
        output = model(x)
        assert output.shape == (2, 2)
    
    def test_create_model(self):
        """Test model creation from config."""
        config = {
            "name": "eegnet",
            "channels": 14,
            "seq_len": 128,
            "num_classes": 2,
            "dropout": 0.3
        }
        model = create_model(config)
        assert isinstance(model, EEGNet)


class TestData:
    """Test data loading and preprocessing."""
    
    def test_eeg_dataset(self):
        """Test EEG dataset creation."""
        dataset = EEGDataset(num_samples=10, channels=14, seq_len=128)
        assert len(dataset) == 10
        
        signal, label = dataset[0]
        assert signal.shape == (14, 128)
        assert label in [0, 1]
    
    def test_data_loaders(self):
        """Test data loader creation."""
        config = {
            "data": {
                "num_samples": 100,
                "channels": 14,
                "seq_len": 128,
                "sampling_rate": 256.0,
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15
            },
            "training": {"batch_size": 16},
            "seed": 42
        }
        
        train_indices = list(range(70))
        val_indices = list(range(70, 85))
        test_indices = list(range(85, 100))
        
        train_loader, val_loader, test_loader = create_data_loaders(
            config, train_indices, val_indices, test_indices
        )
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
        
        # Test batch
        batch = next(iter(train_loader))
        signals, labels = batch
        assert signals.shape[0] <= 16  # batch size
        assert signals.shape[1:] == (14, 128)


class TestLosses:
    """Test loss functions."""
    
    def test_focal_loss(self):
        """Test Focal Loss."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Create dummy data
        inputs = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))
        
        loss = loss_fn(inputs, targets)
        assert loss.item() > 0
    
    def test_weighted_cross_entropy(self):
        """Test Weighted Cross Entropy Loss."""
        loss_fn = WeightedCrossEntropyLoss(class_weights=[1.0, 2.0])
        
        inputs = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))
        
        loss = loss_fn(inputs, targets)
        assert loss.item() > 0
    
    def test_create_loss_function(self):
        """Test loss function creation."""
        config = {"name": "cross_entropy", "class_weights": None}
        loss_fn = create_loss_function(config)
        assert isinstance(loss_fn, WeightedCrossEntropyLoss)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_classification_metrics(self):
        """Test classification metrics computation."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.3, 0.8])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        
        assert "accuracy" in metrics
        assert "auroc" in metrics
        assert "auprc" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["auroc"] <= 1
    
    def test_calibration_metrics(self):
        """Test calibration metrics computation."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.3, 0.8])
        
        metrics = compute_calibration_metrics(y_true, y_prob)
        
        assert "brier_score" in metrics
        assert "ece" in metrics
        assert 0 <= metrics["brier_score"] <= 1
        assert 0 <= metrics["ece"] <= 1


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test numpy
        np.random.seed(42)
        val1 = np.random.random()
        
        set_seed(42)
        np.random.seed(42)
        val2 = np.random.random()
        
        assert val1 == val2
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_normalize_eeg_signal(self):
        """Test EEG signal normalization."""
        signal = np.random.randn(14, 128)
        
        normalized = normalize_eeg_signal(signal, method="zscore")
        assert normalized.shape == signal.shape
        
        # Check that mean is approximately 0
        assert np.allclose(np.mean(normalized, axis=1), 0, atol=1e-10)
    
    def test_apply_bandpass_filter(self):
        """Test bandpass filtering."""
        signal = np.random.randn(14, 128)
        
        filtered = apply_bandpass_filter(signal, sampling_rate=256.0)
        assert filtered.shape == signal.shape


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training loop."""
        # Create small dataset
        dataset = EEGDataset(num_samples=20, channels=14, seq_len=128)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Create model
        model = EEGNet(channels=14, seq_len=128, num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Train for a few steps
        model.train()
        for i, (signals, labels) in enumerate(loader):
            if i >= 2:  # Only test a few steps
                break
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            assert loss.item() > 0
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        from metrics import evaluate_model
        
        # Create model and data
        model = EEGNet(channels=14, seq_len=128, num_classes=2)
        dataset = EEGDataset(num_samples=20, channels=14, seq_len=128)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        
        device = get_device()
        model = model.to(device)
        
        # Evaluate
        metrics = evaluate_model(model, loader, device)
        
        assert "accuracy" in metrics
        assert "auroc" in metrics
        assert "auprc" in metrics


if __name__ == "__main__":
    pytest.main([__file__])
