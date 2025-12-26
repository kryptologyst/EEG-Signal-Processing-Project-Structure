"""Utility functions for EEG signal processing project."""

import random
import numpy as np
import torch
from typing import Optional, List, Tuple, Dict, Any
import logging
from pathlib import Path


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(fallback_order: List[str] = None) -> torch.device:
    """Get the best available device with fallback.
    
    Args:
        fallback_order: List of device types to try in order
        
    Returns:
        Available torch device
    """
    if fallback_order is None:
        fallback_order = ["cuda", "mps", "cpu"]
    
    for device_type in fallback_order:
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_type == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif device_type == "cpu":
            return torch.device("cpu")
    
    return torch.device("cpu")


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory to save log files
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "eeg_processing.log"),
            logging.StreamHandler()
        ]
    )


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: str,
    is_best: bool = False
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        metrics: Evaluation metrics
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'is_best': is_best
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        
    Returns:
        Checkpoint dictionary
    """
    return torch.load(filepath, map_location='cpu')


def create_patient_splits(
    num_samples: int,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """Create patient-level splits to prevent data leakage.
    
    Args:
        num_samples: Total number of samples
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        seed: Random seed
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    
    train_end = int(num_samples * train_split)
    val_end = int(num_samples * (train_split + val_split))
    
    train_indices = indices[:train_end].tolist()
    val_indices = indices[train_end:val_end].tolist()
    test_indices = indices[val_end:].tolist()
    
    return train_indices, val_indices, test_indices


def normalize_eeg_signal(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize EEG signal.
    
    Args:
        signal: EEG signal array (channels, time)
        method: Normalization method ('zscore', 'minmax', 'robust')
        
    Returns:
        Normalized signal
    """
    if method == "zscore":
        return (signal - np.mean(signal, axis=1, keepdims=True)) / np.std(signal, axis=1, keepdims=True)
    elif method == "minmax":
        min_vals = np.min(signal, axis=1, keepdims=True)
        max_vals = np.max(signal, axis=1, keepdims=True)
        return (signal - min_vals) / (max_vals - min_vals + 1e-8)
    elif method == "robust":
        median_vals = np.median(signal, axis=1, keepdims=True)
        mad_vals = np.median(np.abs(signal - median_vals), axis=1, keepdims=True)
        return (signal - median_vals) / (mad_vals + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def apply_bandpass_filter(
    signal: np.ndarray,
    sampling_rate: float,
    low_freq: float = 1.0,
    high_freq: float = 50.0
) -> np.ndarray:
    """Apply bandpass filter to EEG signal.
    
    Args:
        signal: EEG signal array (channels, time)
        sampling_rate: Sampling rate in Hz
        low_freq: Low cutoff frequency
        high_freq: High cutoff frequency
        
    Returns:
        Filtered signal
    """
    from scipy import signal as sp_signal
    
    # Design bandpass filter
    nyquist = sampling_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    b, a = sp_signal.butter(4, [low, high], btype='band')
    
    # Apply filter to each channel
    filtered_signal = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        filtered_signal[i] = sp_signal.filtfilt(b, a, signal[i])
    
    return filtered_signal
