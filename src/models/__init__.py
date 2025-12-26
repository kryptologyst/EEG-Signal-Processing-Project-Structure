"""Model architectures for EEG signal processing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math


class EEGNet(nn.Module):
    """EEGNet architecture for EEG signal classification.
    
    Based on the original EEGNet paper with improvements for better performance.
    """
    
    def __init__(
        self,
        channels: int = 14,
        seq_len: int = 128,
        num_classes: int = 2,
        conv1_filters: int = 32,
        conv2_filters: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.3
    ):
        """Initialize EEGNet.
        
        Args:
            channels: Number of EEG channels
            seq_len: Length of time sequence
            num_classes: Number of output classes
            conv1_filters: Number of filters in first conv layer
            conv2_filters: Number of filters in second conv layer
            kernel_size: Convolution kernel size
            pool_size: Pooling kernel size
            dropout: Dropout rate
        """
        super().__init__()
        
        self.channels = channels
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        # First convolutional block
        self.conv1 = nn.Conv1d(channels, conv1_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(conv1_filters)
        self.pool1 = nn.MaxPool1d(pool_size)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(conv2_filters)
        self.pool2 = nn.MaxPool1d(pool_size)
        self.dropout2 = nn.Dropout(dropout)
        
        # Calculate flattened size
        self.flattened_size = conv2_filters * (seq_len // (pool_size * pool_size))
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, seq_len)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x


class TemporalBlock(nn.Module):
    """Temporal block for TCN architecture."""
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        """Initialize temporal block.
        
        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            dilation: Convolution dilation
            padding: Convolution padding
            dropout: Dropout rate
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride, padding, dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride, padding, dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Chomp1d layer for TCN."""
    
    def __init__(self, chomp_size: int):
        """Initialize Chomp1d.
        
        Args:
            chomp_size: Size to chomp from the end
        """
        super().__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x[:, :, :-self.chomp_size].contiguous()


class TCN(nn.Module):
    """Temporal Convolutional Network for EEG signal processing."""
    
    def __init__(
        self,
        channels: int = 14,
        seq_len: int = 128,
        num_classes: int = 2,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        """Initialize TCN.
        
        Args:
            channels: Number of EEG channels
            seq_len: Length of time sequence
            num_classes: Number of output classes
            num_channels: List of channel sizes for each layer
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()
        
        if num_channels is None:
            num_channels = [64, 64, 64, 64]
        
        self.channels = channels
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            ]
        
        self.network = nn.Sequential(*layers)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(num_channels[-1], num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, seq_len)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.network(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(0), :]


class EEGTransformer(nn.Module):
    """Transformer-based architecture for EEG signal processing."""
    
    def __init__(
        self,
        channels: int = 14,
        seq_len: int = 128,
        num_classes: int = 2,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        """Initialize EEG Transformer.
        
        Args:
            channels: Number of EEG channels
            seq_len: Length of time sequence
            num_classes: Number of output classes
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.channels = channels
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(channels, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, seq_len)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Reshape to (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Project to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x


def create_model(config: dict) -> nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized model
    """
    model_name = config["name"].lower()
    model_config = config.get(model_name, {})
    
    if model_name == "eegnet":
        return EEGNet(
            channels=config["channels"],
            seq_len=config["seq_len"],
            num_classes=config["num_classes"],
            dropout=config.get("dropout", 0.3),
            **model_config
        )
    elif model_name == "tcn":
        return TCN(
            channels=config["channels"],
            seq_len=config["seq_len"],
            num_classes=config["num_classes"],
            **model_config
        )
    elif model_name == "transformer":
        return EEGTransformer(
            channels=config["channels"],
            seq_len=config["seq_len"],
            num_classes=config["num_classes"],
            **model_config
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
