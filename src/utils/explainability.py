"""Explainability and uncertainty quantification for EEG models."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import seaborn as sns


class GradCAM:
    """Gradient-weighted Class Activation Mapping for EEG models."""
    
    def __init__(self, model: nn.Module, target_layer: str):
        """Initialize GradCAM.
        
        Args:
            model: PyTorch model
            target_layer: Name of the target layer for GradCAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find the target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generate GradCAM for the given input and class.
        
        Args:
            input_tensor: Input EEG signal
            class_idx: Class index for which to generate CAM
            
        Returns:
            GradCAM heatmap
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Compute GradCAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=1)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:])
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = torch.relu(cam)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()


def compute_attention_weights(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """Compute attention weights for transformer models.
    
    Args:
        model: Transformer model
        input_tensor: Input EEG signal
        
    Returns:
        Attention weights
    """
    if not hasattr(model, 'transformer'):
        raise ValueError("Model does not have transformer attribute")
    
    # Get attention weights from transformer
    with torch.no_grad():
        # Reshape input for transformer
        x = input_tensor.transpose(1, 2)  # (batch, seq_len, channels)
        x = model.input_projection(x)
        x = model.pos_encoding(x)
        
        # Get attention weights from first layer
        encoder_layer = model.transformer.layers[0]
        attn_output, attn_weights = encoder_layer.self_attn(
            x, x, x, need_weights=True
        )
        
        # Average across heads
        attn_weights = attn_weights.mean(dim=1)  # (batch, seq_len, seq_len)
        
        return attn_weights[0].cpu().numpy()  # Remove batch dimension


def monte_carlo_dropout(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_samples: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Monte Carlo dropout for uncertainty estimation.
    
    Args:
        model: PyTorch model with dropout layers
        input_tensor: Input EEG signal
        num_samples: Number of Monte Carlo samples
        
    Returns:
        Tuple of (mean predictions, uncertainty estimates)
    """
    model.train()  # Enable dropout
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            predictions.append(probs.cpu().numpy())
    
    predictions = np.array(predictions)  # (num_samples, batch_size, num_classes)
    
    # Compute mean and uncertainty
    mean_pred = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)
    
    return mean_pred, uncertainty


def plot_gradcam(
    signal: np.ndarray,
    cam: np.ndarray,
    channels: List[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot GradCAM visualization.
    
    Args:
        signal: Original EEG signal (channels, time)
        cam: GradCAM heatmap
        channels: Channel names
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    if channels is None:
        channels = [f"Ch {i+1}" for i in range(signal.shape[0])]
    
    fig, axes = plt.subplots(signal.shape[0], 1, figsize=(12, 2*signal.shape[0]))
    if signal.shape[0] == 1:
        axes = [axes]
    
    time_axis = np.arange(signal.shape[1])
    
    for i in range(signal.shape[0]):
        ax = axes[i]
        
        # Plot original signal
        ax.plot(time_axis, signal[i], 'b-', alpha=0.7, label='Original Signal')
        
        # Plot GradCAM overlay
        ax.imshow(
            cam.reshape(1, -1),
            aspect='auto',
            extent=[0, signal.shape[1], signal[i].min(), signal[i].max()],
            cmap='Reds',
            alpha=0.3
        )
        
        ax.set_title(f'{channels[i]} - GradCAM')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot attention weight heatmap.
    
    Args:
        attention_weights: Attention weights matrix
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        attention_weights,
        cmap='Blues',
        square=True,
        ax=ax,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    ax.set_title('Attention Weight Heatmap')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Time Steps')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_uncertainty_analysis(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    true_labels: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot uncertainty analysis.
    
    Args:
        predictions: Mean predictions
        uncertainties: Uncertainty estimates
        true_labels: True labels
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Prediction confidence vs uncertainty
    confidences = np.max(predictions, axis=1)
    avg_uncertainties = np.mean(uncertainties, axis=1)
    
    scatter = axes[0].scatter(confidences, avg_uncertainties, c=true_labels, cmap='viridis')
    axes[0].set_xlabel('Prediction Confidence')
    axes[0].set_ylabel('Average Uncertainty')
    axes[0].set_title('Confidence vs Uncertainty')
    axes[0].colorbar(scatter, label='True Label')
    
    # Plot 2: Uncertainty distribution by class
    normal_uncertainty = avg_uncertainties[true_labels == 0]
    epileptic_uncertainty = avg_uncertainties[true_labels == 1]
    
    axes[1].hist(normal_uncertainty, alpha=0.7, label='Normal', bins=20)
    axes[1].hist(epileptic_uncertainty, alpha=0.7, label='Epileptic', bins=20)
    axes[1].set_xlabel('Average Uncertainty')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Uncertainty Distribution by Class')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_explainability_report(
    model: nn.Module,
    input_tensor: torch.Tensor,
    true_label: int,
    output_dir: str
) -> None:
    """Generate comprehensive explainability report.
    
    Args:
        model: PyTorch model
        input_tensor: Input EEG signal
        true_label: True label
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_label = torch.argmax(output, dim=1).item()
        confidence = probs[0][pred_label].item()
    
    # Generate GradCAM if applicable
    if hasattr(model, 'conv1'):  # CNN-based model
        gradcam = GradCAM(model, 'conv1')
        cam = gradcam.generate_cam(input_tensor, pred_label)
        
        signal_np = input_tensor[0].cpu().numpy()
        plot_gradcam(signal_np, cam, save_path=output_dir / "gradcam.png")
    
    # Generate attention weights if transformer
    if hasattr(model, 'transformer'):
        try:
            attention_weights = compute_attention_weights(model, input_tensor)
            plot_attention_heatmap(attention_weights, save_path=output_dir / "attention.png")
        except Exception as e:
            print(f"Could not generate attention weights: {e}")
    
    # Monte Carlo dropout uncertainty
    try:
        mean_pred, uncertainty = monte_carlo_dropout(model, input_tensor)
        print(f"Prediction: {pred_label}, Confidence: {confidence:.3f}")
        print(f"Uncertainty: {np.mean(uncertainty):.3f}")
    except Exception as e:
        print(f"Could not compute uncertainty: {e}")
    
    print(f"Explainability report saved to {output_dir}")
