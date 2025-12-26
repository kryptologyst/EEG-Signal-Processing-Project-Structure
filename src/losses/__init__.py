"""Loss functions for EEG signal processing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    Focal Loss = -alpha * (1-p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for each class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss for class imbalance."""
    
    def __init__(self, class_weights: Optional[List[float]] = None):
        """Initialize weighted cross entropy loss.
        
        Args:
            class_weights: Weights for each class
        """
        super().__init__()
        self.class_weights = class_weights
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross entropy loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Weighted cross entropy loss value
        """
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, device=inputs.device)
            return F.cross_entropy(inputs, targets, weight=weights)
        else:
            return F.cross_entropy(inputs, targets)


def create_loss_function(config: dict) -> nn.Module:
    """Create loss function based on configuration.
    
    Args:
        config: Loss configuration
        
    Returns:
        Loss function
    """
    loss_name = config["name"].lower()
    
    if loss_name == "cross_entropy":
        class_weights = config.get("class_weights")
        return WeightedCrossEntropyLoss(class_weights)
    
    elif loss_name == "focal":
        alpha = config.get("focal_alpha")
        gamma = config.get("focal_gamma", 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_name == "focal_alpha":
        alpha = config.get("focal_alpha", 0.25)
        gamma = config.get("focal_gamma", 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
