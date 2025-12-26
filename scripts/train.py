"""Training script for EEG signal processing models."""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np

from src.models import create_model
from src.data import create_data_loaders
from src.losses import create_loss_function
from src.metrics import evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, plot_calibration_curve
from src.utils import set_seed, get_device, setup_logging, save_checkpoint, create_patient_splits


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int
) -> float:
    """Train model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to run on
        epoch: Current epoch number
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for signals, labels in pbar:
        signals = signals.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(signals)
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int
) -> tuple[float, dict]:
    """Validate model for one epoch.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average validation loss, metrics dictionary)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            loss = loss_fn(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    
    # Compute metrics
    from src.metrics import compute_classification_metrics, compute_calibration_metrics
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    classification_metrics = compute_classification_metrics(all_labels, all_preds, all_probs)
    calibration_metrics = compute_calibration_metrics(all_labels, all_probs)
    
    metrics = {**classification_metrics, **calibration_metrics}
    
    return avg_loss, metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train EEG signal processing model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    setup_logging(config["logging"]["level"], config["logging"]["log_dir"])
    logger = logging.getLogger(__name__)
    
    # Set seed for reproducibility
    set_seed(config["seed"], config["deterministic"])
    
    # Get device
    device = get_device(config["device"]["fallback_order"])
    logger.info(f"Using device: {device}")
    
    # Create data splits
    train_indices, val_indices, test_indices = create_patient_splits(
        config["data"]["num_samples"],
        config["data"]["train_split"],
        config["data"]["val_split"],
        config["data"]["test_split"],
        config["seed"]
    )
    
    logger.info(f"Data splits - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config, train_indices, val_indices, test_indices
    )
    
    # Create model
    model = create_model(config["model"]).to(device)
    logger.info(f"Created model: {config['model']['name']}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create loss function
    loss_fn = create_loss_function(config["loss"])
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Create scheduler
    if config["training"]["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["num_epochs"]
        )
    elif config["training"]["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config["training"]["num_epochs"]//3, gamma=0.1
        )
    elif config["training"]["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
    else:
        scheduler = None
    
    # Training loop
    best_auroc = 0.0
    patience_counter = 0
    
    logger.info("Starting training...")
    
    for epoch in range(1, config["training"]["num_epochs"] + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, loss_fn, device, epoch)
        
        # Update scheduler
        if scheduler is not None:
            if config["training"]["scheduler"] == "plateau":
                scheduler.step(val_metrics["auroc"])
            else:
                scheduler.step()
        
        # Log metrics
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Val Metrics: AUROC: {val_metrics['auroc']:.4f}, AUPRC: {val_metrics['auprc']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics["auroc"] > best_auroc
        if is_best:
            best_auroc = val_metrics["auroc"]
            patience_counter = 0
        else:
            patience_counter += 1
        
        save_checkpoint(
            model, optimizer, epoch, val_loss, val_metrics,
            output_dir / f"checkpoint_epoch_{epoch}.pt", is_best
        )
        
        if is_best:
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        
        # Early stopping
        if patience_counter >= config["training"]["early_stopping_patience"]:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, config["evaluation"]["metrics"])
    
    logger.info("Test Metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save plots
    logger.info("Generating evaluation plots...")
    
    # Get test predictions for plotting
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Generate plots
    plot_confusion_matrix(all_labels, all_preds, save_path=output_dir / "confusion_matrix.png")
    plot_roc_curve(all_labels, all_probs, save_path=output_dir / "roc_curve.png")
    plot_precision_recall_curve(all_labels, all_probs, save_path=output_dir / "pr_curve.png")
    plot_calibration_curve(all_labels, all_probs, save_path=output_dir / "calibration_curve.png")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
