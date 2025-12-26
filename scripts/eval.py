"""Evaluation script for EEG signal processing models."""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt

from src.models import create_model
from src.data import create_data_loaders
from src.metrics import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_calibration_curve
)
from src.utils import set_seed, get_device, setup_logging, create_patient_splits, load_checkpoint


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate EEG signal processing model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
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
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config, train_indices, val_indices, test_indices
    )
    
    # Create model
    model = create_model(config["model"]).to(device)
    
    # Load checkpoint
    if Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.warning(f"Checkpoint not found: {args.checkpoint}")
    
    # Evaluate on all splits
    splits = {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader
    }
    
    all_results = {}
    
    for split_name, loader in splits.items():
        logger.info(f"Evaluating on {split_name} set...")
        
        # Evaluate model
        metrics = evaluate_model(model, loader, device, config["evaluation"]["metrics"])
        all_results[split_name] = metrics
        
        # Log metrics
        logger.info(f"{split_name.capitalize()} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Generate plots for test set
        if split_name == "test":
            logger.info("Generating evaluation plots...")
            
            # Get predictions for plotting
            model.eval()
            all_preds = []
            all_probs = []
            all_labels = []
            
            with torch.no_grad():
                for signals, labels in loader:
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
            plot_confusion_matrix(
                all_labels, all_preds,
                save_path=output_dir / "confusion_matrix.png"
            )
            plot_roc_curve(
                all_labels, all_probs,
                save_path=output_dir / "roc_curve.png"
            )
            plot_precision_recall_curve(
                all_labels, all_probs,
                save_path=output_dir / "pr_curve.png"
            )
            plot_calibration_curve(
                all_labels, all_probs,
                save_path=output_dir / "calibration_curve.png"
            )
    
    # Save results summary
    import json
    with open(output_dir / "results_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    
    # Create summary table
    metrics_to_show = ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "brier_score", "ece"]
    
    print(f"{'Metric':<15}", end="")
    for split in splits.keys():
        print(f"{split.capitalize():<12}", end="")
    print()
    
    print("-" * (15 + 12 * len(splits)))
    
    for metric in metrics_to_show:
        print(f"{metric:<15}", end="")
        for split in splits.keys():
            value = all_results[split].get(metric, 0.0)
            print(f"{value:<12.4f}", end="")
        print()
    
    logger.info(f"\nEvaluation completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
