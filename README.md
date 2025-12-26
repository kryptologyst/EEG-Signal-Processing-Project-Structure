# EEG Signal Processing Project Structure

## DISCLAIMER
**This is a research demonstration project for educational purposes only.**
- NOT for clinical diagnosis or medical advice
- NOT FDA approved or validated for clinical use
- Requires clinician supervision for any medical applications
- Synthetic data only - no real patient data used

## Project Overview
This project implements EEG signal processing and classification using modern deep learning techniques. It focuses on epileptic seizure detection as a binary classification task.

## Quick Start

1. **Setup Environment**
```bash
pip install -r requirements.txt
```

2. **Run Training**
```bash
python scripts/train.py --config configs/default.yaml
```

3. **Launch Demo**
```bash
streamlit run demo/app.py
```

## Dataset
- Synthetic EEG data generation with realistic characteristics
- Binary classification: Normal vs Epileptic patterns
- Configurable channels (14 default), sequence length (128 default)
- Patient-level splits to prevent data leakage

## Models
- **EEGNet**: 1D CNN baseline
- **TCN**: Temporal Convolutional Network
- **Transformer**: Attention-based architecture
- **Ensemble**: Multiple model combination

## Evaluation Metrics
- AUROC, AUPRC for classification performance
- Sensitivity, Specificity, PPV, NPV
- Calibration metrics (Brier Score, ECE)
- Per-channel analysis and attention visualization

## Repository Structure
```
src/
├── models/          # Model architectures
├── data/            # Data loading and preprocessing
├── losses/          # Loss functions
├── metrics/         # Evaluation metrics
├── utils/           # Utility functions
├── train.py         # Training script
└── eval.py          # Evaluation script

configs/             # Configuration files
scripts/             # Training and evaluation scripts
demo/                # Streamlit demo application
tests/               # Unit tests
assets/              # Generated plots and results
data/                # Data storage (synthetic)
```

## Configuration
All hyperparameters and settings are managed through YAML configuration files in `configs/`. See `configs/default.yaml` for the default setup.

## Safety and Privacy
- No real patient data used
- Synthetic data generation only
- De-identification hooks available for future real data integration
- Clear disclaimers in all outputs and documentation
# EEG-Signal-Processing-Project-Structure
