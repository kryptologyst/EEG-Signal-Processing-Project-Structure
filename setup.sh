#!/bin/bash

# EEG Signal Processing Project - Quick Start Script

echo "🧠 EEG Signal Processing Project - Quick Start"
echo "=============================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python version $python_version is too old. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p outputs logs evaluation_results assets data

# Set up pre-commit hooks (optional)
if command -v pre-commit &> /dev/null; then
    echo "🔧 Setting up pre-commit hooks..."
    pre-commit install
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Train a model:"
echo "   python scripts/train.py --config configs/default.yaml"
echo ""
echo "2. Evaluate the model:"
echo "   python scripts/eval.py --config configs/default.yaml"
echo ""
echo "3. Launch the demo:"
echo "   streamlit run demo/app.py"
echo ""
echo "4. Run tests:"
echo "   python -m pytest tests/"
echo ""
echo "Available model configurations:"
echo "  - configs/default.yaml (EEGNet)"
echo "  - configs/tcn.yaml (Temporal Convolutional Network)"
echo "  - configs/transformer.yaml (Transformer)"
echo ""
echo "⚠️  DISCLAIMER: This is a research demonstration for educational purposes only."
echo "   NOT for clinical diagnosis or medical advice."
echo "   NOT FDA approved or validated for clinical use."
echo "   Requires clinician supervision for any medical applications."
echo "   Synthetic data only - no real patient data used."
