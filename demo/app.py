"""Streamlit demo application for EEG signal processing."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model
from src.data import EEGDataset
from src.utils import get_device, normalize_eeg_signal, apply_bandpass_filter


# Page configuration
st.set_page_config(
    page_title="EEG Signal Processing Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disclaimer banner
st.error("""
**DISCLAIMER: This is a research demonstration for educational purposes only.**
- NOT for clinical diagnosis or medical advice
- NOT FDA approved or validated for clinical use
- Requires clinician supervision for any medical applications
- Synthetic data only - no real patient data used
""")

# Title
st.title("🧠 EEG Signal Processing Demo")
st.markdown("""
This demo showcases EEG signal processing and epileptic seizure detection using deep learning models.
All data shown is synthetically generated for research and educational purposes.
""")

# Sidebar
st.sidebar.header("Configuration")

# Model selection
model_name = st.sidebar.selectbox(
    "Select Model",
    ["eegnet", "tcn", "transformer"],
    help="Choose the model architecture to use"
)

# Load configuration
@st.cache_resource
def load_config():
    """Load configuration file."""
    with open("configs/default.yaml", 'r') as f:
        return yaml.safe_load(f)

config = load_config()
config["model"]["name"] = model_name

# Load model
@st.cache_resource
def load_model(model_config):
    """Load the selected model."""
    device = get_device()
    model = create_model(model_config)
    
    # Try to load trained weights
    model_path = Path("outputs/best_model.pt")
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
    else:
        st.warning("No trained model found. Using randomly initialized weights.")
    
    return model, device

model, device = load_model(config["model"])

# Generate synthetic EEG data
@st.cache_data
def generate_sample_data(num_samples=10):
    """Generate sample EEG data for demonstration."""
    dataset = EEGDataset(
        num_samples=num_samples,
        channels=config["data"]["channels"],
        seq_len=config["data"]["seq_len"],
        sampling_rate=config["data"]["sampling_rate"],
        normalize=True,
        apply_filter=True
    )
    return dataset

sample_dataset = generate_sample_data()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("EEG Signal Visualization")
    
    # Select sample
    sample_idx = st.selectbox(
        "Select Sample",
        range(len(sample_dataset)),
        format_func=lambda x: f"Sample {x+1}"
    )
    
    # Get sample data
    signal, label = sample_dataset[sample_idx]
    signal_np = signal.numpy()
    
    # Display signal
    fig = make_subplots(
        rows=config["data"]["channels"], cols=1,
        subplot_titles=[f"Channel {i+1}" for i in range(config["data"]["channels"])],
        vertical_spacing=0.02
    )
    
    time_axis = np.linspace(0, config["data"]["seq_len"]/config["data"]["sampling_rate"], config["data"]["seq_len"])
    
    for i in range(config["data"]["channels"]):
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_np[i],
                mode='lines',
                name=f'Channel {i+1}',
                line=dict(width=1)
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        height=800,
        title_text="EEG Signal Channels",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Amplitude (μV)")
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("Model Prediction")
    
    # Make prediction
    with torch.no_grad():
        signal_input = signal.unsqueeze(0).to(device)
        output = model(signal_input)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).item()
    
    # Display results
    st.subheader("Classification Results")
    
    # True label
    true_label = "Epileptic" if label == 1 else "Normal"
    st.write(f"**True Label:** {true_label}")
    
    # Predicted label
    pred_label = "Epileptic" if pred == 1 else "Normal"
    st.write(f"**Predicted Label:** {pred_label}")
    
    # Confidence
    confidence = probs[0][pred].item()
    st.write(f"**Confidence:** {confidence:.3f}")
    
    # Probability distribution
    st.subheader("Probability Distribution")
    
    prob_data = {
        'Class': ['Normal', 'Epileptic'],
        'Probability': [probs[0][0].item(), probs[0][1].item()]
    }
    
    fig_prob = px.bar(
        prob_data,
        x='Class',
        y='Probability',
        color='Class',
        color_discrete_map={'Normal': 'blue', 'Epileptic': 'red'}
    )
    fig_prob.update_layout(height=300)
    st.plotly_chart(fig_prob, use_container_width=True)
    
    # Model info
    st.subheader("Model Information")
    st.write(f"**Model:** {model_name.upper()}")
    st.write(f"**Channels:** {config['data']['channels']}")
    st.write(f"**Sequence Length:** {config['data']['seq_len']}")
    st.write(f"**Sampling Rate:** {config['data']['sampling_rate']} Hz")

# Additional analysis
st.header("Signal Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Frequency Analysis")
    
    # Compute FFT
    fft_data = np.fft.fft(signal_np, axis=1)
    freqs = np.fft.fftfreq(config["data"]["seq_len"], 1/config["data"]["sampling_rate"])
    
    # Plot power spectral density for first few channels
    fig_psd = make_subplots(
        rows=min(4, config["data"]["channels"]), cols=1,
        subplot_titles=[f"Channel {i+1} PSD" for i in range(min(4, config["data"]["channels"]))],
        vertical_spacing=0.05
    )
    
    for i in range(min(4, config["data"]["channels"])):
        psd = np.abs(fft_data[i])**2
        # Only plot positive frequencies
        pos_freqs = freqs[:len(freqs)//2]
        pos_psd = psd[:len(psd)//2]
        
        fig_psd.add_trace(
            go.Scatter(
                x=pos_freqs,
                y=pos_psd,
                mode='lines',
                name=f'Channel {i+1}',
                line=dict(width=1)
            ),
            row=i+1, col=1
        )
    
    fig_psd.update_layout(
        height=600,
        title_text="Power Spectral Density",
        showlegend=False
    )
    
    fig_psd.update_xaxes(title_text="Frequency (Hz)")
    fig_psd.update_yaxes(title_text="Power")
    
    st.plotly_chart(fig_psd, use_container_width=True)

with col2:
    st.subheader("Statistical Analysis")
    
    # Compute statistics
    stats_data = {
        'Channel': [f'Ch {i+1}' for i in range(config["data"]["channels"])],
        'Mean': [np.mean(signal_np[i]) for i in range(config["data"]["channels"])],
        'Std': [np.std(signal_np[i]) for i in range(config["data"]["channels"])],
        'Min': [np.min(signal_np[i]) for i in range(config["data"]["channels"])],
        'Max': [np.max(signal_np[i]) for i in range(config["data"]["channels"])]
    }
    
    st.dataframe(stats_data, use_container_width=True)
    
    # Channel correlation heatmap
    st.subheader("Channel Correlation")
    
    correlation_matrix = np.corrcoef(signal_np)
    
    fig_corr = px.imshow(
        correlation_matrix,
        labels=dict(x="Channel", y="Channel", color="Correlation"),
        x=[f"Ch {i+1}" for i in range(config["data"]["channels"])],
        y=[f"Ch {i+1}" for i in range(config["data"]["channels"])],
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Note:** This demonstration uses synthetically generated EEG data for research and educational purposes only.
For clinical applications, proper validation, regulatory approval, and clinical supervision are required.
""")
