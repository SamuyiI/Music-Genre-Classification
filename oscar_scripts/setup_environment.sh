#!/bin/bash
#
# Environment setup script for OSCAR
# Sets up conda environment with all necessary dependencies
#

echo "================================"
echo "OSCAR Environment Setup"
echo "================================"

# Load required modules
echo "Loading modules..."
module load miniconda3/23.11.0s
module load cuda/11.4
module load cudnn/8.2

# Create conda environment
echo "Creating conda environment 'csci1470'..."
conda create -n csci1470 python=3.9 -y

# Activate environment
echo "Activating environment..."
source activate csci1470

# Install TensorFlow with GPU support
echo "Installing TensorFlow..."
conda install -c conda-forge tensorflow-gpu=2.10 -y

# Install other dependencies
echo "Installing other dependencies..."
pip install librosa==0.10.1
pip install soundfile==0.12.1
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install scikit-learn==1.3.0
pip install wandb==0.15.8
pip install tqdm==4.66.1
pip install audioread==3.0.0

# Verify installation
echo ""
echo "================================"
echo "Verifying installation..."
echo "================================"

python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))"
python -c "import librosa; print('Librosa version:', librosa.__version__)"
python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"

echo ""
echo "================================"
echo "Setup complete!"
echo "================================"
echo ""
echo "To activate the environment, run:"
echo "  module load miniconda3/23.11.0s cuda/11.4 cudnn/8.2"
echo "  source activate csci1470"
echo ""
