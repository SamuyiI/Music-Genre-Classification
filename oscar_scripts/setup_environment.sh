#!/bin/bash
################################################################################
# OSCAR Environment Setup Script for Music Genre Classification
# Run this script ONCE to set up your Python environment with all dependencies
#
# Usage: bash setup_environment.sh
################################################################################

set -e  # Exit on any error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ENV_NAME="csci1470"

# Check if running on OSCAR
if [[ ! -d "/oscar" ]]; then
    echo -e "${RED}ERROR: This script must be run on OSCAR!${NC}"
    exit 1
fi

# Load conda module
echo -e "${BLUE}[1/6] Loading miniconda module...${NC}"
module load miniconda3/23.11.0s
echo -e "${GREEN}Module loaded :)${NC}"
echo ""

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment '${ENV_NAME}' already exists.${NC}"
    read -p "Do you want to remove it and reinstall? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Removing existing environment...${NC}"
        conda env remove -n ${ENV_NAME} -y
        echo -e "${GREEN}Old environment removed :)${NC}"
    else
        echo -e "${YELLOW}Keeping existing environment. Exiting.${NC}"
        exit 0
    fi
fi

# Create conda environment
echo -e "${BLUE}[2/6] Creating conda environment '${ENV_NAME}' with Python 3.10...${NC}"
conda create -n ${ENV_NAME} python=3.10 -y
echo -e "${GREEN}Environment created :)${NC}"
echo ""

# Activate environment
echo -e "${BLUE}[3/6] Activating environment...${NC}"
source activate ${ENV_NAME}
echo -e "${GREEN}Environment activated :)${NC}"
echo ""

# Load CUDA modules
echo -e "${BLUE}[4/6] Loading CUDA modules for GPU support...${NC}"
module load cuda/11.8.0-lpttyok
echo -e "${GREEN}CUDA modules loaded :)${NC}"
echo ""

# Install packages
echo -e "${BLUE}[5/6] Installing required packages...${NC}"
echo "   - tensorflow (this may take a few minutes)"
pip install --quiet tensorflow==2.15.0
echo "   - librosa (for audio processing)"
pip install --quiet librosa==0.10.1
echo "   - soundfile (for audio I/O)"
pip install --quiet soundfile==0.12.1
echo "   - audioread (for audio format support)"
pip install --quiet audioread==3.0.0
echo "   - scikit-learn (for data splitting and preprocessing)"
pip install --quiet scikit-learn==1.3.0
echo "   - numpy"
pip install --quiet numpy==1.24.3
echo "   - pandas (for metadata handling)"
pip install --quiet pandas==2.0.3
echo "   - matplotlib (for visualization)"
pip install --quiet matplotlib==3.7.2
echo "   - seaborn (for visualization)"
pip install --quiet seaborn==0.12.2
echo "   - wandb dependencies"
pip install --quiet click pyyaml platformdirs
echo "   - wandb (for experiment tracking)"
pip install --quiet wandb==0.15.8
echo "   - tqdm (for progress bars)"
pip install --quiet tqdm==4.66.1
echo -e "${GREEN}All packages installed :)${NC}"
echo ""

# Test installation
echo -e "${BLUE}[6/6] Testing installation...${NC}"
python << 'EOF'
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import sklearn
import wandb
import tqdm
import matplotlib
import seaborn

print(f"Python version: {sys.version.split()[0]}")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Librosa version: {librosa.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"wandb version: {wandb.__version__}")
print(f"tqdm version: {tqdm.__version__}")
print(f"matplotlib version: {matplotlib.__version__}")
print(f"seaborn version: {seaborn.__version__}")

# Check GPU availability (will show 0 on login node, but that's expected)
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs detected on this node: {len(gpus)}")
if len(gpus) == 0:
    print("  (Note: You're on a login node. GPUs will be available when you submit a job)")
EOF

echo -e "${GREEN}Installation test passed :)${NC}"
echo ""

# Print usage instructions
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "To use this environment in the future:"
echo -e "  ${YELLOW}1. In interactive sessions:${NC}"
echo -e "     module load miniconda3/23.11.0s"
echo -e "     module load cuda/11.8.0-lpttyok"
echo -e "     conda activate ${ENV_NAME}"
echo ""
echo -e "  ${YELLOW}2. In SLURM batch scripts, add these lines:${NC}"
echo -e "     module load miniconda3/23.11.0s"
echo -e "     module load cuda/11.8.0-lpttyok"
echo -e "     source activate ${ENV_NAME}"
echo ""
echo -e "  ${YELLOW}3. To test GPU access (on a GPU node):${NC}"
echo -e "     interact -g 1"
echo -e "     module load miniconda3/23.11.0s cuda/11.8.0-lpttyok"
echo -e "     conda activate ${ENV_NAME}"
echo -e "     python -c \"import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))\""
echo ""
echo -e "  ${YELLOW}4. Don't forget to set your Wandb API key:${NC}"
echo -e "     export WANDB_API_KEY='your-api-key-here'"
echo -e "     # Or add to ~/.bashrc for persistence:"
echo -e "     echo 'export WANDB_API_KEY=\"your-api-key-here\"' >> ~/.bashrc"
echo ""
echo -e "${GREEN}You're ready to train your music genre classification models!${NC}"
