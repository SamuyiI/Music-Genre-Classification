# Music Genre Classification
## CSCI 1470 Deep Learning - Final Project

This project explores how deep learning models capture genre-distinguishing features in music, with a particular focus on hybrid genres like Afrobeats that blend West African traditions with hip-hop and electronic elements.

## Project Overview

This repository implements three deep learning architectures for classifying music into 8 genres using the FMA-small dataset:

1. **CNN Model**: Convolutional Neural Network for mel-spectrogram classification
2. **LSTM Model**: Bidirectional LSTM for MFCC temporal sequence modeling
3. **Hybrid CNN-LSTM**: Combined architecture leveraging both spatial and temporal features

## Dataset

**FMA-small**: 8,000 tracks of 30-second audio clips across 8 genres
- Download from: https://github.com/mdeff/fma
- Sample rate: 22050 Hz
- Duration: 30 seconds per track
- Genres: Hip-Hop, Rock, Electronic, Experimental, Folk, Instrumental, Pop, International

## Project Structure

```
Music-Genre-Classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── fma_small/              # Raw audio files (not in repo)
│   ├── fma_metadata.csv        # Metadata file (not in repo)
│   └── processed/              # Preprocessed features
│       ├── mel_features.pkl
│       └── mfcc_features.pkl
├── src/
│   ├── data_preprocessing.py   # Audio preprocessing pipeline
│   ├── augmentation.py         # Data augmentation techniques
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── models/
│   │   ├── cnn_model.py       # CNN architecture
│   │   ├── rnn_model.py       # LSTM/GRU architecture
│   │   └── hybrid_model.py    # Hybrid CNN-LSTM architecture
│   └── visualization/
│       ├── gradcam.py         # Grad-CAM visualization
│       ├── tsne_viz.py        # t-SNE embeddings
│       └── confusion_matrix.py # Confusion matrix analysis
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_experiments.ipynb
│   └── 03_results_analysis.ipynb
├── oscar_scripts/
│   ├── setup_environment.sh   # OSCAR environment setup
│   ├── train_cnn.sh          # SLURM script for CNN
│   ├── train_lstm.sh         # SLURM script for LSTM
│   └── train_hybrid.sh       # SLURM script for Hybrid
├── checkpoints/               # Saved model weights
├── results/
│   ├── figures/              # Generated visualizations
│   ├── tables/               # Metrics and results
│   └── metrics/              # Evaluation metrics
└── logs/                     # Training logs

```

## Installation

### Local Setup (macOS with M1/M2)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### OSCAR Setup

```bash
# SSH into OSCAR
ssh username@ssh.ccv.brown.edu

# Clone repository
git clone git@github.com:yourusername/Music-Genre-Classification.git
cd Music-Genre-Classification

# Run setup script
bash oscar_scripts/setup_environment.sh
```

## Usage

### 1. Data Preprocessing

```bash
# Preprocess audio files to extract mel-spectrograms and MFCCs
python src/data_preprocessing.py

# Test on small subset first (recommended)
python src/data_preprocessing.py --max_samples 100
```

### 2. Training Models

#### Local Training (for testing)

```bash
# Train CNN model
python src/train.py \
    --model_type cnn \
    --data_path data/processed/mel_features.pkl \
    --epochs 5 \
    --batch_size 16 \
    --subset 1000

# Train LSTM model
python src/train.py \
    --model_type lstm \
    --data_path data/processed/mfcc_features.pkl \
    --epochs 5 \
    --batch_size 16 \
    --subset 1000

# Train Hybrid model
python src/train.py \
    --model_type hybrid \
    --data_path data/processed/mel_features.pkl \
    --epochs 5 \
    --batch_size 16 \
    --subset 1000
```

#### OSCAR Training (for full models)

```bash
# Submit training jobs
sbatch oscar_scripts/train_cnn.sh
sbatch oscar_scripts/train_lstm.sh
sbatch oscar_scripts/train_hybrid.sh

# Check job status
squeue -u $USER

# View logs
tail -f logs/cnn_<job_id>.out
```

### 3. Evaluation

```bash
# Evaluate trained model
python src/evaluate.py \
    --model_path checkpoints/cnn_best.h5 \
    --data_path data/processed/mel_features.pkl \
    --output_dir results/metrics \
    --save_predictions
```

### 4. Visualization

```python
# In Python or Jupyter notebook

from src.visualization.gradcam import visualize_genre_patterns
from src.visualization.tsne_viz import create_tsne_visualization
from src.visualization.confusion_matrix import analyze_confusion_matrix

# Grad-CAM visualization
visualize_genre_patterns(model, X_test, y_test, label_encoder)

# t-SNE embeddings
create_tsne_visualization(model, X_test, y_test, label_encoder)

# Confusion matrix analysis
analyze_confusion_matrix(y_true, y_pred, class_names)
```

## Model Architectures

### CNN Model
- 4 Convolutional blocks (32, 64, 128, 256 filters)
- Batch Normalization + MaxPooling + Dropout
- Global Average Pooling
- Dense layers (512 units) with Dropout
- Output: Softmax (8 classes)

### LSTM Model
- 2 Bidirectional LSTM layers (256 units each)
- Dropout regularization
- Dense layers (512 units)
- Output: Softmax (8 classes)

### Hybrid CNN-LSTM Model
- CNN feature extraction (3 blocks)
- Reshape for temporal processing
- 2 Bidirectional LSTM layers
- Dense classification head
- Output: Softmax (8 classes)

## Expected Performance

| Model Type | Expected Accuracy | Training Time |
|------------|------------------|---------------|
| Random Baseline | 12.5% | - |
| Basic CNN | 50-60% | 2-3 hours |
| Well-tuned CNN | 65-75% | 5-8 hours |
| LSTM on MFCCs | 55-65% | 8-12 hours |
| Hybrid CNN-LSTM | 70-80% | 10-15 hours |

## Ablation Studies

The project includes 4 ablation studies:

1. **Audio Representation Comparison**: Raw spectrograms vs. Mel-spectrograms vs. MFCCs
2. **Architecture Comparison**: CNN vs. LSTM vs. GRU vs. Hybrid
3. **Data Augmentation Impact**: No augmentation vs. various augmentation techniques
4. **Hyperparameter Tuning**: Learning rate, dropout, LSTM units, batch size

## Key Features

- **Comprehensive Pipeline**: From raw audio to trained models
- **Multiple Architectures**: CNN, LSTM, and Hybrid models
- **Visualization Tools**: Grad-CAM, t-SNE, confusion matrices
- **OSCAR Integration**: SLURM scripts for GPU training
- **Wandb Logging**: Track experiments and compare results
- **Ablation Studies**: Systematic analysis of design choices

## Musical Insights

This project leverages musical domain knowledge to interpret results:

- **Afrobeats ↔ Hip-Hop confusion**: Shared rhythmic patterns and 808 drums
- **Instrumental signatures**: How models detect talking drums, bass lines, etc.
- **Spectral patterns**: Relationship between frequency features and genre characteristics
- **Temporal modeling**: How LSTM captures chord progressions and song structure

## Dependencies

- TensorFlow 2.13+ (tensorflow-macos for M1/M2)
- Librosa 0.10+
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- Wandb

See [requirements.txt](requirements.txt) for complete list.

## Checkpoint (December 1, 2025)

Required deliverables:
- ✓ CNN model trained (50-60% accuracy)
- ✓ LSTM model trained (55-65% accuracy)
- ✓ Training curves for both models
- ✓ Basic confusion matrix
- ✓ Presentation slides

## Final Submission (December 11, 2025)

Complete deliverables:
- All three models trained and evaluated
- 3-4 ablation studies completed
- All visualizations generated (Grad-CAM, t-SNE, confusion matrices)
- Final report (8-12 pages)
- Code repository with documentation
- Presentation slides

## Timeline

See [Music_Genre_Classification_Guide.pdf](Music_Genre_Classification_Guide.pdf) for detailed day-by-day schedule.

## Troubleshooting

### Common Issues

1. **CNN not learning**: Check spectrogram normalization, learning rate
2. **LSTM too slow**: Use GRU, reduce sequence length, or train on OSCAR
3. **Model overfitting**: Increase dropout, add data augmentation
4. **OSCAR job failing**: Check memory allocation, verify data paths
5. **Accuracy stuck at 12.5%**: Verify labels are loaded correctly

See guide for detailed troubleshooting steps.

## References

- FMA Dataset: https://github.com/mdeff/fma
- Course materials: CSCI 1470 Deep Learning
- Grad-CAM: https://arxiv.org/abs/1610.02391

## Author

Samuyi - CSCI 1470 Deep Learning

## License

This project is for educational purposes as part of CSCI 1470.
