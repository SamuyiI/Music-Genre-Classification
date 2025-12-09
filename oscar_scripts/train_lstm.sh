#!/bin/bash
#SBATCH --job-name=music_lstm
#SBATCH --output=logs/lstm_%j.out
#SBATCH --error=logs/lstm_%j.err
#SBATCH --time=15:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

# NOTE: LSTM/RNN training doesn't benefit much from GPU with TensorFlow eager execution
# Running on CPU partition for better performance

echo "================================"
echo "Training LSTM Model on OSCAR"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules
module load miniconda3/23.11.0s

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate environment
conda activate csci1470

# Set environment variables
export WANDB_API_KEY=${WANDB_API_KEY}
export TF_CPP_MIN_LOG_LEVEL=2

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Train LSTM model
echo "Starting LSTM training..."
python src/train.py \
    --model_type lstm \
    --data_path data/processed/mfcc_features.pkl \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --dropout_rate 0.3 \
    --lstm_units 256 \
    --use_wandb

echo ""
echo "Training complete!"
echo "End time: $(date)"
