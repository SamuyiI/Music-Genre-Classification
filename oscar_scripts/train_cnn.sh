#!/bin/bash
#SBATCH --job-name=music_cnn
#SBATCH --output=logs/cnn_%j.out
#SBATCH --error=logs/cnn_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "================================"
echo "Training CNN Model on OSCAR"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules
module load miniconda3/23.11.0s
module load cuda/11.4
module load cudnn/8.2

# Activate environment
source activate csci1470

# Set environment variables
export WANDB_API_KEY=${WANDB_API_KEY}
export TF_CPP_MIN_LOG_LEVEL=2

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Print GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Train CNN model
echo "Starting CNN training..."
python src/train.py \
    --model_type cnn \
    --data_path data/processed/mel_features.pkl \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --dropout_rate 0.25 \
    --use_wandb

echo ""
echo "Training complete!"
echo "End time: $(date)"
