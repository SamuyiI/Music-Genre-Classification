#!/bin/bash
#SBATCH --job-name=confusion_matrices
#SBATCH --output=logs/confusion_%j.out
#SBATCH --error=logs/confusion_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch

echo "================================"
echo "Generating Confusion Matrices on OSCAR"
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
export TF_CPP_MIN_LOG_LEVEL=2

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Generate confusion matrices
echo "Generating confusion matrices..."
python src/generate_confusion_matrices.py

echo ""
echo "Confusion matrix generation complete!"
echo "End time: $(date)"
echo ""
echo "To download the confusion matrices to your local machine, run:"
echo "scp -r omizevbi@ssh.ccv.brown.edu:~/Music-Genre-Classification/results/figures/confusion_matrices/ results/figures/"
