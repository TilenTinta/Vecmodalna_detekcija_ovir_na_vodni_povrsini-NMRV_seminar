#!/bin/bash
#SBATCH --job-name=barcica        # Job name
#SBATCH --output=job_%j.out       # Standard output and error log (%j expands to jobId)
#SBATCH --error=job_%j.err        # Error log
#SBATCH --time=12:00:00           # Time limit hrs:min:sec
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks per node
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=10        # CPU cores per task
#SBATCH --mem=32GB                # Memory limit
#SBATCH --partition=gpu           # GPU partition

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

# Load necessary modules (modify according to your cluster's setup)
module purge
module load CUDA/12.8.0
module load PyTorch/1.12.0-foss-2022a-CUDA-12.8.0
module load Python/3.10.4-GCCcore-11.3.0
module load matplotlib/3.5.2-foss-2022a
module load tqdm/4.64.1-GCCcore-12.2.0

# First, let's install albumentations since it's not available as a module
pip install torchvision
pip install ultralytics

# Run the training script
python train.py

# Print end time
echo "End time: $(date)"
