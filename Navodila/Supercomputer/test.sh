#!/bin/bash
#SBATCH --job-name=barcica_test     # Job name
#SBATCH --output=test_%j.out       # Standard output and error log
#SBATCH --error=test_%j.err        # Error log
#SBATCH --time=01:00:00            # Time limit hrs:min:sec (less than training)
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks per node
#SBATCH --gres=gpu:2              # Request 1 GPU
#SBATCH --cpus-per-task=4         # CPU cores per task
#SBATCH --mem=64GB                # Memory limit
#SBATCH --partition=gpu           # GPU partition

# Check if model path is provided
#if [ -z "$1" ]; then
#    echo "Error: Model path not provided"
#    echo "Usage: sbatch test.sh <path_to_model_checkpoint>"
#    exit 1
#fi

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
#echo "Model path: $1"

# Load necessary modules
module purge
module load CUDA/12.8.0
module load PyTorch/1.12.0-foss-2022a-CUDA-12.8.0
module load Python/3.10.4-GCCcore-11.3.0
module load matplotlib/3.5.2-foss-2022a
module load tqdm/4.64.1-GCCcore-12.2.0

# First, let's install albumentations since it's not available as a module
pip install torchvision
pip install ultralytics

# Run the test script with provided model path
export MERGED_OUTPUT=/home/tt5935/output/merged_detections
python test.py 

# Print end time
echo "End time: $(date)"
