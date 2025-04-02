#!/bin/bash

# Script to run the standalone pretraining for job shop scheduling models
# This script does not rely on pretrained weights

# Go to HybridPPO directory to ensure correct relative paths
cd "$(dirname "$0")"

# Create necessary directories if they don't exist
mkdir -p checkpoints

# Force recreate the dataset subset every time
echo "Creating/Refreshing dataset subset using prepare_subset.py..."

# Check if original dataset exists
if [ -d "../SelfLabelingJobShop/dataset5k" ]; then
    # Use prepare_subset.py to create a diverse subset
    python utils/prepare_subset.py \
        --source ../SelfLabelingJobShop/dataset5k \
        --target dataset5k_subset \
        --num 2500
        
    echo "Created dataset subset with $(ls dataset5k_subset | wc -l) instances"
else
    echo "Original dataset not found at ../SelfLabelingJobShop/dataset5k"
    echo "Please ensure the SelfLabelingJobShop repository is cloned correctly"
    exit 1
fi

# Set up Python environment
# Uncomment and modify the following lines according to your environment
# conda activate jss_env  # or
# source venv/bin/activate

# Run the pretraining script
python self_labeling_pretrain_standalone.py \
    --data_path ./dataset5k_subset \
    --val_path ../SelfLabelingJobShop/benchmarks/validation \
    --model_path ./checkpoints/pretrained_model_standalone.pt \
    --num_instances 50 \
    --iterations 2 \
    --batch_size 32 \
    --lr 2e-4 \
    --epochs 1 \
    --hidden_dim 64 \
    --embed_dim 128 \
    --num_heads 3 \
    --dropout 0.15 \
    --seed 42

echo "Pretraining completed. Model saved to ./checkpoints/pretrained_model_standalone.pt" 
