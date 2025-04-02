#!/bin/bash
# Script to run a single benchmark with the Hybrid PPO agent
# Usage: ./run_hybrid_single.sh BENCHMARK [NUM_EPISODES] [EVAL_EPISODES]

# Fail on first error
set -e

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Process command line arguments
BENCHMARK="${1:-ta01}"  # Default benchmark: ta01
NUM_EPISODES="${2:-1000}"  # Default episodes: 1000
EVAL_EPISODES="${3:-10}"  # Default evaluation episodes: 10

# Configuration
SAVE_DIR="$PARENT_DIR/hybrid_models"
RESULTS_FILE="$SAVE_DIR/ppo_results.json"
HIDDEN_DIM=64
LR=3e-4
GAMMA=0.99
CLIP_RATIO=0.2
VALUE_COEF=0.5
ENTROPY_COEF=0.01
MAX_GRAD_NORM=0.5
PPO_EPOCHS=10
BATCH_SIZE=64
GAE_LAMBDA=0.95
UPDATE_FREQ=10
EVAL_FREQ=20
INITIAL_STEPS=1000
STEPS_INCREMENT=500
INCREMENT_INTERVAL=100
ENTROPY_DECAY=0.7
MAX_STEPS=5000
SEED=42

# Create directories
mkdir -p "$SAVE_DIR"

# Get pretrained model path
PRETRAINED_MODEL_PATH="$PARENT_DIR/checkpoints/pretrained_model_standalone.pt"

# Verify pretrained model exists
if [ -f "$PRETRAINED_MODEL_PATH" ]; then
    echo "Found pretrained model at: $PRETRAINED_MODEL_PATH"
else
    echo "Warning: Pretrained model not found at $PRETRAINED_MODEL_PATH"
    echo "Training will proceed without pretrained weights"
    PRETRAINED_MODEL_PATH=""
fi

echo "========================================================"
echo "Training Hybrid PPO on Benchmark: $BENCHMARK"
echo "Episodes: $NUM_EPISODES, Evaluation episodes: $EVAL_EPISODES"
echo "========================================================"

# Build command with correct parameters
CMD="python -u \"$SCRIPT_DIR/hybrid_ppo_agent.py\" --mode train --benchmarks \"$BENCHMARK\" \
    --num_episodes $NUM_EPISODES \
    --hidden_dim $HIDDEN_DIM \
    --lr $LR \
    --gamma $GAMMA \
    --clip_ratio $CLIP_RATIO \
    --value_coef $VALUE_COEF \
    --entropy_coef $ENTROPY_COEF \
    --max_grad_norm $MAX_GRAD_NORM \
    --ppo_epochs $PPO_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gae_lambda $GAE_LAMBDA \
    --update_frequency $UPDATE_FREQ \
    --evaluation_frequency $EVAL_FREQ \
    --use_progressive_schedule \
    --initial_steps_per_episode $INITIAL_STEPS \
    --steps_increment $STEPS_INCREMENT \
    --increment_interval $INCREMENT_INTERVAL \
    --auto_entropy_tuning \
    --entropy_decay_rate $ENTROPY_DECAY \
    --eval_episodes $EVAL_EPISODES \
    --results_dir \"$SAVE_DIR\" \
    --save_json \"$RESULTS_FILE\" \
    --max_steps_per_episode $MAX_STEPS \
    --seed $SEED"

# Add pretrained model path if available
if [ -n "$PRETRAINED_MODEL_PATH" ]; then
    CMD="$CMD --pretrained_model_path \"$PRETRAINED_MODEL_PATH\" --use_gnn"
fi

# Execute the command
eval $CMD

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully for $BENCHMARK!"
    echo "Model saved to: $SAVE_DIR/${BENCHMARK}_hybrid_ppo_agent.pt"
    echo "Training curves saved to: $SAVE_DIR/${BENCHMARK}_training_curves.png"
    echo "Results saved to: $RESULTS_FILE"
    echo "========================================================"
    echo "Hybrid PPO process completed successfully!"
    echo "========================================================"
else
    echo "Error training on benchmark: $BENCHMARK"
    exit 1
fi 