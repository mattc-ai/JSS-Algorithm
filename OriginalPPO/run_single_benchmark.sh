#!/bin/bash
# Script to train the PPO baseline on a single benchmark
# This is useful for testing a single benchmark before running the full experiment

set -e  # Exit on any error

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Create necessary directories - ensure they're in the OriginalPPO directory
mkdir -p "$SCRIPT_DIR/ppo_models"

# Default benchmark
BENCHMARK=${1:-ta01}

echo "==============================================="
echo "Training PPO Baseline on Benchmark: $BENCHMARK"
echo "==============================================="

# Run training with verbose output
python -u "$SCRIPT_DIR/train_single_benchmark.py" \
    --benchmark "$BENCHMARK" \
    --num_episodes 1000 \
    --hidden_dim 64 \
    --lr 3e-4 \
    --gamma 0.99 \
    --clip_ratio 0.2 \
    --value_coef 0.5 \
    --entropy_coef 0.01 \
    --max_grad_norm 0.5 \
    --ppo_epochs 10 \
    --batch_size 64 \
    --gae_lambda 0.95 \
    --update_frequency 10 \
    --evaluation_frequency 20 \
    --use_progressive_schedule \
    --initial_steps_per_episode 1000 \
    --steps_increment 500 \
    --increment_interval 100 \
    --auto_entropy_tuning \
    --entropy_decay_rate 0.7 \
    --eval_episodes 10 \
    --save_dir "$SCRIPT_DIR/ppo_models" \
    --results_file "$SCRIPT_DIR/ppo_models/ppo_results.json" \
    --max_steps_per_episode 5000 \
    --seed 42

if [ $? -ne 0 ]; then
    echo "PPO baseline training failed. Please check the error messages."
    exit 1
fi

echo "==============================================="
echo "PPO baseline training completed successfully!"
echo "Model saved to: $SCRIPT_DIR/ppo_models/${BENCHMARK}_ppo_agent.pt"
echo "Results saved to: $SCRIPT_DIR/ppo_models/ppo_results.json"
echo "===============================================" 