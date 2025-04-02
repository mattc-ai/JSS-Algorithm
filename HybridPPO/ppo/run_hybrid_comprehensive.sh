#!/bin/bash
# Comprehensive script for training Hybrid PPO agents on multiple benchmarks
# with cooling periods to prevent thermal throttling and proper error handling

# Fail on first error
set -e

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration - Match original PPO implementation
SAVE_DIR="$PARENT_DIR/hybrid_models"
RESULTS_FILE="$SAVE_DIR/ppo_results.json"
NUM_EPISODES=300 #1000
HIDDEN_DIM=64
LR=0.0003
GAMMA=0.99
CLIP_RATIO=0.2
VALUE_COEF=0.5
ENTROPY_COEF=0.01
MAX_GRAD_NORM=0.5
PPO_EPOCHS=10
BATCH_SIZE=64
GAE_LAMBDA=0.95
UPDATE_FREQ=20 #10
EVAL_FREQ=20
INITIAL_STEPS=1000
STEPS_INCREMENT=200 #500
INCREMENT_INTERVAL=100
ENTROPY_DECAY=0.7
EVAL_EPISODES=10
MAX_STEPS=2000 #5000
SEED=42
COOLING_PERIOD=60  # 1 minute cooling between benchmarks
MAX_RETRIES=3      # Maximum number of retries for failed benchmarks

# Create directories
mkdir -p "$SAVE_DIR"

# Log file
LOG_FILE="$PARENT_DIR/hybrid_training_log.txt"
echo "Starting Hybrid PPO training at $(date)" > "$LOG_FILE"

# Get pretrained model path (can be skipped with SKIP_PRETRAINED=1)
if [ "${SKIP_PRETRAINED}" == "1" ]; then
    echo "Skipping pretrained model as requested by SKIP_PRETRAINED=1"
    PRETRAINED_MODEL_PATH=""
else
    PRETRAINED_MODEL_PATH="$PARENT_DIR/checkpoints/pretrained_model_standalone.pt"

    # Verify pretrained model exists
    if [ -f "$PRETRAINED_MODEL_PATH" ]; then
        echo "Found pretrained model at: $PRETRAINED_MODEL_PATH"
    else
        echo "Warning: Pretrained model not found at $PRETRAINED_MODEL_PATH"
        echo "Training will proceed without pretrained weights"
        PRETRAINED_MODEL_PATH=""
    fi
fi

# Function to train on a benchmark with retries
train_benchmark() {
    benchmark=$1
    retries=0
    success=false
    
    echo "===== Starting training on benchmark: $benchmark =====" | tee -a "$LOG_FILE"
    echo "$(date): Starting $benchmark" >> "$LOG_FILE"
    
    while [ $retries -lt $MAX_RETRIES ] && [ "$success" = false ]; do
        if [ $retries -gt 0 ]; then
            echo "Retry $retries for benchmark $benchmark" | tee -a "$LOG_FILE"
            # Short cooling period before retry
            sleep 30
        fi
        
        # Build command with correct parameters according to hybrid_ppo_agent.py
        CMD="python -u \"$SCRIPT_DIR/hybrid_ppo_agent.py\" --mode train --benchmarks \"$benchmark\" \
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
            echo "Training completed successfully for $benchmark." | tee -a "$LOG_FILE"
            echo "Model saved to: $SAVE_DIR/${benchmark}_hybrid_ppo_agent.pt" | tee -a "$LOG_FILE"
            echo "Training curves saved to: $SAVE_DIR/${benchmark}_training_curves.png" | tee -a "$LOG_FILE"
            success=true
        else
            echo "Error training on benchmark: $benchmark (Attempt $((retries+1)))" | tee -a "$LOG_FILE"
            retries=$((retries+1))
        fi
    done
    
    if [ "$success" = false ]; then
        echo "Failed to train on benchmark $benchmark after $MAX_RETRIES attempts." | tee -a "$LOG_FILE"
    fi
    
    echo "$(date): Finished $benchmark" >> "$LOG_FILE"
    
    # Return success status
    if [ "$success" = true ]; then
        return 0
    else
        return 1
    fi
}

# Function for cooling period
cooling_period() {
    duration=$1
    echo "===== Starting cooling period for $duration seconds... =====" | tee -a "$LOG_FILE"
    echo "$(date): Starting cooling period" >> "$LOG_FILE"
    
    # Display progress every 10 seconds
    for i in $(seq $duration -10 0); do
        if [ $((i % 60)) -eq 0 ]; then
            echo "Cooling: $((i / 60)) minutes remaining..."
        fi
        sleep 10
    done
    
    echo "Cooling period complete." | tee -a "$LOG_FILE"
    echo "$(date): Cooling period complete" >> "$LOG_FILE"
}

# Function to check system temperature
check_temperature() {
    # This is a placeholder. You might use platform-specific tools for actual measurement
    echo "Checking system temperature..." | tee -a "$LOG_FILE"
    
    # Example: On macOS with osx-cpu-temp
    if command -v osx-cpu-temp &> /dev/null; then
        TEMP=$(osx-cpu-temp | grep -oE '[0-9]+\.[0-9]+')
        echo "Current CPU temperature: ${TEMP}°C" | tee -a "$LOG_FILE"
        
        # Check if temperature is too high (example threshold: 80°C)
        if (( $(echo "$TEMP > 80" | bc -l) )); then
            echo "Temperature too high! Cooling for 5 minutes before proceeding." | tee -a "$LOG_FILE"
            cooling_period 300
        fi
    fi
    
    return 0  # Continue regardless of temperature check
}

# Main benchmark list - matching the original PPO script
BENCHMARKS=(
    "ta01" "ta02" "ta03" "ta11" "ta12" "ta21" "ta22"
)

# Declare arrays to track progress
successful_benchmarks=()
failed_benchmarks=()

# Process all benchmarks with cooling periods
echo "===== Starting Hybrid PPO training on all benchmarks =====" | tee -a "$LOG_FILE"
echo "Configuration:" | tee -a "$LOG_FILE"
echo "- Episodes per benchmark: $NUM_EPISODES" | tee -a "$LOG_FILE"
echo "- Evaluation episodes: $EVAL_EPISODES" | tee -a "$LOG_FILE"
echo "- Results directory: $SAVE_DIR" | tee -a "$LOG_FILE"
echo "- Results file: $RESULTS_FILE" | tee -a "$LOG_FILE"
echo "- Cooling period: $COOLING_PERIOD seconds" | tee -a "$LOG_FILE"
echo "- Max retries: $MAX_RETRIES" | tee -a "$LOG_FILE"
echo "- Using pretrained: $([ -n "$PRETRAINED_MODEL_PATH" ] && echo "Yes" || echo "No")" | tee -a "$LOG_FILE"

total_benchmarks=${#BENCHMARKS[@]}
current=1

for benchmark in "${BENCHMARKS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "===== Processing benchmark $current of $total_benchmarks: $benchmark =====" | tee -a "$LOG_FILE"
    
    # Check if we already have results for this benchmark
    if [ -f "$SAVE_DIR/${benchmark}_hybrid_ppo_agent.pt" ] && [ -f "$RESULTS_FILE" ]; then
        if grep -q "\"$benchmark\"" "$RESULTS_FILE"; then
            echo "Benchmark $benchmark already trained. Skipping." | tee -a "$LOG_FILE"
            successful_benchmarks+=("$benchmark")
            current=$((current+1))
            continue
        fi
    fi
    
    # Check temperature before processing
    check_temperature
    
    # Train on benchmark
    if train_benchmark "$benchmark"; then
        echo "Training $benchmark completed successfully" | tee -a "$LOG_FILE"
        successful_benchmarks+=("$benchmark")
    else
        echo "Training $benchmark failed after $MAX_RETRIES attempts" | tee -a "$LOG_FILE"
        failed_benchmarks+=("$benchmark")
    fi
    
    # Cooling period if not the last benchmark
    if [ $current -lt $total_benchmarks ]; then
        cooling_period $COOLING_PERIOD
    fi
    
    # Increment the counter
    current=$((current+1))
done

# Print summary
echo "" | tee -a "$LOG_FILE"
echo "===== Hybrid PPO Training Summary =====" | tee -a "$LOG_FILE"
echo "Total benchmarks processed: ${#BENCHMARKS[@]}" | tee -a "$LOG_FILE"
echo "Successfully trained: ${#successful_benchmarks[@]}" | tee -a "$LOG_FILE"
echo "Failed benchmarks: ${#failed_benchmarks[@]}" | tee -a "$LOG_FILE"

# List successful benchmarks
echo "" | tee -a "$LOG_FILE"
echo "Successful benchmarks:" | tee -a "$LOG_FILE"
for b in "${successful_benchmarks[@]}"; do
    echo "- $b" | tee -a "$LOG_FILE"
done

# List failed benchmarks
if [ ${#failed_benchmarks[@]} -gt 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "Failed benchmarks:" | tee -a "$LOG_FILE"
    for b in "${failed_benchmarks[@]}"; do
        echo "- $b" | tee -a "$LOG_FILE"
    done
fi

echo "" | tee -a "$LOG_FILE"
echo "===== Hybrid PPO training process completed at $(date) =====" | tee -a "$LOG_FILE"
echo "Results saved to: $RESULTS_FILE" | tee -a "$LOG_FILE"

# Generate comparison report if PPO results exist
PPO_RESULTS="$PARENT_DIR/../OriginalPPO/ppo_models/ppo_results.json"
if [ -f "$PPO_RESULTS" ]; then
    echo "=================================================" | tee -a "$LOG_FILE"
    echo "Generating comparison with original PPO results" | tee -a "$LOG_FILE"
    echo "=================================================" | tee -a "$LOG_FILE"
