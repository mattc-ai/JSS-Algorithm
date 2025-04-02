#!/bin/bash
# Script to train PPO agents on multiple benchmarks sequentially
# with cooling periods in between to prevent thermal throttling

# Make the script executable
# chmod +x train_all_benchmarks.sh

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Configuration
SAVE_DIR="$SCRIPT_DIR/ppo_models"
RESULTS_FILE="$SAVE_DIR/ppo_results.json"
NUM_EPISODES=1000
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
UPDATE_FREQ=10
EVAL_FREQ=20
USE_PROGRESSIVE=true
INITIAL_STEPS=1000
STEPS_INCREMENT=500
INCREMENT_INTERVAL=100
AUTO_ENTROPY=true
ENTROPY_DECAY=0.7
EVAL_EPISODES=10
COOLING_PERIOD=60  # 1 minute in seconds
MAX_RETRIES=3       # Maximum number of retries for a failed benchmark
MAX_STEPS=5000      # Maximum steps per episode - increased for larger benchmarks
SEED=42            # Fixed seed for reproducibility

# Create directories
mkdir -p "$SAVE_DIR"

# Log file
LOG_FILE="$SCRIPT_DIR/training_log.txt"
echo "Starting training at $(date)" > "$LOG_FILE"

# Function to train on a benchmark with retries
train_benchmark() {
    benchmark=$1
    retries=0
    success=false
    
    echo "Starting training on benchmark: $benchmark" | tee -a "$LOG_FILE"
    echo "$(date): Starting $benchmark" >> "$LOG_FILE"
    
    while [ $retries -lt $MAX_RETRIES ] && [ "$success" = false ]; do
        if [ $retries -gt 0 ]; then
            echo "Retry $retries for benchmark $benchmark" | tee -a "$LOG_FILE"
            # Short cooling period before retry
            sleep 30
        fi
        
        # Try to train
        CMD="python \"$SCRIPT_DIR/train_single_benchmark.py\" --benchmark \"$benchmark\" --num_episodes $NUM_EPISODES --hidden_dim $HIDDEN_DIM --lr $LR --gamma $GAMMA --clip_ratio $CLIP_RATIO --value_coef $VALUE_COEF --entropy_coef $ENTROPY_COEF --max_grad_norm $MAX_GRAD_NORM --ppo_epochs $PPO_EPOCHS --batch_size $BATCH_SIZE --gae_lambda $GAE_LAMBDA --update_frequency $UPDATE_FREQ"
        
        # Add optional flags
        if [ "$USE_PROGRESSIVE" = true ]; then
            CMD="$CMD --use_progressive_schedule"
        fi
        
        CMD="$CMD --initial_steps_per_episode $INITIAL_STEPS --steps_increment $STEPS_INCREMENT --increment_interval $INCREMENT_INTERVAL"
        
        if [ "$AUTO_ENTROPY" = true ]; then
            CMD="$CMD --auto_entropy_tuning"
        fi
        
        CMD="$CMD --entropy_decay_rate $ENTROPY_DECAY --eval_episodes $EVAL_EPISODES --save_dir \"$SAVE_DIR\" --results_file \"$RESULTS_FILE\" --max_steps_per_episode $MAX_STEPS --resume --seed $SEED"
        
        # Execute the command
        eval $CMD
        
        # Check if training was successful
        if [ $? -eq 0 ]; then
            echo "Training completed successfully for $benchmark." | tee -a "$LOG_FILE"
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
    echo "Starting cooling period for $duration seconds..." | tee -a "$LOG_FILE"
    echo "$(date): Starting cooling period" >> "$LOG_FILE"
    
    for i in $(seq $duration -10 0); do
        if [ $((i % 60)) -eq 0 ]; then
            echo "Cooling: $((i / 60)) minutes remaining..."
        fi
        sleep 10
    done
    
    echo "Cooling period complete." | tee -a "$LOG_FILE"
    echo "$(date): Cooling period complete" >> "$LOG_FILE"
}

# Function to check system temperature (placeholder - implement if you have a way to check)
check_temperature() {
    # This is a placeholder. On macOS, you might use a tool like 'osx-cpu-temp'
    # For now, we'll just return a fixed value
    echo "Checking system temperature..." | tee -a "$LOG_FILE"
    return 0  # Assume temperature is OK
}

# Train on each benchmark with cooling periods in between
# Expanded list of benchmarks to include more instances

# Taillard benchmarks (ta)
#BENCHMARKS=("ta01" "ta02" "ta03" "ta04" "ta05" "ta06" "ta07" "ta08" "ta09" "ta10")
#BENCHMARKS=("ta11" "ta12" "ta13" "ta14" "ta15" "ta16" "ta17" "ta18" "ta19" "ta20")
#BENCHMARKS=("ta21" "ta22" "ta23" "ta24" "ta25" "ta26" "ta27" "ta28" "ta29" "ta30")
#BENCHMARKS=("ta31" "ta32" "ta33" "ta34" "ta35" "ta36" "ta37" "ta38" "ta39" "ta40")
#BENCHMARKS=("ta41" "ta42" "ta43" "ta44" "ta45" "ta46" "ta47" "ta48" "ta49" "ta50")
#BENCHMARKS=("ta51" "ta52" "ta53" "ta54" "ta55" "ta56" "ta57" "ta58" "ta59" "ta60")
#BENCHMARKS=("ta61" "ta62" "ta63" "ta64" "ta65" "ta66" "ta67" "ta68" "ta69" "ta70")
#BENCHMARKS=("ta71" "ta72" "ta73" "ta74" "ta75" "ta76" "ta77" "ta78" "ta79" "ta80")

# Lawrence benchmarks (la)
#BENCHMARKS=("la01" "la02" "la03" "la04" "la05" "la06" "la07" "la08" "la09" "la10")
#BENCHMARKS=("la11" "la12" "la13" "la14" "la15" "la16" "la17" "la18" "la19" "la20")
#BENCHMARKS=("la21" "la22" "la23" "la24" "la25" "la26" "la27" "la28" "la29" "la30")
#BENCHMARKS=("la31" "la32" "la33" "la34" "la35" "la36" "la37" "la38" "la39" "la40")

# DMU benchmarks (dmu)
#BENCHMARKS=("dmu01" "dmu02" "dmu03" "dmu04" "dmu05" "dmu06" "dmu07" "dmu08" "dmu09" "dmu10")
#BENCHMARKS=("dmu11" "dmu12" "dmu13" "dmu14" "dmu15" "dmu16" "dmu17" "dmu18" "dmu19" "dmu20")
#BENCHMARKS=("dmu21" "dmu22" "dmu23" "dmu24" "dmu25" "dmu26" "dmu27" "dmu28" "dmu29" "dmu30")
#BENCHMARKS=("dmu31" "dmu32" "dmu33" "dmu34" "dmu35" "dmu36" "dmu37" "dmu38" "dmu39" "dmu40")
#BENCHMARKS=("dmu41" "dmu42" "dmu43" "dmu44" "dmu45" "dmu46" "dmu47" "dmu48" "dmu49" "dmu50")
#BENCHMARKS=("dmu51" "dmu52" "dmu53" "dmu54" "dmu55" "dmu56" "dmu57" "dmu58" "dmu59" "dmu60")
#BENCHMARKS=("dmu61" "dmu62" "dmu63" "dmu64" "dmu65" "dmu66" "dmu67" "dmu68" "dmu69" "dmu70")
#BENCHMARKS=("dmu71" "dmu72" "dmu73" "dmu74" "dmu75" "dmu76" "dmu77" "dmu78" "dmu79" "dmu80")

# Selected benchmarks
BENCHMARKS=("ta01" "ta02" "ta03" "ta11" "ta12" "ta21" "ta22")

SUCCESSFUL_BENCHMARKS=()
FAILED_BENCHMARKS=()

for ((i=0; i<${#BENCHMARKS[@]}; i++)); do
    benchmark="${BENCHMARKS[$i]}"
    
    # Check if we already have results for this benchmark
    if [ -f "$SAVE_DIR/${benchmark}_ppo_agent.pt" ] && [ -f "$RESULTS_FILE" ]; then
        if grep -q "\"$benchmark\"" "$RESULTS_FILE"; then
            echo "Benchmark $benchmark already trained. Skipping." | tee -a "$LOG_FILE"
            SUCCESSFUL_BENCHMARKS+=("$benchmark")
            continue
        fi
    fi
    
    # Check temperature before training
    check_temperature
    
    # Train on benchmark
    train_benchmark "$benchmark"
    
    # Record success or failure
    if [ $? -eq 0 ]; then
        SUCCESSFUL_BENCHMARKS+=("$benchmark")
    else
        FAILED_BENCHMARKS+=("$benchmark")
    fi
    
    # Skip cooling period after the last benchmark
    if [ $i -lt $((${#BENCHMARKS[@]} - 1)) ]; then
        cooling_period $COOLING_PERIOD
    fi
done

echo "All training complete!" | tee -a "$LOG_FILE"
echo "Results saved to $RESULTS_FILE" | tee -a "$LOG_FILE"
echo "$(date): All training complete" >> "$LOG_FILE"

# Print summary
echo "Training Summary:" | tee -a "$LOG_FILE"
echo "Successful benchmarks: ${SUCCESSFUL_BENCHMARKS[*]}" | tee -a "$LOG_FILE"
echo "Failed benchmarks: ${FAILED_BENCHMARKS[*]}" | tee -a "$LOG_FILE"

# Generate a comparison report if SelfLabelingJobShop results exist
if [ -f "converted_selflabeling_results.json" ]; then
    echo "Generating comparison report..." | tee -a "$LOG_FILE"
    
    python "jss_ppo_agent.py" \
        --benchmarks "${SUCCESSFUL_BENCHMARKS[@]}" \
        --compare \
        --model_dir "$SAVE_DIR" \
        --other_results "converted_selflabeling_results.json" \
        --report_file "comparison_report.txt" \
        --eval_episodes $EVAL_EPISODES
    
    echo "Comparison report saved to comparison_report.txt" | tee -a "$LOG_FILE"
fi 