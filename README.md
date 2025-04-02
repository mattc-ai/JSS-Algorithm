# Job Shop Scheduling with Enhanced Implementations

This repository contains a new model implementation for the Job Shop Scheduling problem, that combines Self-Labeling and Reinforcement Learning approaches.

## Setup Instructions

### Step 1: Clone the Required Original Repositories

First, clone the two original repositories:

```bash
# Clone the JSSEnv repository (environment for Job Shop Scheduling)
git clone https://github.com/prosysscience/JSSEnv.git

# Clone the SelfLabelingJobShop repository
git clone https://github.com/AndreaCorsini1/SelfLabelingJobShop.git
```

### Step 2: Install JSSEnv

Install the JSSEnv package which provides the gym environment for job shop scheduling:

```bash
# Navigate to the JSSEnv directory
cd JSSEnv

# Install the package
pip install -e .

# Return to the root directory
cd ..
```

This makes the JSSEnv environment available to your Python environment, which is required for the reinforcement learning components.

### Step 3: Replace the Files in SelfLabelingJobShop

Replace the original files with the enhanced versions provided in this repository:

```bash
# Copy the modified Python files from file_to_replace/SelfLabelingJobShop
cp file_to_replace/SelfLabelingJobShop/sampling.py SelfLabelingJobShop/
cp file_to_replace/SelfLabelingJobShop/test.py SelfLabelingJobShop/
cp file_to_replace/SelfLabelingJobShop/train.py SelfLabelingJobShop/
```

### Step 4: Install Dependencies

```bash
# Install required packages
pip install -r SelfLabelingJobShop/requirements.txt
```

## Using the Code

### Using HybridPPO

The HybridPPO approach combines self-labeling pretraining with PPO:

```bash
cd HybridPPO
./run_pretrain_standalone.sh
```

This script:
1. Creates a dataset subset for pretraining
2. Trains the neural networks using multiple heuristics
3. Saves the pretrained model to be used by PPO
   
Then to refine the model on a benchmark with reinforcement learning using the HybridPPO, on a single benchmark

```bash
cd HybridPPO/ppo
./run_hybrid_single.sh ta01  # Replace ta01 with your benchmark name
```

To run HybridPPO on the selected benchmarks:

```bash
cd OriginalPPO
./run_hybrid_comprehensive.sh
```

### Using OriginalPPO

To run the standard PPO approach on a single benchmark:

```bash
cd OriginalPPO
./run_single_benchmark.sh ta01  # Replace ta01 with your benchmark name
```

To run PPO on the selected benchmarks:

```bash
cd OriginalPPO
./run_all_benchmarks.sh
```

## Repository Structure

- **Root Directory**
  - `README.md`: This file
  - `file_to_replace/`: Contains files to replace in original repositories
    - `SelfLabelingJobShop/`: Modified files
      - `sampling.py`: Enhanced sampling implementation
      - `test.py`: Improved testing script
      - `train.py`: Updated training script
  - `HybridPPO/`: Hybrid approach implementation
  - `OriginalPPO/`: Standard PPO implementation  
