# HybridPPO for Job Shop Scheduling

This directory contains a hybrid approach that combines self-labeling pretraining with Proximal Policy Optimization (PPO) for solving Job Shop Scheduling problems more efficiently.

## Overview

The HybridPPO approach combines the strengths of:
1. **Self-Labeling**: Pretraining neural networks using generated heuristic solutions
2. **PPO**: Fine-tuning the pretrained networks using reinforcement learning

This hybrid approach can achieve better results faster than traditional PPO by leveraging domain knowledge through pretraining before reinforcement learning.

## Directory Structure

- `self_labeling_pretrain_standalone.py`: Implementation of the standalone pretraining step
- `run_pretrain_standalone.sh`: Shell script to run the pretraining process
- `models_v2/`: Neural network architecture definitions for the hybrid approach
- `utils/`: Utility functions, data processing, and helper classes
- `ppo/`: PPO implementation that uses the pretrained models
- `checkpoints/`: Directory where pretrained models are saved
- `hybrid_models/`: Directory where final hybrid models are saved
- `dataset5k_subset/`: Subset of JSS instances used for pretraining

## Prerequisites

- Python 3.7+
- PyTorch 1.9+
- SelfLabelingJobShop repository (for the sampling module)
- JSSEnv environment (the Job Shop Scheduling Gym environment)

## Usage

### Pretraining Step

```bash
./run_pretrain_standalone.sh
```

This script:
1. Creates a subset of the dataset for pretraining
2. Trains the neural networks to imitate solutions from multiple heuristics
3. Saves the pretrained model to `checkpoints/pretrained_model_standalone.pt`

### Configuration

The pretraining script supports the following parameters (configured in the shell script):

- `--data_path`: Path to the dataset for pretraining
- `--val_path`: Path to validation set
- `--model_path`: Where to save the pretrained model
- `--num_instances`: Number of instances to use per epoch
- `--iterations`: Number of iterations over each instance
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--epochs`: Number of training epochs
- `--hidden_dim`: Hidden dimension size of neural networks
- `--embed_dim`: Embedding dimension size
- `--num_heads`: Number of attention heads
- `--dropout`: Dropout rate
- `--seed`: Random seed for reproducibility

## Technical Details

### Pretraining Process

1. **Heuristic Solution Generation**: Multiple heuristics (SPT, MWR, FCFS) are used to generate diverse solutions
2. **Network Architecture**: Uses a Graph Attention Network (GAT) encoder and Multi-Head Attention (MHA) decoder
3. **Learning Objective**: The networks learn to predict the next job to schedule based on the current state

### Integration with PPO

The pretrained networks provide:
1. A better initialization for the policy network
2. Enhanced feature extraction capabilities
3. Domain-specific knowledge of job shop scheduling

This gives the PPO algorithm a significant head start compared to random initialization.

## Results

Training logs are saved to `hybrid_training_log.txt` and contain:
- Pretraining loss metrics
- Validation performance
- Comparison to baseline heuristics

The pretrained models consistently help PPO converge faster and to better solutions than training from scratch. 