# OriginalPPO for Job Shop Scheduling

This directory contains a Proximal Policy Optimization (PPO) implementation for solving Job Shop Scheduling (JSS) problems.

## Overview

The PPO algorithm is a policy gradient method that uses clipped surrogate objectives to optimize a neural network policy for reinforcement learning tasks. This implementation applies PPO to job shop scheduling problems, where the goal is to schedule operations on machines to minimize the makespan.

## Directory Structure

- `train_single_benchmark.py`: Main training script for a single benchmark instance
- `run_single_benchmark.sh`: Shell script to run training on a specific benchmark
- `run_all_benchmarks.sh`: Shell script to run training on all available benchmarks
- `models/`: Neural network architecture definitions
- `utils/`: Utility functions and helper classes
- `ppo_models/`: Directory where trained models are saved

## Prerequisites

- Python 3.7+
- PyTorch 1.9+
- JSSEnv environment (the Job Shop Scheduling Gym environment)

## Usage

### Training on a Single Benchmark

```bash
./run_single_benchmark.sh <benchmark_name>
```

Example:
```bash
./run_single_benchmark.sh ta01
```

This will train a PPO agent on the specified benchmark and save the model to `ppo_models/<benchmark_name>_ppo_agent.pt`.

### Training on All Benchmarks

```bash
./run_all_benchmarks.sh
```

This will train PPO agents on all benchmark instances sequentially, saving each model to the `ppo_models/` directory.

## Training Parameters

The training script supports the following parameters (configured in the shell scripts):

- `--benchmark`: Name of the benchmark instance
- `--num_episodes`: Number of training episodes
- `--hidden_dim`: Hidden dimension size of neural networks
- `--lr`: Learning rate
- `--gamma`: Discount factor
- `--clip_ratio`: PPO clipping parameter
- `--value_coef`: Value loss coefficient
- `--entropy_coef`: Entropy bonus coefficient
- `--max_grad_norm`: Maximum gradient norm for clipping
- `--ppo_epochs`: Number of PPO optimization epochs per update
- `--batch_size`: Batch size for PPO updates
- `--gae_lambda`: GAE lambda parameter
- `--update_frequency`: Frequency of policy updates
- `--evaluation_frequency`: Frequency of evaluation during training
- `--max_steps_per_episode`: Maximum steps allowed per episode
- `--seed`: Random seed for reproducibility

## Progressive Training

This implementation supports progressive training to enhance learning:

- `--use_progressive_schedule`: Enable progressive increase in episode length
- `--initial_steps_per_episode`: Starting number of steps per episode
- `--steps_increment`: How much to increase steps by at each increment
- `--increment_interval`: How often to increase the number of steps

## Entropy Scheduling

To balance exploration and exploitation:

- `--auto_entropy_tuning`: Enable automatic entropy coefficient tuning
- `--entropy_decay_rate`: Rate at which entropy coefficient decays

## Results

Training results are saved to `ppo_models/ppo_results.json` and include:
- Training and validation metrics
- Best makespan achieved
- Comparison to known optimal solutions
- Training time statistics
