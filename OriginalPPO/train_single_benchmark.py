#!/usr/bin/env python3
"""
Script to train a PPO agent on a single benchmark.
This helps avoid memory issues and allows for better monitoring of progress.
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib
import random
# Use the 'Agg' backend which doesn't require a display
matplotlib.use('Agg')

# Add the parent directory to the path so we can import from models
# Get the directory where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to the path
sys.path.append(os.path.dirname(script_dir))

# Now import the PPOAgent from models directly
from models.jss_ppo_agent import PPOAgent
import gym
import JSSEnv
import time
import traceback

# Import gap calculation utilities
from utils.best_known_solutions import get_best_known_solution, calculate_optimality_gap

def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def train_single_benchmark(benchmark_name, num_episodes=500, hidden_dim=64, 
                          lr=3e-4, gamma=0.99, 
                          clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01,
                          max_grad_norm=0.5, ppo_epochs=10, batch_size=64,
                          gae_lambda=0.95, update_frequency=10, 
                          evaluation_frequency=20, 
                          use_progressive_schedule=True, initial_steps_per_episode=1000,
                          steps_increment=500, increment_interval=100,
                          auto_entropy_tuning=True, entropy_decay_rate=0.7,
                          eval_episodes=10, save_dir='ppo_models', 
                          results_file=None, resume=False, max_steps_per_episode=3000,
                          seed=None):
    """
    Train a PPO agent on a single benchmark.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., 'ta01')
        num_episodes: Number of episodes to train for
        hidden_dim: Hidden dimension for neural networks
        lr: Learning rate
        gamma: Discount factor
        clip_ratio: PPO clipping parameter
        value_coef: Value loss coefficient
        entropy_coef: Entropy coefficient
        max_grad_norm: Maximum gradient norm for clipping
        ppo_epochs: Number of PPO epochs per update
        batch_size: Batch size for updates
        gae_lambda: GAE lambda parameter
        update_frequency: Frequency of policy updates (in episodes)
        evaluation_frequency: How often to evaluate and save model (in episodes)
        use_progressive_schedule: Whether to gradually increase episode length
        initial_steps_per_episode: Starting episode length for progressive training
        steps_increment: How much to increase steps per episode each time
        increment_interval: How often to increase episode length (in episodes)
        auto_entropy_tuning: Whether to decrease entropy coefficient over time
        entropy_decay_rate: Rate to multiply entropy coefficient at each adjustment
        eval_episodes: Number of episodes to evaluate for
        save_dir: Directory to save trained models
        results_file: Path to save results as JSON
        resume: Whether to resume training from a saved model
        max_steps_per_episode: Maximum steps per episode
        seed: Random seed for reproducibility
    
    Returns:
        Evaluation results for the benchmark
    """
    # Set random seeds if provided
    if seed is not None:
        set_random_seeds(seed)
        print(f"Set random seed: {seed}")
    
    # Convert relative paths to absolute paths if they aren't already
    if not os.path.isabs(save_dir):
        # Get the absolute path to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, save_dir)
        print(f"Using absolute save directory: {save_dir}")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert results_file to absolute path if provided
    if results_file and not os.path.isabs(results_file):
        if os.path.dirname(results_file):
            # If results_file includes a directory part
            results_file = os.path.join(script_dir, results_file)
        else:
            # If results_file is just a filename, put it in save_dir
            results_file = os.path.join(save_dir, results_file)
        print(f"Using absolute results file path: {results_file}")
    
    print(f"\n{'='*50}")
    print(f"Training on benchmark: {benchmark_name}")
    print(f"{'='*50}\n")
    
    try:
        # Get the absolute path to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Try multiple locations for the benchmark file
        instance_paths = [
            # Try the installed package
            os.path.join(os.path.dirname(JSSEnv.__file__), 'envs', 'instances', benchmark_name),
            # Try the local repository with absolute path
            os.path.join(project_root, 'JSSEnv', 'JSSEnv', 'envs', 'instances', benchmark_name)
        ]
        
        # Find the first path that exists
        instance_path = None
        for path in instance_paths:
            if os.path.exists(path):
                instance_path = path
                print(f"Using benchmark file: {path}")
                break
        
        # Check if a valid path was found
        if instance_path is None:
            print(f"Error: Benchmark file for {benchmark_name} not found in any of these locations:")
            for path in instance_paths:
                print(f"  - {path}")
            return None
        
        # Create environment
        env = gym.make('jss-v1', env_config={'instance_path': instance_path})
        
        # Create agent with all parameters
        agent = PPOAgent(
            env, 
            hidden_dim=hidden_dim, 
            lr=lr,
            gamma=gamma,
            clip_ratio=clip_ratio,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs,
            batch_size=batch_size,
            gae_lambda=gae_lambda,
            max_steps_per_episode=max_steps_per_episode,
            update_frequency=update_frequency,
            evaluation_frequency=evaluation_frequency,
            use_progressive_schedule=use_progressive_schedule,
            initial_steps_per_episode=initial_steps_per_episode,
            steps_increment=steps_increment,
            increment_interval=increment_interval,
            auto_entropy_tuning=auto_entropy_tuning,
            entropy_decay_rate=entropy_decay_rate
        )
        
        # Model path for saving/loading
        model_path = os.path.join(save_dir, f"{benchmark_name}_ppo_agent.pt")
        
        # Resume training if requested
        if resume and os.path.exists(model_path):
            print(f"Resuming training from {model_path}")
            try:
                agent.load(model_path)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting with a fresh model.")
        
        # Train agent
        print(f"Training agent for {num_episodes} episodes")
        start_time = time.time()
        
        try:
            agent.train(num_episodes, update_frequency=update_frequency)
            print("Training completed successfully.")
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
            # Save the model even if training fails
            print(f"Saving partial model to {model_path}")
            agent.save(model_path)
            raise
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save agent
        print(f"Saving agent to {model_path}")
        agent.save(model_path)
        
        # Plot training curves
        curves_path = os.path.join(save_dir, f"{benchmark_name}_training_curves.png")
        agent.plot_training_curves(save_path=curves_path)
        
        # Extract training statistics
        training_makespans = agent.training_log['episode_makespans']
        training_rewards = agent.training_log['episode_rewards']
        training_decisions = agent.training_log['episode_decisions']
        
        # Calculate training statistics
        training_min_makespan = min(training_makespans) if training_makespans else float('inf')
        training_avg_makespan = sum(training_makespans) / len(training_makespans) if training_makespans else 0
        training_std_makespan = np.std(training_makespans) if training_makespans else 0
        training_avg_reward = sum(training_rewards) / len(training_rewards) if training_rewards else 0
        training_avg_decisions = sum(training_decisions) / len(training_decisions) if training_decisions else 0
        
        # Evaluate agent
        print(f"Evaluating agent for {eval_episodes} episodes")
        eval_results = agent.evaluate(eval_episodes)
        
        # Try to calculate optimality gap based on best-known solution for this benchmark
        try:
            # First try to import from HybridPPO/utils - this is the most reliable source
            if not 'calculate_optimality_gap' in locals() or not 'get_best_known_solution' in locals():
                try:
                    from HybridPPO.utils.benchmark_gap_bksol import calculate_optimality_gap, get_best_known_solution
                except ImportError:
                    # Fall back to local utils if available
                    try:
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from utils import get_best_known_solution as gbs, calculate_gap

                        # Create wrapper functions to match the HybridPPO API
                        def get_best_known_solution(benchmark):
                            return gbs(benchmark)

                        def calculate_optimality_gap(makespan, benchmark):
                            return calculate_gap(makespan, get_best_known_solution(benchmark))
                    except ImportError:
                        print("Gap calculation utilities not found. Skipping gap calculation.")
                        gap = None
            
            # Get the best-known solution for this benchmark (if available)
            best_known = get_best_known_solution(benchmark_name)
            
            if best_known:
                # Calculate gap based on minimum makespan achieved in evaluation
                min_makespan = eval_results['min_makespan']
                gap = calculate_optimality_gap(min_makespan, benchmark_name)
                print(f"Best known solution for {benchmark_name}: {best_known}")
                print(f"Minimum makespan achieved: {min_makespan}")
                print(f"Gap to best known solution: {gap:.2f}%")
            else:
                gap = None
                print(f"No best known solution found for {benchmark_name}")
        except Exception as e:
            print(f"Error calculating gap: {e}")
            traceback.print_exc()
            gap = None
        
        # Create a new combined results format
        combined_results = {
            'benchmark': benchmark_name,
            'min_makespan': convert_to_serializable(training_min_makespan),
            'avg_makespan': convert_to_serializable(training_avg_makespan),
            'std_makespan': convert_to_serializable(training_std_makespan),
            'avg_reward': convert_to_serializable(training_avg_reward),
            'avg_decisions': convert_to_serializable(training_avg_decisions),
            'total_training_time': convert_to_serializable(training_time),
            'evaluation_min_makespan': convert_to_serializable(eval_results['min_makespan']),
            'evaluation_avg_makespan': convert_to_serializable(eval_results['avg_makespan']),
            'evaluation_std_makespan': convert_to_serializable(eval_results['std_makespan']),
            'evaluation_avg_reward': convert_to_serializable(eval_results['avg_reward']),
            'evaluation_avg_decisions': convert_to_serializable(eval_results['avg_decisions']),
            'evaluation_avg_time': convert_to_serializable(eval_results['avg_time'])
        }
        
        # Add gap if available
        if gap is not None:
            combined_results['gap'] = f"{gap:.2f}%"
        
        # Save results to JSON file if specified
        if results_file:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)
            
            # Try to load existing results
            existing_results = {}
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        existing_results = json.load(f)
                        # Remove lengths and avg_length from existing results if present
                        for benchmark in existing_results:
                            if 'lengths' in existing_results[benchmark]:
                                del existing_results[benchmark]['lengths']
                            if 'avg_length' in existing_results[benchmark]:
                                del existing_results[benchmark]['avg_length']
                except json.JSONDecodeError:
                    print(f"Error loading existing results from {results_file}. Creating new file.")
            
            # Update existing results with new ones
            existing_results[benchmark_name] = combined_results
            
            # Save updated results
            with open(results_file, 'w') as f:
                json.dump(existing_results, f, indent=4, default=convert_to_serializable)
            
            print(f"Results saved to {results_file}")
        
        # Return combined results
        return combined_results
    
    except Exception as e:
        print(f"Error training on benchmark {benchmark_name}: {e}")
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Train PPO agent on a single benchmark')
    parser.add_argument('--benchmark', type=str, required=True,
                      help='Name of the benchmark to train on (e.g., ta01)')
    parser.add_argument('--num_episodes', type=int, default=500,
                      help='Number of episodes to train for')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Hidden dimension for neural networks')
    parser.add_argument('--lr', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                      help='PPO clipping parameter')
    parser.add_argument('--value_coef', type=float, default=0.5,
                      help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                      help='Entropy coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                      help='Maximum gradient norm for clipping')
    parser.add_argument('--ppo_epochs', type=int, default=10,
                      help='Number of PPO epochs per update')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for updates')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                      help='GAE lambda parameter')
    parser.add_argument('--update_frequency', type=int, default=10,
                      help='Frequency of policy updates (in episodes)')
    parser.add_argument('--evaluation_frequency', type=int, default=20,
                      help='How often to evaluate and save model (in episodes)')
    parser.add_argument('--use_progressive_schedule', action='store_true',
                      help='Whether to gradually increase episode length')
    parser.add_argument('--initial_steps_per_episode', type=int, default=1000,
                      help='Starting episode length for progressive training')
    parser.add_argument('--steps_increment', type=int, default=500,
                      help='How much to increase steps per episode each time')
    parser.add_argument('--increment_interval', type=int, default=100,
                      help='How often to increase episode length (in episodes)')
    parser.add_argument('--auto_entropy_tuning', action='store_true',
                      help='Whether to decrease entropy coefficient over time')
    parser.add_argument('--entropy_decay_rate', type=float, default=0.7,
                      help='Rate to multiply entropy coefficient at each adjustment')
    parser.add_argument('--eval_episodes', type=int, default=10,
                      help='Number of episodes to evaluate for')
    parser.add_argument('--save_dir', type=str, default='ppo_models',
                      help='Directory to save trained models')
    parser.add_argument('--results_file', type=str, default='ppo_models/ppo_results.json',
                      help='Path to save results as JSON')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from a saved model')
    parser.add_argument('--max_steps_per_episode', type=int, default=3000,
                      help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Train on the benchmark
    result = train_single_benchmark(
        benchmark_name=args.benchmark,
        num_episodes=args.num_episodes,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        gae_lambda=args.gae_lambda,
        update_frequency=args.update_frequency,
        evaluation_frequency=args.evaluation_frequency,
        use_progressive_schedule=args.use_progressive_schedule,
        initial_steps_per_episode=args.initial_steps_per_episode,
        steps_increment=args.steps_increment,
        increment_interval=args.increment_interval,
        auto_entropy_tuning=args.auto_entropy_tuning,
        entropy_decay_rate=args.entropy_decay_rate,
        eval_episodes=args.eval_episodes,
        save_dir=args.save_dir,
        results_file=args.results_file,
        resume=args.resume,
        max_steps_per_episode=args.max_steps_per_episode,
        seed=args.seed
    )
    
    # Exit with appropriate status code
    if result is None:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main() 
