#!/usr/bin/env python3
"""
Hybrid PPO Agent for Job Shop Scheduling.

This file implements a Hybrid PPO approach that combines:
1. Graph neural networks from self-labeling for structural understanding
2. Reinforcement learning with PPO for sequential decision optimization

The hybrid approach leverages pretrained graph knowledge from self-labeling
to enhance the policy and value networks in the PPO framework.
"""

import os
import time
import argparse
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import JSSEnv
from pathlib import Path
import matplotlib.pyplot as plt
import json
import sys
import random
from tqdm import tqdm
import copy
import dill

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import GraphAttentionNetwork - handle both module import and direct script execution
try:
    # Try relative import first (when imported as a module)
    from .gat import GraphAttentionNetwork
except ImportError:
    # Fall back to local import when executed as a script
    from gat import GraphAttentionNetwork

from models_v2.mha import MultiHeadAttention

# Import gap calculation utilities
from utils.benchmark_gap_bksol import get_best_known_solution, calculate_optimality_gap

# Set gap calculation as available
gap_calculation_available = True

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
def set_seed(seed):
    """Set seed for all random number generators."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class GNNFeatureExtractor(nn.Module):
    """
    Feature extractor using Graph Neural Networks for the JSSEnv observation space.
    
    This extractor utilizes pretrained knowledge from self-labeling to
    process job shop scheduling observations.
    """
    def __init__(self, obs_space, hidden_dim=64, use_gnn=True):
        super(GNNFeatureExtractor, self).__init__()
        
        # Get dimensions from observation space
        self.jobs = obs_space["real_obs"].shape[0]
        self.features_per_job = obs_space["real_obs"].shape[1]
        self.action_mask_dim = obs_space["action_mask"].shape[0]
        
        # Calculate input dimension
        self.input_dim = self.jobs * self.features_per_job
        
        # Flag to determine whether to use GNN or MLP
        self.use_gnn = use_gnn
        
        if use_gnn:
            # Graph Neural Network for structured feature extraction
            # Use dimensions to match the pretrained model exactly
            gnn_hidden_dim = 64      # Hidden dimension
            gnn_out_dim = 128        # Output dimension for each head
            
            # Initialize the GNN with forced pretrained dimensions to ensure compatibility
            # Regardless of input size, the network will adapt internally
            self.gnn = GraphAttentionNetwork(
                in_features=15,       # Fixed at 15 for pretrained model compatibility
                hidden_dim=gnn_hidden_dim,
                out_features=gnn_out_dim,
                num_heads=3,
                dropout=0.15,
                concat=True,
                second_layer_concat=True,  # Use concat in second layer for pretrained model
                residual=False,
                force_pretrained_dim=True  # Force the network to use pretrained dimensions
            )
            
            # Expected output size: 15 + 128 = 143 (based on second layer having 1 head as in pretrained model)
            expected_output_size = 143
            
            # Create projection layer for the GNN output
            self.projection = nn.Sequential(
                nn.Linear(expected_output_size, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            )
        else:
            # Standard MLP feature extractor (used if GNN knowledge isn't applied)
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            )
        
        # Output dimension
        self.output_dim = hidden_dim
    
    def forward(self, obs_dict):
        """Extract features using GNN or MLP based on configuration"""
        # Process observation dictionary
        if isinstance(obs_dict, dict):
            real_obs = obs_dict["real_obs"]
            legal_actions = obs_dict["action_mask"]
        else:
            # Handle legacy observation format
            real_obs = obs_dict
            legal_actions = None
            
        batch_size = real_obs.shape[0] if len(real_obs.shape) > 2 else 1
        
        # Convert observation to appropriate dimensions
        if self.use_gnn:
            # Extract node features directly from observation
            node_features = real_obs
            
            # Reshape for GNN processing
            if len(node_features.shape) == 3:  # Batched data
                # Reshape to 2D for GNN processing
                original_shape = node_features.shape
                flat_nodes = node_features.reshape(batch_size * original_shape[1], original_shape[2])
            else:
                # Single observation (non-batched)
                original_shape = node_features.shape
                flat_nodes = node_features.reshape(-1, original_shape[1])
            
            # Create a fully-connected graph (all nodes connected to all others)
            num_nodes = flat_nodes.shape[0]
            edge_index = torch.stack([
                torch.repeat_interleave(torch.arange(num_nodes, device=flat_nodes.device), num_nodes),
                torch.tile(torch.arange(num_nodes, device=flat_nodes.device), (num_nodes,))
            ])
            
            # Process through the GNN
            gnn_output = self.gnn(flat_nodes, edge_index)
            
            # Process through projection layers
            features = self.projection(gnn_output)
            
            # Reshape back to batch format if needed
            if len(real_obs.shape) == 3:
                features = features.reshape(batch_size, -1, features.shape[1])
                features = features.mean(dim=1)  # Pool over nodes
        else:
            # Use MLP approach for feature extraction
            if len(real_obs.shape) == 3:
                # Flatten batched observations
                flat_obs = real_obs.reshape(batch_size, -1)
            else:
                # Already flat or single observation
                flat_obs = real_obs.reshape(1, -1) if len(real_obs.shape) == 2 else real_obs
                
            # Pass through MLP feature extractor
            features = self.mlp(flat_obs)
        
        return features, legal_actions

class HybridPPOPolicy(nn.Module):
    """
    Hybrid PPO Policy Network for Job Shop Scheduling.
    
    Combines graph neural network knowledge with traditional PPO policy.
    Includes both actor (policy) and critic (value) networks.
    """
    def __init__(self, obs_space, action_space, hidden_dim=64, use_gnn=True):
        super(HybridPPOPolicy, self).__init__()
        
        # Feature extractor (either GNN-based or standard)
        self.feature_extractor = GNNFeatureExtractor(obs_space, hidden_dim, use_gnn=use_gnn)
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(self.feature_extractor.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space.n)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(self.feature_extractor.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs_dict):
        """
        Forward pass through both actor and critic networks.
        
        Args:
            obs_dict: Observation dictionary from environment
            
        Returns:
            Action logits, action mask, and state value
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(obs_dict["real_obs"], np.ndarray):
            real_obs = torch.FloatTensor(obs_dict["real_obs"]).to(device)
            action_mask = torch.FloatTensor(obs_dict["action_mask"]).to(device)
            obs_dict = {"real_obs": real_obs, "action_mask": action_mask}
        
        # Extract features using the feature extractor
        features, action_mask = self.feature_extractor(obs_dict)
        
        # Get action logits and state value
        action_logits = self.actor(features)
        state_value = self.critic(features)
        
        return action_logits, action_mask, state_value
    
    def get_action(self, obs_dict, deterministic=False):
        """
        Get an action from the policy.
        
        Args:
            obs_dict: Observation dictionary from environment
            deterministic: Whether to select the action deterministically
            
        Returns:
            Action, log probability, and state value
        """
        # Convert observation to tensor
        if isinstance(obs_dict["real_obs"], np.ndarray):
            real_obs = torch.FloatTensor(obs_dict["real_obs"]).unsqueeze(0).to(device)
        else:
            real_obs = obs_dict["real_obs"].unsqueeze(0).to(device)
            
        if isinstance(obs_dict["action_mask"], np.ndarray):
            action_mask = torch.FloatTensor(obs_dict["action_mask"]).unsqueeze(0).to(device)
        else:
            action_mask = obs_dict["action_mask"].unsqueeze(0).to(device)
        
        # Create observation dictionary
        obs_tensor = {"real_obs": real_obs, "action_mask": action_mask}
        
        # Forward pass
        with torch.no_grad():  # Use no_grad to prevent tracking computation graph
            action_logits, action_mask, state_value = self.forward(obs_tensor)
        
        # Apply action mask - set logits of invalid actions to a large negative value
        masked_logits = action_logits.clone()
        masked_logits[action_mask < 0.5] = -1e10  # Set invalid actions to very negative value
        
        # Create categorical distribution
        action_probs = torch.softmax(masked_logits, dim=-1)
        
        # Ensure there are valid actions
        if torch.sum(action_mask) == 0:
            print("Warning: No valid actions available in the environment!")
            # Return a dummy action (this should never happen with proper environment setup)
            return torch.tensor(0, device=device), torch.tensor([0.0], device=device), state_value
        
        if deterministic:
            # Choose the action with highest probability
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from the distribution for exploration
            dist = Categorical(action_probs)
            action = dist.sample()
        
        # Get log probability
        log_prob = torch.log(action_probs.squeeze(0)[action])
        
        return action, log_prob.unsqueeze(0), state_value
    
    def evaluate_actions(self, obs_dict, actions):
        """
        Evaluate actions for PPO update.
        
        Args:
            obs_dict: Observation dictionary from environment
            actions: Actions to evaluate
            
        Returns:
            Log probabilities, state values, and entropy
        """
        # Forward pass
        action_logits, action_mask, state_value = self.forward(obs_dict)
        
        # Apply action mask - set logits of invalid actions to a large negative value
        masked_logits = action_logits.clone()
        masked_logits[action_mask < 0.5] = -1e10  # Set invalid actions to very negative value
        
        # Create categorical distribution
        action_probs = torch.softmax(masked_logits, dim=-1)
        
        try:
            # Create distribution
            dist = Categorical(action_probs)
            
            # Get log probabilities and entropy
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            return log_probs, state_value, entropy
            
        except Exception as e:
            print(f"Error in evaluate_actions: {e}")
            # Return zeros as fallback (this should rarely happen)
            log_probs = torch.zeros_like(actions, dtype=torch.float32).to(device)
            entropy = torch.tensor(0.0).to(device)
            return log_probs, state_value, entropy 

class HybridPPOAgent:
    """
    Hybrid PPO Agent for Job Shop Scheduling.
    
    Integrates graph neural network knowledge with PPO algorithm for JSS problems.
    """
    def __init__(self, env, hidden_dim=64, lr=3e-4, gamma=0.99, clip_ratio=0.2, value_coef=0.5,
                 entropy_coef=0.01, max_grad_norm=0.5, ppo_epochs=10, batch_size=64,
                 gae_lambda=0.95, update_frequency=10, evaluation_frequency=20,
                 use_progressive_schedule=False, initial_steps_per_episode=1000,
                 steps_increment=500, increment_interval=100, auto_entropy_tuning=False,
                 entropy_decay_rate=0.7, max_steps_per_episode=3000,
                 use_gnn=True, pretrained_model_path=None, gnn_lr_factor=0.1,
                 results_dir='hybrid_models'):
        """
        Initialize the Hybrid PPO agent.
        
        Args:
            env: Environment to train on
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
            max_steps_per_episode: Maximum steps per episode
            use_gnn: Whether to use GNN features in the policy
            pretrained_model_path: Path to pretrained model from self-labeling
            gnn_lr_factor: Factor to reduce learning rate for pretrained parts
            results_dir: Directory to save results
        """
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.initial_entropy_coef = entropy_coef  # Store the initial value
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.update_frequency = update_frequency
        self.evaluation_frequency = evaluation_frequency
        self.max_steps_per_episode = max_steps_per_episode
        self.use_gnn = use_gnn
        self.pretrained_model_path = pretrained_model_path
        self.gnn_lr_factor = gnn_lr_factor
        self.results_dir = results_dir
        
        # Create results directory
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        # Progressive scheduling parameters
        self.use_progressive_schedule = use_progressive_schedule
        self.initial_steps_per_episode = initial_steps_per_episode
        self.current_steps_per_episode = initial_steps_per_episode if use_progressive_schedule else max_steps_per_episode
        self.steps_increment = steps_increment
        self.increment_interval = increment_interval
        self.episode_counter = 0
        
        # Entropy tuning parameters
        self.auto_entropy_tuning = auto_entropy_tuning
        self.entropy_decay_rate = entropy_decay_rate
        
        # Update counter
        self.update_count = 0
        
        # Create policy network
        self.policy = HybridPPOPolicy(
            env.observation_space,
            env.action_space,
            hidden_dim=hidden_dim,
            use_gnn=use_gnn
        )
        
        # Load pretrained weights if provided
        if pretrained_model_path is not None and use_gnn:
            self._load_pretrained_gnn(pretrained_model_path)
        
        # Set up optimizer with different learning rates
        if use_gnn and pretrained_model_path is not None:
            # Use different learning rates for pretrained and new parts
            pretrained_params = []
            new_params = []
            
            # Identify pretrained parameters (feature extractor gnn) and new parameters
            for name, param in self.policy.named_parameters():
                if 'feature_extractor.gnn' in name:
                    pretrained_params.append(param)
                else:
                    new_params.append(param)
            
            # Create parameter groups with different learning rates
            param_groups = [
                {'params': pretrained_params, 'lr': lr * gnn_lr_factor},  # Lower LR for pretrained
                {'params': new_params, 'lr': lr}                           # Normal LR for new parts
            ]
            
            self.optimizer = optim.Adam(param_groups)
            print(f"Using differential learning rates: {lr * gnn_lr_factor} for GNN, {lr} for other parts")
        else:
            # Use same learning rate for all parameters
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
            print(f"Using uniform learning rate of {lr} for all parameters")
        
        # Initialize storage
        self.reset_storage()
        
        # Initialize training log
        self.training_log = {
            'episode_rewards': [],
            'episode_decisions': [],
            'episode_makespans': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropy_losses': [],
            'total_losses': [],
            'learning_rates': [],
            'entropy_coeffs': [],
            'steps_per_episode': [],
            'total_training_time': 0
        }
        
        # Extract benchmark name for file naming
        try:
            if hasattr(env, 'env_config') and 'instance_path' in env.env_config:
                # Extract filename from the full path
                instance_path = env.env_config['instance_path']
                self.benchmark_name = os.path.basename(instance_path)
            else:
                # Fallback to environment ID
                env_id = env.spec.id
                if ':' in env_id:
                    self.benchmark_name = env_id.split(':')[-1].split('-')[1]
                elif '-' in env_id:
                    self.benchmark_name = env_id.split('-')[1]
                else:
                    self.benchmark_name = env_id
        except Exception as e:
            self.benchmark_name = "unknown"
    
    def _load_pretrained_gnn(self, pretrained_model_path):
        """Load pretrained GNN model from path"""
        try:
            # Add numpy types to safe globals for PyTorch 2.6+
            try:
                import torch.serialization
                torch.serialization.add_safe_globals([
                    np.dtype, np.bool_, np.int_, np.intc, np.intp,
                    np.int8, np.int16, np.int32, np.int64, np.uint8,
                    np.uint16, np.uint32, np.uint64, np.float16, np.float32,
                    np.float64, np.complex64, np.complex128, np.array, np.zeros,
                    np.ndarray, np.ndindex, np.empty
                ])
            except (ImportError, AttributeError):
                pass
                
            # Load pretrained model with weights_only=False (since it's trusted)
            checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=False)
            
            # Extract GNN encoder state dict (expect it in a specific format)
            if 'encoder_state_dict' in checkpoint:
                encoder_state_dict = checkpoint['encoder_state_dict']
                
                # Collect appropriate GNN keys
                gnn_keys = {}
                for key, value in encoder_state_dict.items():
                    if 'embedding1' in key or 'embedding2' in key:
                        gnn_keys[key] = value
                
                # Directly load weights
                unexpected_keys = self.policy.feature_extractor.gnn.load_state_dict(gnn_keys, strict=False)
                print("GNN weights successfully loaded!")
            else:
                print("No encoder_state_dict found in pretrained model")
                return False
        
            return True
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            return False
    
    def reset_storage(self):
        """Reset storage for a new trajectory."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        """Store a transition in the trajectory storage."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        # Detach value to prevent backward through the graph a second time
        self.values.append(value.detach())
        self.dones.append(done)
    
    def compute_returns_and_advantages(self):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        
        Returns:
            returns: Estimated returns for each step
            advantages: Estimated advantages for each step
        """
        # Convert rewards and values to tensors
        rewards = torch.FloatTensor(self.rewards).to(device)
        values = torch.cat(self.values).to(device)
        dones = torch.FloatTensor(self.dones).to(device)
        
        # Set up storage
        returns = torch.zeros_like(rewards).to(device)
        advantages = torch.zeros_like(rewards).to(device)
        
        # Initialize next value and advantage for bootstrapping
        next_value = 0
        next_advantage = 0
        
        # Compute returns and advantages in reverse order (from last to first time step)
        for t in reversed(range(len(rewards))):
            # Compute return (bootstrapped value)
            returns[t] = rewards[t] + self.gamma * next_value * (1 - dones[t])
            
            # Compute TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # Compute GAE advantage
            advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage * (1 - dones[t])
            
            # Update for next iteration
            next_value = values[t]
            next_advantage = advantages[t]
        
        return returns, advantages
    
    def update(self):
        """Update the policy using PPO."""
        try:
            # Compute returns and advantages
            returns, advantages = self.compute_returns_and_advantages()
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Convert trajectory data to tensors
            old_states = self.states
            old_actions = torch.LongTensor(self.actions).to(device)
            old_log_probs = torch.cat(self.log_probs).to(device)
            
            # PPO update
            actor_losses = []
            critic_losses = []
            entropy_losses = []
            total_losses = []
            
            # Perform multiple epochs of updates
            for _ in range(self.ppo_epochs):
                # Create mini-batches
                batch_indices = np.random.permutation(len(old_actions))
                
                for start_idx in range(0, len(old_actions), self.batch_size):
                    # Get mini-batch indices
                    batch_idx = batch_indices[start_idx:start_idx + self.batch_size]
                    
                    # Extract mini-batch data
                    batch_states = [old_states[i] for i in batch_idx]
                    batch_actions = old_actions[batch_idx]
                    batch_log_probs = old_log_probs[batch_idx]
                    batch_returns = returns[batch_idx]
                    batch_advantages = advantages[batch_idx]
                    
                    # Create batch observation dictionary
                    batch_real_obs = torch.FloatTensor(np.stack([s["real_obs"] for s in batch_states])).to(device)
                    batch_action_mask = torch.FloatTensor(np.stack([s["action_mask"] for s in batch_states])).to(device)
                    batch_obs_dict = {"real_obs": batch_real_obs, "action_mask": batch_action_mask}
                    
                    # Evaluate actions
                    new_log_probs, state_values, entropy = self.policy.evaluate_actions(batch_obs_dict, batch_actions)
                    
                    # Calculate ratios
                    ratios = torch.exp(new_log_probs - batch_log_probs)
                    
                    # Calculate surrogate losses
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                    
                    # Calculate losses
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = self.value_coef * torch.mean((batch_returns - state_values.squeeze()) ** 2)
                    entropy_loss = -self.entropy_coef * entropy.mean()
                    
                    # Total loss
                    loss = actor_loss + critic_loss + entropy_loss
                    
                    # Optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    
                    self.optimizer.step()
                    
                    # Store losses for logging
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    total_losses.append(loss.item())
            
            # Log losses
            self.training_log['actor_losses'].append(np.mean(actor_losses))
            self.training_log['critic_losses'].append(np.mean(critic_losses))
            self.training_log['entropy_losses'].append(np.mean(entropy_losses))
            self.training_log['total_losses'].append(np.mean(total_losses))
            
            # Log learning rates
            param_groups = self.optimizer.param_groups
            if len(param_groups) > 1:
                self.training_log['learning_rates'].append([
                    param_groups[0]['lr'],  # GNN learning rate
                    param_groups[1]['lr']   # Policy learning rate
                ])
            else:
                self.training_log['learning_rates'].append(param_groups[0]['lr'])
            
            # Log entropy coefficient
            self.training_log['entropy_coeffs'].append(self.entropy_coef)
            
            self.update_count += 1
            
            # Adjust entropy coefficient if auto-tuning is enabled
            if self.auto_entropy_tuning and self.update_count % 10 == 0:
                self.entropy_coef = max(self.entropy_coef * self.entropy_decay_rate, 1e-4)
                
            return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropy_losses), np.mean(total_losses)
            
        except Exception as e:
            print(f"Error during policy update: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0, 0, 0 

    def train(self, num_episodes, update_frequency=None):
        """
        Train the agent for a specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to train for
            update_frequency: How often to update the policy (in episodes)
            
        Returns:
            The trained policy and training log
        """
        if update_frequency is not None:
            self.update_frequency = update_frequency
        
        print(f"\nStarting training for {num_episodes} episodes...")
        print(f"Update frequency: {self.update_frequency} episodes")
        print(f"Using {'GNN-enhanced' if self.use_gnn else 'standard MLP'} feature extractor")
        
        # Training loop
        episode_counter = 0
        running_reward = 0
        
        # Track total training time
        total_training_start_time = time.time()
        
        for episode in range(1, num_episodes + 1):
            # Reset environment and trajectory storage
            state = self.env.reset()
            self.reset_storage()
            
            # Tracking variables for this episode
            episode_reward = 0
            episode_decisions = 0
            episode_start_time = time.time()
            
            # Increment internal episode counter for progressive scheduling
            self.episode_counter += 1
            
            # Update episode length if using progressive schedule
            if self.use_progressive_schedule and self.episode_counter % self.increment_interval == 0:
                self.current_steps_per_episode = min(
                    self.current_steps_per_episode + self.steps_increment,
                    self.max_steps_per_episode
                )
                print(f"Increased episode length to {self.current_steps_per_episode} decisions")
            
            # Log current steps per episode
            self.training_log['steps_per_episode'].append(self.current_steps_per_episode)
            
            # Episode loop
            done = False
            steps = 0
            
            while not done and steps < self.current_steps_per_episode:
                # Select action
                action, log_prob, value = self.policy.get_action(state)
                
                # Ensure action is a simple integer
                action_value = action.item() if isinstance(action, torch.Tensor) else action
                
                # Execute action
                next_state, reward, done, info = self.env.step(action_value)
                
                # Store transition
                self.store_transition(state, action, log_prob, reward, value, done)
                
                # Update state and counters
                state = next_state
                episode_reward += reward
                episode_decisions += 1
                steps += 1
            
            # Calculate final makespan from environment info or using extraction methods
            makespan = self._extract_makespan(info, episode_reward, done)
            
            # Store episode statistics
            self.training_log['episode_rewards'].append(episode_reward)
            self.training_log['episode_decisions'].append(episode_decisions)
            self.training_log['episode_makespans'].append(makespan)
            
            # Update running reward
            running_reward = 0.05 * episode_reward + 0.95 * running_reward
            
            # Update policy if it's time
            if episode % self.update_frequency == 0:
                actor_loss, critic_loss, entropy_loss, total_loss = self.update()
                print(f"Update {self.update_count}: actor_loss={actor_loss:.4f}, critic_loss={critic_loss:.4f}, entropy_loss={entropy_loss:.4f}, total_loss={total_loss:.4f}")
            
            # Print progress
            episode_time = time.time() - episode_start_time
            print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, Makespan: {makespan:.2f}, Decisions: {steps}, Time: {episode_time:.2f}s")
        
        # Calculate total training time
        total_training_time = time.time() - total_training_start_time
        self.training_log['total_training_time'] = total_training_time
        
        # Training completed - save final model and plot curves using proper naming pattern
        print("\nTraining completed!")
        
        # Save model using benchmark name pattern from original PPO
        model_path = os.path.join(self.results_dir, f"{self.benchmark_name}_hybrid_ppo_agent.pt")
        print(f"Saving agent to {model_path}")
        self.save(model_path)
        
        # Plot training curves using original PPO pattern
        plot_path = os.path.join(self.results_dir, f"{self.benchmark_name}_training_curves.png")
        print(f"Creating plot at {plot_path}")
        self.plot_training_curves(save_path=plot_path)
        
        return self.policy, self.training_log
    
    def _extract_makespan(self, info, reward=None, done=False):
        """
        Extract makespan using the approach from the original jss_ppo_agent.py.
        
        Args:
            info: Info dictionary from environment step
            reward: The reward received (used as fallback)
            done: Whether the episode is done
            
        Returns:
            Extracted makespan value
        """
        # Get makespan directly from environment's last_time_step attribute
        makespan = self.env.last_time_step if hasattr(self.env, 'last_time_step') else float('inf')
        
        # If episode didn't complete or makespan is infinite, use a large but finite value
        if makespan == float('inf') or not done:
            # Use a large but finite value for incomplete episodes
            makespan = 10000 + (abs(reward) if reward is not None else 0)  # Add reward to differentiate between episodes
            if not hasattr(self, '_incomplete_makespan_warning_shown'):
                print(f"Warning: Episode did not complete all jobs. Using estimated makespan: {makespan}")
                self._incomplete_makespan_warning_shown = True
        
        return makespan
    
    def evaluate(self, num_episodes=10, render=False):
        """
        Evaluate the agent on the environment for a number of episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate for
            render: Whether to render the environment
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Set evaluation flag for makespan calculation
        self._in_evaluation = True
        
        # Store metrics
        episode_rewards = []
        episode_makespans = []
        episode_decisions = []
        episode_times = []
        
        # Set evaluation mode for the policy
        self.policy.eval()
        
        with torch.no_grad():
            for i in range(num_episodes):
                obs = self.env.reset()
                
                # Convert observation to tensor
                obs_dict = {
                    "real_obs": torch.FloatTensor(obs["real_obs"]).to(device),
                    "action_mask": torch.FloatTensor(obs["action_mask"]).to(device)
                }
                
                done = False
                episode_reward = 0
                num_decisions = 0
                start_time = time.time()
                
                while not done:
                    # Get action (deterministic for evaluation)
                    action, _, _ = self.policy.get_action(obs_dict, deterministic=True)
                    
                    # Take step in environment - ensure action is a simple integer
                    action_value = action.item() if isinstance(action, torch.Tensor) else action
                    next_obs, reward, done, info = self.env.step(action_value)
                    
                    # Render if specified
                    if render:
                        self.env.render()
                    
                    # Update stats
                    episode_reward += reward
                    num_decisions += 1
                    
                    # Update observation
                    obs = next_obs
                    obs_dict = {
                        "real_obs": torch.FloatTensor(obs["real_obs"]).to(device),
                        "action_mask": torch.FloatTensor(obs["action_mask"]).to(device)
                    }
                
                # Get the makespan using shared extraction function
                makespan = self._extract_makespan(info, episode_reward, done)
                
                # Record episode stats
                episode_rewards.append(episode_reward)
                episode_makespans.append(makespan)
                episode_decisions.append(num_decisions)
                episode_times.append(time.time() - start_time)
                
                print(f"Evaluation Episode {i+1}/{num_episodes}: "
                      f"Reward={episode_reward:.2f}, Makespan={makespan:.2f}")
        
        # Calculate metrics
        avg_reward = sum(episode_rewards) / num_episodes
        avg_makespan = sum(episode_makespans) / num_episodes
        min_makespan = min(episode_makespans)
        std_makespan = np.std(episode_makespans)
        avg_decisions = sum(episode_decisions) / num_episodes
        avg_time = sum(episode_times) / num_episodes
        
        # Calculate optimality gap using evaluation min_makespan
        gap = "N/A"
        gap_value = None
        try:
            bks = get_best_known_solution(self.benchmark_name)
            if bks and min_makespan != float('inf'):
                gap_value = (min_makespan / bks - 1) * 100
                gap = f"{gap_value:.2f}%"
                gap_value = round(gap_value, 2)
        except Exception as e:
            print(f"Error calculating optimality gap: {e}")
        
        # Clear evaluation flag
        if hasattr(self, '_in_evaluation'):
            delattr(self, '_in_evaluation')
            
        # Return evaluation results
        return {
            "min_makespan": int(min_makespan),
            "avg_makespan": float(avg_makespan),
            "std_makespan": float(std_makespan),
            "avg_reward": float(avg_reward),
            "avg_decisions": float(avg_decisions),
            "avg_time": float(avg_time),
            "optimality_gap": gap,
            "total_time": float(sum(episode_times)),
            "decisions": episode_decisions,  # Add full list of decisions
            "times": episode_times  # Add full list of times
        }
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_log': self.training_log
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        try:
            # Add numpy types to safe globals for PyTorch 2.6+
            try:
                import torch.serialization
                torch.serialization.add_safe_globals([
                    np.dtype, np.bool_, np.int_, np.intc, np.intp,
                    np.int8, np.int16, np.int32, np.int64, np.uint8,
                    np.uint16, np.uint32, np.uint64, np.float16, np.float32,
                    np.float64, np.complex64, np.complex128, np.array, np.zeros,
                    np.ndarray, np.ndindex, np.empty, np.broadcast_to,
                    np.core.multiarray.scalar
                ])
            except (ImportError, AttributeError):
                pass
                
            # Try loading with weights_only=False explicitly
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                print("Loaded checkpoint with weights_only=False")
            except Exception as e1:
                print(f"Failed to load with weights_only=False: {e1}")
                # Fall back to standard loading without weights_only
                checkpoint = torch.load(path, map_location=device)
                print("Loaded checkpoint with default options")
                
            # Load policy state
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            print("Successfully loaded policy state")
            
            # Try to load optimizer state, but handle errors gracefully
            if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Successfully loaded optimizer state")
                except ValueError as e:
                    print(f"Warning: Could not load optimizer state: {e}")
                    print("Continuing with current optimizer state")
            
            # Load training log if available
            if 'training_log' in checkpoint:
                self.training_log = checkpoint['training_log']
                print("Successfully loaded training log")
                
            print(f"Model successfully loaded from {path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def plot_training_curves(self, save_path=None):
        """
        Plot training curves with a 3x2 layout matching the original PPO implementation.
        
        Args:
            save_path: Path to save the plot to
        """
        if not self.training_log['episode_rewards']:
            print("No training data to plot.")
            return
        
        # Create figure with 3x2 subplots layout
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        
        # Plot episode rewards
        axs[0, 0].plot(self.training_log['episode_rewards'])
        axs[0, 0].set_title('Episode Rewards')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Reward')
        
        # Plot episode makespans
        axs[0, 1].plot(self.training_log['episode_makespans'])
        axs[0, 1].set_title('Episode Makespans')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Makespan')
        
        # Plot actor and critic losses with separate y-axes
        ax1 = axs[1, 0]
        ax2 = ax1.twinx()  # Create a second y-axis
        
        # Plot critic loss on the left y-axis
        critic_line, = ax1.plot(self.training_log['critic_losses'], 'orange', label='Critic Loss')
        ax1.set_xlabel('Update')
        ax1.set_ylabel('Critic Loss', color='orange')
        ax1.tick_params(axis='y', labelcolor='orange')
        
        # Plot actor loss on the right y-axis with a different scale
        actor_line, = ax2.plot(self.training_log['actor_losses'], 'blue', label='Actor Loss')
        ax2.set_ylabel('Actor Loss', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Add a title
        ax1.set_title('Actor and Critic Losses')
        
        # Add a legend
        lines = [critic_line, actor_line]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        # Plot total loss
        axs[1, 1].plot(self.training_log['total_losses'])
        axs[1, 1].set_title('Total Loss')
        axs[1, 1].set_xlabel('Update')
        axs[1, 1].set_ylabel('Loss')
        
        # Plot progressive schedule metrics - steps per episode
        axs[2, 0].plot(self.training_log['steps_per_episode'])
        axs[2, 0].set_title('Steps per Episode')
        axs[2, 0].set_xlabel('Episode')
        axs[2, 0].set_ylabel('Steps')
        
        # Plot entropy coefficient
        axs[2, 1].plot(self.training_log['entropy_coeffs'])
        axs[2, 1].set_title('Entropy Coefficient')
        axs[2, 1].set_xlabel('Episode')
        axs[2, 1].set_ylabel('Coefficient Value')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            print(f"Saved training curves to {save_path}")
        else:
            plt.show()
            
        # Close the figure to prevent memory leaks
        plt.close(fig)

def train_on_multiple_benchmarks(benchmark_names, num_episodes=1000, hidden_dim=64, 
                               lr=3e-4, gamma=0.99, 
                               clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01,
                               max_grad_norm=0.5, ppo_epochs=10, batch_size=64,
                               gae_lambda=0.95, update_frequency=10, 
                               evaluation_frequency=20, 
                               use_progressive_schedule=True, initial_steps_per_episode=1000,
                               steps_increment=500, increment_interval=100,
                               auto_entropy_tuning=True, entropy_decay_rate=0.7,
                               eval_episodes=10, results_dir='hybrid_models', 
                               save_json=None, max_steps_per_episode=3000,
                               use_gnn=True, pretrained_model_path='./checkpoints/pretrained_model_standalone.pt',
                               gnn_lr_factor=0.1, seed=None):
    """
    Train Hybrid PPO agents on multiple benchmark instances.
    
    Args:
        benchmark_names: List of benchmark names (e.g., ['ta01', 'ta02'])
        num_episodes: Number of episodes to train for each benchmark
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
        results_dir: Directory to save trained models and results
        save_json: Path to save results as JSON (optional)
        max_steps_per_episode: Maximum steps per episode
        use_gnn: Whether to use the GNN feature extractor
        pretrained_model_path: Path to the pretrained model for initialization
        gnn_lr_factor: Learning rate factor for GNN components
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with results for each benchmark
    """
    # Set a random seed for reproducibility if provided
    if seed is not None:
        print(f"Random seed: {seed}")
        set_seed(seed)
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Results dictionary
    results = {}
    
    # Train on each benchmark
    for benchmark in benchmark_names:
        print(f"\n{'='*50}")
        print(f"Training on benchmark: {benchmark}")
        print(f"{'='*50}\n")
        
        # Get the full path to the benchmark file
        instance_path = None
        potential_paths = [
            os.path.join(os.path.dirname(JSSEnv.__file__), 'envs', 'instances', benchmark),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        'JSSEnv', 'JSSEnv', 'envs', 'instances', benchmark)
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                instance_path = path
                print(f"Found instance file: {path}")
                break
                
        if instance_path is None:
            print(f"Error: Could not find instance file for benchmark {benchmark}")
            print("Searched in:")
            for path in potential_paths:
                print(f"  - {path}")
            continue
        
        # Create environment
        env = gym.make('jss-v1', env_config={'instance_path': instance_path})
        
        # Create agent with all parameters
        agent = HybridPPOAgent(
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
            entropy_decay_rate=entropy_decay_rate,
            use_gnn=use_gnn,
            pretrained_model_path=pretrained_model_path,
            gnn_lr_factor=gnn_lr_factor,
            results_dir=results_dir
        )
        
        # Explicitly set benchmark name to ensure correct file naming
        agent.benchmark_name = benchmark
        
        # Train agent
        print(f"Training agent for {num_episodes} episodes with {'GNN-enhanced' if use_gnn else 'standard MLP'} feature extractor")
        training_start_time = time.time()
        agent.train(num_episodes, update_frequency=update_frequency)
        total_training_time = time.time() - training_start_time
        
        # Calculate training statistics
        training_makespans = agent.training_log['episode_makespans']
        training_rewards = agent.training_log['episode_rewards']
        training_times = agent.training_log.get('episode_times', [0])
        training_decisions = agent.training_log.get('episode_decisions', [0])
        
        # Prepare training results dictionary
        min_makespan = min(training_makespans) if training_makespans else float('inf')
        avg_makespan = np.mean(training_makespans) if training_makespans else float('inf')
        std_makespan = np.std(training_makespans) if training_makespans else 0
        avg_reward = np.mean(training_rewards) if training_rewards else 0
        avg_decisions = np.mean(training_decisions) if training_decisions else 0
        
        # Save agent
        save_path = os.path.join(results_dir, f"{benchmark}_hybrid_ppo_agent.pt")
        print(f"Saving agent to {save_path}")
        agent.save(save_path)
        
        # Plot training curves using original PPO pattern
        curves_path = os.path.join(results_dir, f"{benchmark}_training_curves.png")
        agent.plot_training_curves(save_path=curves_path)
        
        # Evaluate agent
        print(f"Evaluating agent for {eval_episodes} episodes")
        eval_results = agent.evaluate(num_episodes=eval_episodes)
        
        # Store results combining training and evaluation data
        results[benchmark] = {
            'min_makespan': min_makespan,
            'avg_makespan': avg_makespan,
            'std_makespan': std_makespan,
            'avg_reward': avg_reward,
            'avg_decisions': avg_decisions,
            'total_training_time': total_training_time,  # Use the actual training time
            'evaluation_min_makespan': eval_results['min_makespan'],
            'evaluation_avg_makespan': eval_results['avg_makespan'],
            'evaluation_std_makespan': eval_results['std_makespan'],
            'evaluation_avg_reward': eval_results['avg_reward'],
            'evaluation_avg_decisions': eval_results['avg_decisions'],
            'evaluation_avg_time': eval_results['avg_time'],
            'gap': eval_results['optimality_gap']  # Use the gap from evaluation
        }
        
        print(f"\nResults for {benchmark}:")
        print(f"Training Min Makespan: {min_makespan}")
        print(f"Training Avg Makespan: {avg_makespan:.2f} ± {std_makespan:.2f}")
        print(f"Training Avg Reward: {avg_reward:.2f}")
        print(f"Evaluation Min Makespan: {eval_results['min_makespan']}")
        print(f"Evaluation Avg Makespan: {eval_results['avg_makespan']:.2f} ± {eval_results['std_makespan']:.2f}")
        print(f"Evaluation Avg Reward: {eval_results['avg_reward']:.2f}")
        print(f"Evaluation Avg Time: {eval_results['avg_time']:.2f}s")
        print(f"Optimality Gap: {eval_results['optimality_gap']}")  # Use the gap from evaluation
    
    # Save results to JSON if requested
    if save_json:
        save_results_to_json(results, save_json, approach_name='HybridPPO')
    
    return results

def evaluate_on_multiple_benchmarks(benchmark_names, model_dir, num_episodes=10, render=False, save_json=None):
    """
    Evaluate trained Hybrid PPO agents on multiple benchmark instances.
    
    Args:
        benchmark_names: List of benchmark names (e.g., ['ta01', 'ta02'])
        model_dir: Directory containing trained models
        num_episodes: Number of episodes to evaluate for
        render: Whether to render the environment during evaluation
        save_json: Path to save results as JSON (optional)
        
    Returns:
        Dictionary with evaluation results for each benchmark
    """
    # Results dictionary
    results = {}
    
    # Evaluate on each benchmark
    for benchmark in benchmark_names:
        print(f"\n{'='*50}")
        print(f"Evaluating on benchmark: {benchmark}")
        print(f"{'='*50}\n")
        
        # Get the full path to the benchmark file
        instance_path = None
        potential_paths = [
            os.path.join(os.path.dirname(JSSEnv.__file__), 'envs', 'instances', benchmark),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        'JSSEnv', 'JSSEnv', 'envs', 'instances', benchmark)
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                instance_path = path
                print(f"Found instance file: {path}")
                break
                
        if instance_path is None:
            print(f"Error: Could not find instance file for benchmark {benchmark}")
            print("Searched in:")
            for path in potential_paths:
                print(f"  - {path}")
            continue
        
        # Create environment
        env = gym.make('jss-v1', env_config={'instance_path': instance_path})
        
        # Create agent (with default parameters for evaluation)
        agent = HybridPPOAgent(env)
        
        # Load trained model
        model_path = os.path.join(model_dir, f"{benchmark}_hybrid_ppo_agent.pt")
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            agent.load(model_path)
        else:
            print(f"Warning: No trained model found for {benchmark} at {model_path}")
            print("Using untrained agent for evaluation.")
        
        # Evaluate agent
        print(f"Evaluating agent for {num_episodes} episodes")
        eval_results = agent.evaluate(num_episodes=num_episodes, render=render)
        
        # Calculate optimality gap if available
        gap = "N/A"
        gap_value = None
        try:
            bks = get_best_known_solution(benchmark)
            if bks and eval_results['min_makespan'] != float('inf'):
                gap_value = (eval_results['min_makespan'] / bks - 1) * 100
                gap = f"{gap_value:.2f}%"
                gap_value = round(gap_value, 2)
        except Exception as e:
            print(f"Error calculating optimality gap: {e}")
        
        # Prepare combined results dictionary
        benchmark_results = {
            'min_makespan': eval_results['min_makespan'],
            'avg_makespan': eval_results['avg_makespan'],
            'std_makespan': eval_results['std_makespan'],
            'avg_reward': eval_results['avg_reward'],
            'avg_decisions': eval_results['avg_decisions'],
            'total_training_time': eval_results['total_time'],
            'evaluation_min_makespan': eval_results['min_makespan'],
            'evaluation_avg_makespan': eval_results['avg_makespan'],
            'evaluation_std_makespan': eval_results['std_makespan'],
            'evaluation_avg_reward': eval_results['avg_reward'],
            'evaluation_avg_decisions': eval_results['avg_decisions'],
            'evaluation_avg_time': eval_results['avg_time'],
            'gap': gap_value
        }
        
        # Store results
        results[benchmark] = benchmark_results
        
        # Print evaluation results
        print(f"\nResults for {benchmark}:")
        print(f"Min Makespan: {benchmark_results['min_makespan']}")
        print(f"Avg Makespan: {benchmark_results['avg_makespan']:.2f} ± {benchmark_results['std_makespan']:.2f}")
        print(f"Avg Reward: {benchmark_results['avg_reward']:.2f}")
        print(f"Avg Time: {benchmark_results['evaluation_avg_time']:.2f}s")
        print(f"Optimality Gap: {benchmark_results['gap']}")
    
    # Save results to JSON if requested
    if save_json:
        save_results_to_json(results, save_json, approach_name='HybridPPO')
    
    return results

def save_results_to_json(results, output_file, approach_name='HybridPPO'):
    """
    Save results to a JSON file in the exact format of the original PPO results.
    If the file exists, it will update existing benchmark entries or add new ones.
    
    Args:
        results: Dictionary of results
        output_file: Path to output file
        approach_name: Name of the approach to use as key in the JSON
    """
    # Load existing results if the file exists
    existing_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
                print(f"Loaded existing results from {output_file}")
        except Exception as e:
            print(f"Warning: Failed to load existing results from {output_file}: {e}")
            # Continue with empty existing_results
    
    # Format results to match the original PPO results format
    formatted_results = existing_results.copy()  # Start with existing results
    
    for benchmark, result in results.items():
        # Calculate optimality gap using evaluation min_makespan
        gap = "N/A"
        gap_value = None
        try:
            bks = get_best_known_solution(benchmark)
            if bks and result.get("evaluation_min_makespan") != float('inf'):
                gap_value = (result["evaluation_min_makespan"] / bks - 1) * 100
                gap = f"{gap_value:.2f}%"
                gap_value = round(gap_value, 2)
        except Exception as e:
            print(f"Error calculating optimality gap: {e}")
        
        # Format the result with appropriate types - use training metrics for training stats
        formatted_result = {
            "benchmark": benchmark,
            # Training metrics
            "min_makespan": int(result.get("min_makespan", 0)),
            "avg_makespan": float(result.get("avg_makespan", 0.0)),
            "std_makespan": float(result.get("std_makespan", 0.0)),
            "avg_reward": float(result.get("avg_reward", 0.0)),
            "avg_decisions": float(result.get("avg_decisions", 0.0)),
            "total_training_time": float(result.get("total_training_time", 0.0)),
            # Evaluation metrics
            "evaluation_min_makespan": int(result.get("evaluation_min_makespan", 0)),
            "evaluation_avg_makespan": float(result.get("evaluation_avg_makespan", 0.0)),
            "evaluation_std_makespan": float(result.get("evaluation_std_makespan", 0.0)),
            "evaluation_avg_reward": float(result.get("evaluation_avg_reward", 0.0)),
            "evaluation_avg_decisions": float(result.get("evaluation_avg_decisions", 0.0)),
            "evaluation_avg_time": float(result.get("evaluation_avg_time", 0.0)),
            # Gap from evaluation
            "gap": gap
        }
        formatted_results[benchmark] = formatted_result
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(formatted_results, f, indent=4)
    
    print(f"Results saved to {output_file}")
    return formatted_results

def main():
    """
    Main entry point for running Hybrid PPO agent.
    """
    parser = argparse.ArgumentParser(description='Hybrid PPO for Job Shop Scheduling')
    
    # Basic parameters
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help='Mode: train or evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+', default=['jss-env:JSS-20x10-v0'],
                        help='Benchmark environments to use')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=1000,
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
    
    # Progressive scheduling parameters
    parser.add_argument('--use_progressive_schedule', action='store_true',
                        help='Whether to gradually increase episode length')
    parser.add_argument('--initial_steps_per_episode', type=int, default=1000,
                        help='Starting episode length for progressive training')
    parser.add_argument('--steps_increment', type=int, default=500,
                        help='How much to increase steps per episode each time')
    parser.add_argument('--increment_interval', type=int, default=100,
                        help='How often to increase episode length (in episodes)')
    parser.add_argument('--max_steps_per_episode', type=int, default=3000,
                        help='Maximum steps per episode')
    
    # Entropy tuning parameters
    parser.add_argument('--auto_entropy_tuning', action='store_true',
                        help='Whether to decrease entropy coefficient over time')
    parser.add_argument('--entropy_decay_rate', type=float, default=0.7,
                        help='Rate to multiply entropy coefficient at each adjustment')
    
    # Hybrid parameters
    parser.add_argument('--use_gnn', action='store_true', default=True,
                        help='Whether to use GNN features in the policy')
    parser.add_argument('--pretrained_model_path', type=str, 
                        default='./checkpoints/pretrained_model_standalone.pt',
                        help='Path to pretrained model from self-labeling')
    parser.add_argument('--gnn_lr_factor', type=float, default=0.1,
                        help='Factor to reduce learning rate for pretrained parts')
    
    # Evaluation parameters
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of episodes to evaluate on')
    parser.add_argument('--render', action='store_true',
                        help='Whether to render the environment during evaluation')
    parser.add_argument('--model_dir', type=str, default='hybrid_models',
                        help='Directory containing trained models for evaluation')
    
    # Output parameters
    parser.add_argument('--results_dir', type=str, default='hybrid_models',
                        help='Directory to save results')
    parser.add_argument('--save_json', type=str, default=None,
                        help='Path to save results JSON')
    
    args = parser.parse_args()
    
    # Print basic information
    print(f"Running Hybrid PPO in {args.mode} mode")
    print(f"Using {'GNN-enhanced' if args.use_gnn else 'standard MLP'} feature extractor")
    if args.use_gnn and args.pretrained_model_path:
        print(f"Using pretrained GNN from {args.pretrained_model_path}")
    print(f"Benchmarks: {args.benchmarks}")
    
    # Set random seed and only print once
    if args.seed is not None:
        set_seed(args.seed)  # Don't print inside set_seed
    
    if args.mode == 'train':
        # Train on multiple benchmarks
        results = train_on_multiple_benchmarks(
            benchmark_names=args.benchmarks,
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
            results_dir=args.results_dir,
            save_json=args.save_json,
            max_steps_per_episode=args.max_steps_per_episode,
            use_gnn=args.use_gnn,
            pretrained_model_path=args.pretrained_model_path,
            gnn_lr_factor=args.gnn_lr_factor,
            seed=args.seed
        )
    else:
        # Evaluate on multiple benchmarks
        results = evaluate_on_multiple_benchmarks(
            benchmark_names=args.benchmarks,
            model_dir=args.model_dir,
            num_episodes=args.eval_episodes,
            render=args.render,
            save_json=args.save_json
        )
    
    if args.save_json and args.mode == 'evaluate':
        save_results_to_json(results, args.save_json)

if __name__ == "__main__":
    main() 
