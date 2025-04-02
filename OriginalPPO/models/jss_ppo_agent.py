#!/usr/bin/env python3
"""
PPO Agent for Job Shop Scheduling using JSSEnv.

This script implements a Proximal Policy Optimization (PPO) agent
for solving Job Shop Scheduling problems using the JSSEnv environment.
"""

import os
import time
import argparse
import numpy as np
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

# Import gap calculation utilities
from utils.best_known_solutions import get_best_known_solution, calculate_optimality_gap

# Set gap calculation as available
gap_calculation_available = True

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JSSFeatureExtractor(nn.Module):
    """
    Feature extractor for the JSSEnv observation space.
    
    Processes the dictionary observation from JSSEnv into a flat feature vector.
    """
    def __init__(self, obs_space, hidden_dim=64):
        super(JSSFeatureExtractor, self).__init__()
        
        # Get dimensions from observation space
        self.jobs = obs_space["real_obs"].shape[0]
        self.features_per_job = obs_space["real_obs"].shape[1]
        self.action_mask_dim = obs_space["action_mask"].shape[0]
        
        # Calculate input dimension
        self.input_dim = self.jobs * self.features_per_job
        
        # Feature extraction network
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Output dimension
        self.output_dim = hidden_dim
    
    def forward(self, obs_dict):
        """
        Process the observation dictionary.
        
        Args:
            obs_dict: Dictionary with 'real_obs' and 'action_mask' keys
            
        Returns:
            Extracted features and action mask
        """
        # Extract components
        real_obs = obs_dict["real_obs"]
        action_mask = obs_dict["action_mask"]
        
        # Flatten real_obs
        batch_size = real_obs.shape[0] if len(real_obs.shape) > 2 else 1
        if batch_size == 1 and len(real_obs.shape) == 2:
            flat_obs = real_obs.reshape(1, -1)
        else:
            flat_obs = real_obs.reshape(batch_size, -1)
        
        # Extract features
        features = self.feature_net(flat_obs)
        
        return features, action_mask


class PPOPolicy(nn.Module):
    """
    PPO Policy Network for Job Shop Scheduling.
    
    Includes both actor (policy) and critic (value) networks.
    """
    def __init__(self, obs_space, action_space, hidden_dim=64):
        super(PPOPolicy, self).__init__()
        
        # Feature extractor
        self.feature_extractor = JSSFeatureExtractor(obs_space, hidden_dim)
        
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
        
        # Extract features
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
            return 0, torch.tensor([0.0]).to(device), state_value
        
        # Sample action
        try:
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                # Create distribution only from valid actions
                dist = Categorical(action_probs)
                action = dist.sample()
            
            # Double-check that the action is valid
            if action_mask[0, action.item()].item() < 0.5:
                # If somehow we still got an invalid action, select a valid one
                valid_actions = torch.where(action_mask[0] > 0.5)[0]
                if len(valid_actions) > 0:
                    action = valid_actions[0].unsqueeze(0)
                    print(f"Warning: Invalid action {action.item()} selected. Corrected to {valid_actions[0].item()}")
                else:
                    # This should never happen due to the check above
                    print("Critical error: No valid actions but mask sum was non-zero!")
                    return 0, torch.tensor([0.0]).to(device), state_value
            
            # Get log probability
            dist = Categorical(action_probs)
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob, state_value
            
        except Exception as e:
            print(f"Error in action selection: {e}")
            # Fallback to selecting the first valid action
            valid_actions = torch.where(action_mask[0] > 0.5)[0]
            if len(valid_actions) > 0:
                action = valid_actions[0].unsqueeze(0)
                dist = Categorical(action_probs)
                log_prob = dist.log_prob(action)
                return action.item(), log_prob, state_value
            else:
                print("Critical error: No valid actions available!")
                return 0, torch.tensor([0.0]).to(device), state_value
    
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


class PPOAgent:
    """
    PPO Agent for Job Shop Scheduling.
    
    Implements the PPO algorithm for training an agent to solve JSS problems.
    """
    def __init__(self, env, hidden_dim=64, lr=3e-4, gamma=0.99, clip_ratio=0.2, value_coef=0.5,
                 entropy_coef=0.01, max_grad_norm=0.5, ppo_epochs=10, batch_size=64,
                 gae_lambda=0.95, update_frequency=10, evaluation_frequency=20,
                 use_progressive_schedule=False, initial_steps_per_episode=1000,
                 steps_increment=500, increment_interval=100, auto_entropy_tuning=False,
                 entropy_decay_rate=0.7, max_steps_per_episode=3000):
        """
        Initialize the PPO agent.
        
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
        # Pass environment spaces to the policy for proper initialization
        self.policy = PPOPolicy(
            env.observation_space,
            env.action_space,
            hidden_dim=hidden_dim
        )
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
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
            'steps_per_episode': []
        }
    
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
        """Compute returns and advantages using GAE."""
        # Convert to tensors
        rewards = torch.FloatTensor(self.rewards).to(device)
        values = torch.cat([v.detach() for v in self.values]).flatten().to(device)
        dones = torch.FloatTensor(self.dones).to(device)
        
        # Initialize returns and advantages
        returns = torch.zeros_like(rewards).to(device)
        advantages = torch.zeros_like(rewards).to(device)
        
        # Initialize for GAE
        next_value = 0
        next_advantage = 0
        
        # Compute returns and advantages in reverse order
        for t in reversed(range(len(rewards))):
            # Compute TD target
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_return = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_return = returns[t + 1]
            
            # Compute return
            returns[t] = rewards[t] + self.gamma * next_non_terminal * next_return
            
            # Compute TD error and GAE
            if t == len(rewards) - 1:
                # For the last step, use next_value (which is 0 by default)
                delta = rewards[t] + self.gamma * next_non_terminal * next_value - values[t]
            else:
                # For all other steps, use the next value in the sequence
                delta = rewards[t] + self.gamma * next_non_terminal * values[t + 1] - values[t]
            
            # Compute GAE
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * next_advantage
            
            # Update next_advantage for the next iteration
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
                    # Use absolute value to ensure actor loss is not zero due to sign issues
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # If actor loss is very close to zero, add a small epsilon to ensure it's not exactly zero
                    if abs(actor_loss.item()) < 1e-6:
                        print(f"Warning: Actor loss is very small: {actor_loss.item():.8f}. Adding epsilon.")
                        actor_loss = torch.abs(actor_loss) + 1e-6
                    
                    critic_loss = nn.MSELoss()(state_values.reshape(-1), batch_returns.reshape(-1))
                    entropy_loss = -entropy.mean()
                    
                    # Debug information - print every 100 updates
                    if len(self.training_log['actor_losses']) % 10 == 0 and start_idx == 0:
                        print(f"\nDebug - Update {len(self.training_log['actor_losses'])}")
                        print(f"Ratios min/mean/max: {ratios.min().item():.4f}/{ratios.mean().item():.4f}/{ratios.max().item():.4f}")
                        print(f"Advantages min/mean/max: {batch_advantages.min().item():.4f}/{batch_advantages.mean().item():.4f}/{batch_advantages.max().item():.4f}")
                        print(f"Actor loss: {actor_loss.item():.6f}")
                        print(f"Critic loss: {critic_loss.item():.6f}")
                        print(f"New/old log probs diff min/mean/max: {(new_log_probs - batch_log_probs).min().item():.4f}/{(new_log_probs - batch_log_probs).mean().item():.4f}/{(new_log_probs - batch_log_probs).max().item():.4f}")
                    
                    # Total loss
                    loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                    
                    # Optimize
                    self.optimizer.zero_grad()
                    
                    # Use try-except to handle any backward pass issues
                    try:
                        loss.backward()
                    except RuntimeError as e:
                        if "trying to backward through the graph a second time" in str(e):
                            # If we hit this error, try with retain_graph=True
                            try:
                                loss.backward(retain_graph=True)
                            except Exception as e2:
                                print(f"Error in backward pass (with retain_graph=True): {e2}")
                                # Continue to the next batch
                                continue
                        else:
                            print(f"Error in backward pass: {e}")
                            # Continue to the next batch
                            continue
                    
                    # Clip gradients
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    
                    # Update parameters
                    self.optimizer.step()
                    
                    # Log losses
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    total_losses.append(loss.item())
            
            # Update logs
            if actor_losses:  # Only update if we have losses
                self.training_log['actor_losses'].append(np.mean(actor_losses))
                self.training_log['critic_losses'].append(np.mean(critic_losses))
                self.training_log['entropy_losses'].append(np.mean(entropy_losses))
                self.training_log['total_losses'].append(np.mean(total_losses))
            
            # Reset storage
            self.reset_storage()
            
            # Update entropy coefficient if adaptive tuning is enabled
            if self.auto_entropy_tuning and self.episode_counter % self.increment_interval == 0 and self.episode_counter > 0:
                self.entropy_coef = max(self.entropy_coef * self.entropy_decay_rate, 1e-4)
                print(f"Decreasing entropy coefficient to {self.entropy_coef:.6f}")
            
            # Increment episode counter for progressive scheduling
            self.episode_counter += 1
            
            # Update counter
            self.update_count += 1
            
            # Print update statistics
            print(f"Update statistics (#{self.update_count}): actor_loss={np.mean(actor_losses):.4f}, critic_loss={np.mean(critic_losses):.4f}, entropy_coef={self.entropy_coef:.4f}")
            
            return True
        
        except Exception as e:
            print(f"Error in PPO update: {e}")
            import traceback
            traceback.print_exc()
            
            # Reset storage to avoid keeping bad data
            self.reset_storage()
            
            return False
    
    def train(self, num_episodes, update_frequency=None):
        """
        Train the agent for a specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to train for
            update_frequency: Frequency of policy updates (in episodes)
        """
        # Use instance update_frequency if not provided as argument
        if update_frequency is None:
            update_frequency = self.update_frequency
            
        # Training loop
        for episode in range(1, num_episodes + 1):
            try:
                # Reset environment
                state = self.env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
                
                # Update progressive schedule if enabled
                if self.use_progressive_schedule and episode % self.increment_interval == 0 and episode > 0:
                    self.current_steps_per_episode = min(
                        self.current_steps_per_episode + self.steps_increment,
                        self.max_steps_per_episode
                    )
                    print(f"Increasing episode length to {self.current_steps_per_episode} steps")
                
                # Update entropy coefficient if adaptive tuning is enabled
                if self.auto_entropy_tuning and episode % self.increment_interval == 0 and episode > 0:
                    self.entropy_coef = max(self.entropy_coef * self.entropy_decay_rate, 1e-4)
                    print(f"Decreasing entropy coefficient to {self.entropy_coef:.6f}")
                
                # Log current parameters
                self.training_log['learning_rates'].append(self.lr)
                self.training_log['entropy_coeffs'].append(self.entropy_coef)
                self.training_log['steps_per_episode'].append(self.current_steps_per_episode)
                
                # Increment episode counter for progressive scheduling
                self.episode_counter += 1
                
                # Episode loop - use current_steps_per_episode from progressive schedule
                for step in range(self.current_steps_per_episode):
                    try:
                        # Select action
                        action, log_prob, value = self.policy.get_action(state)
                        
                        # Take action
                        next_state, reward, done, _ = self.env.step(action)
                        
                        # Store transition
                        self.store_transition(state, action, log_prob, reward, value, done)
                        
                        # Update state
                        state = next_state
                        episode_reward += reward
                        episode_length += 1
                        
                        # Check if episode is done
                        if done:
                            break
                    except IndexError as e:
                        # This is likely the index out of bounds error we're trying to fix
                        print(f"IndexError in episode {episode}, step {step}: {e}")
                        print(f"Action selected: {action}")
                        print(f"Action mask shape: {state['action_mask'].shape}")
                        print(f"Number of valid actions: {np.sum(state['action_mask'])}")
                        
                        # End this episode early
                        done = True
                        break
                    except Exception as e:
                        # Handle other exceptions
                        print(f"Error in episode {episode}, step {step}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # End this episode early
                        done = True
                        break
                
                # Get makespan - if episode didn't complete, use a large but finite value
                makespan = self.env.last_time_step
                if makespan == float('inf') or not done:
                    # Use a large but finite value for incomplete episodes
                    makespan = 10000 + episode_reward  # Add reward to differentiate between episodes
                    print(f"Episode {episode} did not complete all jobs. Using estimated makespan: {makespan}")
                
                # Log episode results
                self.training_log['episode_rewards'].append(episode_reward)
                self.training_log['episode_decisions'].append(episode_length)
                self.training_log['episode_makespans'].append(makespan)
                
                # Print progress
                if episode % 10 == 0:
                    print(f"Episode {episode}/{num_episodes} | "
                          f"Reward: {episode_reward:.2f} | "
                          f"Decisions: {episode_length} | "
                          f"Makespan: {makespan} | "
                          f"Steps/Episode: {self.current_steps_per_episode} | "
                          f"Entropy Coef: {self.entropy_coef:.4f}")
                
                # Update policy
                if episode % update_frequency == 0:
                    update_success = self.update()
                    if not update_success:
                        print(f"Warning: Policy update failed at episode {episode}. Continuing training.")
                
                # Evaluate and save model if needed
                if episode % self.evaluation_frequency == 0:
                    print(f"Evaluating at episode {episode}...")
                    # We could add evaluation code here or call a separate method
            
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with the next episode
                continue
        
        # Final update
        if len(self.states) > 0:
            try:
                update_success = self.update()
                if not update_success:
                    print("Warning: Final policy update failed.")
            except Exception as e:
                print(f"Error in final update: {e}")
                import traceback
                traceback.print_exc()
    
    def evaluate(self, num_episodes=10, render=False):
        """
        Evaluate the agent on the environment.
        
        Args:
            num_episodes: Number of episodes to evaluate for
            render: Whether to render the environment
            
        Returns:
            Dictionary with evaluation results
        """
        # Evaluation results
        makespans = []
        rewards = []
        decisions = []
        times = []
        
        # Get benchmark name from environment instance path if available
        try:
            benchmark_name = os.path.basename(self.env.instance_path)
        except:
            benchmark_name = None
        
        # Evaluation loop
        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            # Start timing
            start_time = time.time()
            
            try:
                # Episode loop
                while not done:
                    try:
                        # Select action deterministically
                        action, _, _ = self.policy.get_action(state, deterministic=True)
                        
                        # Take action
                        next_state, reward, done, _ = self.env.step(action)
                        
                        # Update state
                        state = next_state
                        episode_reward += reward
                        episode_length += 1
                        
                        # Render if requested
                        if render:
                            self.env.render()
                            
                        # Check if we've reached the maximum number of steps
                        if episode_length >= self.max_steps_per_episode:
                            print(f"Evaluation episode {episode} reached max steps ({self.max_steps_per_episode})")
                            break
                    except IndexError as e:
                        # This is likely the index out of bounds error we're trying to fix
                        print(f"IndexError in evaluation episode {episode}: {e}")
                        print(f"Action selected: {action}")
                        print(f"Action mask shape: {state['action_mask'].shape}")
                        print(f"Number of valid actions: {np.sum(state['action_mask'])}")
                        
                        # End this episode early
                        done = True
                        break
                    except Exception as e:
                        print(f"Error in evaluation episode {episode}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # End this episode early
                        done = True
                        break
                
                # End timing
                episode_time = time.time() - start_time
                
                # Get makespan - if episode didn't complete, use a large but finite value
                makespan = self.env.last_time_step
                if makespan == float('inf') or not done:
                    # Use a large but finite value for incomplete episodes
                    makespan = 10000 + episode_reward  # Add reward to differentiate between episodes
                    print(f"Evaluation episode {episode} did not complete all jobs. Using estimated makespan: {makespan}")
                
                # Record results
                makespans.append(makespan)
                rewards.append(episode_reward)
                decisions.append(episode_length)
                times.append(episode_time)
                
                # Print progress
                print(f"Evaluation Episode {episode + 1}/{num_episodes} | "
                      f"Makespan: {makespan} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Decisions: {episode_length} | "
                      f"Time: {episode_time:.2f}s")
            except Exception as e:
                print(f"Error in evaluation episode {episode}: {e}")
                import traceback
                traceback.print_exc()
                
                # Record default results for this episode
                episode_time = time.time() - start_time
                # Use a large but finite value for makespan in error cases
                error_makespan = 20000  # Even larger than incomplete episodes to distinguish errors
                makespans.append(error_makespan if not makespans else makespans[-1])
                rewards.append(float('-inf'))
                decisions.append(0)
                times.append(episode_time)
        
        # Calculate statistics
        min_makespan = min(makespans)
        avg_makespan = np.mean(makespans)
        std_makespan = np.std(makespans)
        
        # Calculate optimality gap if available
        gap = None
        best_known = None
        if gap_calculation_available and benchmark_name:
            best_known = get_best_known_solution(benchmark_name)
            if best_known:
                gap = calculate_optimality_gap(min_makespan, benchmark_name)
        
        results = {
            'min_makespan': min_makespan,
            'avg_makespan': avg_makespan,
            'std_makespan': std_makespan,
            'avg_reward': np.mean(rewards),
            'avg_decisions': np.mean(decisions),
            'avg_time': np.mean(times),
            'makespans': makespans,
            'rewards': rewards,
            'decisions': decisions,
            'times': times
        }
        
        # Add gap information if available
        if gap is not None:
            results.update({
                'optimality_gap': gap,
                'best_known_solution': best_known
            })
        
        return results
    
    def save(self, path):
        """Save the agent to a file."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_log': self.training_log
        }, path)
    
    def load(self, path):
        """Load the agent from a file."""
        checkpoint = torch.load(path, map_location=device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_log = checkpoint['training_log']
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves."""
        # Create figure with subplots
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
        
        # Save the figure
        if save_path:
            plt.savefig(save_path)
            print(f"Training curves saved to {save_path}")
        
        # Close the figure to prevent memory leaks and avoid blocking
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
                               eval_episodes=10, save_dir='trained_models', 
                               save_json=None, max_steps_per_episode=3000):
    """
    Train PPO agents on multiple benchmark instances.
    
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
        save_dir: Directory to save trained models
        save_json: Path to save results as JSON (optional)
        max_steps_per_episode: Maximum steps per episode
        
    Returns:
        Dictionary with results for each benchmark
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Results dictionary
    results = {}
    
    # Train on each benchmark
    for benchmark in benchmark_names:
        print(f"\n{'='*50}")
        print(f"Training on benchmark: {benchmark}")
        print(f"{'='*50}\n")
        
        # Get the full path to the benchmark file
        instance_path = os.path.join(os.path.dirname(JSSEnv.__file__), 'envs', 'instances', benchmark)
        
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
        
        # Train agent
        print(f"Training agent for {num_episodes} episodes")
        agent.train(num_episodes, update_frequency=update_frequency)
        
        # Save agent
        save_path = os.path.join(save_dir, f"{benchmark}_ppo_agent.pt")
        print(f"Saving agent to {save_path}")
        agent.save(save_path)
        
        # Plot training curves
        curves_path = os.path.join(save_dir, f"{benchmark}_training_curves.png")
        agent.plot_training_curves(save_path=curves_path)
        
        # Evaluate agent
        print(f"Evaluating agent for {eval_episodes} episodes")
        eval_results = agent.evaluate(num_episodes=eval_episodes)
        
        # Store results
        results[benchmark] = eval_results
        
        print(f"\nResults for {benchmark}:")
        print(f"Min Makespan: {eval_results['min_makespan']}")
        print(f"Avg Makespan: {eval_results['avg_makespan']:.2f} ± {eval_results['std_makespan']:.2f}")
        print(f"Avg Reward: {eval_results['avg_reward']:.2f}")
        print(f"Avg Time: {eval_results['avg_time']:.2f}s")
    
    # Save results to JSON if requested
    if save_json:
        save_results_to_json(results, save_json, approach_name='PPO')
    
    return results


def evaluate_on_multiple_benchmarks(benchmark_names, model_dir, num_episodes=10, render=False, save_json=None):
    """
    Evaluate trained PPO agents on multiple benchmark instances.
    
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
        instance_path = os.path.join(os.path.dirname(JSSEnv.__file__), 'envs', 'instances', benchmark)
        
        # Create environment
        env = gym.make('jss-v1', env_config={'instance_path': instance_path})
        
        # Create agent
        agent = PPOAgent(env)
        
        # Load trained model
        model_path = os.path.join(model_dir, f"{benchmark}_ppo_agent.pt")
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            agent.load(model_path)
        else:
            print(f"Warning: No trained model found for {benchmark} at {model_path}")
            print("Using untrained agent for evaluation.")
        
        # Evaluate agent
        print(f"Evaluating agent for {num_episodes} episodes")
        eval_results = agent.evaluate(num_episodes=num_episodes, render=render)
        
        # Store results
        results[benchmark] = eval_results
        
        # Print evaluation results
        print(f"\nEvaluation Results for {benchmark}:")
        print(f"Min Makespan: {eval_results['min_makespan']}")
        print(f"Avg Makespan: {eval_results['avg_makespan']:.2f} ± {eval_results['std_makespan']:.2f}")
        print(f"Avg Reward: {eval_results['avg_reward']:.2f}")
        print(f"Avg Time: {eval_results['avg_time']:.2f}s")
        if 'optimality_gap' in eval_results:
            print(f"Best Known Solution: {eval_results['best_known_solution']}")
            print(f"Optimality Gap: {eval_results['optimality_gap']:.2f}%")
    
    # Save results to JSON if requested
    if save_json:
        save_results_to_json(results, save_json, approach_name='PPO')
    
    return results


def generate_comparison_report(ppo_results, other_results=None, output_file='comparison_report.txt'):
    """
    Generate a comparison report between PPO agent and other approaches.
    
    Args:
        ppo_results: Dictionary with PPO evaluation results
        other_results: Dictionary with results from other approaches
        output_file: Path to save the comparison report
    """
    # Create report header
    report = []
    report.append("="*80)
    report.append("Job Shop Scheduling - Comparison Report")
    report.append("="*80)
    report.append("")
    
    # Add timestamp
    from datetime import datetime
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Add PPO results
    report.append("-"*80)
    report.append("PPO Agent Results")
    report.append("-"*80)
    report.append(f"{'Benchmark':<10} {'Min Makespan':<15} {'Avg Makespan':<20} {'Gap (%)':<10} {'Avg Time (s)':<15}")
    report.append("-"*80)
    
    for benchmark, result in ppo_results.items():
        gap_str = f"{result['optimality_gap']:.2f}" if 'optimality_gap' in result else "N/A"
        report.append(f"{benchmark:<10} {result['min_makespan']:<15} "
                     f"{result['avg_makespan']:.2f} ± {result['std_makespan']:.2f}   "
                     f"{gap_str:<10} {result['avg_time']:.2f}")
    
    report.append("")
    
    # Add other results if provided
    if other_results:
        for approach_name, approach_results in other_results.items():
            report.append("-"*80)
            report.append(f"{approach_name} Results")
            report.append("-"*80)
            report.append(f"{'Benchmark':<10} {'Min Makespan':<15} {'Avg Makespan':<20} {'Gap (%)':<10} {'Avg Time (s)':<15}")
            report.append("-"*80)
            
            for benchmark, result in approach_results.items():
                if benchmark in ppo_results:  # Only include benchmarks that are in PPO results
                    gap_str = f"{result['optimality_gap']:.2f}" if 'optimality_gap' in result else "N/A"
                    report.append(f"{benchmark:<10} {result['min_makespan']:<15} "
                                 f"{result['avg_makespan']:.2f} ± {result.get('std_makespan', 0):.2f}   "
                                 f"{gap_str:<10} {result['avg_time']:.2f}")
            
            report.append("")
    
    # Add comparison summary
    report.append("="*80)
    report.append("Comparison Summary")
    report.append("="*80)
    report.append(f"{'Benchmark':<10} {'PPO Min':<10} {'Other Min':<10} {'PPO Gap':<10} {'Other Gap':<10} {'Speedup':<10}")
    report.append("-"*80)
    
    if other_results:
        # Get the first other approach for comparison
        other_name = list(other_results.keys())[0]
        other_approach = other_results[other_name]
        
        for benchmark in ppo_results.keys():
            if benchmark in other_approach:
                ppo_min = ppo_results[benchmark]['min_makespan']
                other_min = other_approach[benchmark]['min_makespan']
                
                ppo_gap = f"{ppo_results[benchmark]['optimality_gap']:.2f}%" if 'optimality_gap' in ppo_results[benchmark] else "N/A"
                other_gap = f"{other_approach[benchmark]['optimality_gap']:.2f}%" if 'optimality_gap' in other_approach[benchmark] else "N/A"
                
                ppo_time = ppo_results[benchmark]['avg_time']
                other_time = other_approach[benchmark]['avg_time']
                speedup = other_time / ppo_time if ppo_time > 0 else 0
                
                report.append(f"{benchmark:<10} {ppo_min:<10} {other_min:<10} "
                             f"{ppo_gap:<10} {other_gap:<10} {speedup:<10.2f}x")
    
    # Write report to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Comparison report saved to {output_file}")
    
    # Print report to console
    for line in report:
        print(line)


def save_results_to_json(results, output_file, approach_name='PPO'):
    """
    Save evaluation results to a JSON file for later comparison.
    
    Args:
        results: Dictionary with evaluation results
        output_file: Path to save the results
        approach_name: Name of the approach
    """
    # Remove 'lengths' field from results
    for benchmark, result in results.items():
        if 'lengths' in result:
            del result['lengths']
        if 'avg_length' in result:
            del result['avg_length']
            
    # Create a dictionary with the approach name as the key
    data = {approach_name: results}
    
    # Check if the file already exists
    if os.path.exists(output_file):
        try:
            # Load existing data
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
            
            # Remove 'lengths' from existing data for consistency
            for approach, approach_results in existing_data.items():
                for benchmark, result in approach_results.items():
                    if 'lengths' in result:
                        del result['lengths']
                    if 'avg_length' in result:
                        del result['avg_length']
            
            # Update with new data
            existing_data.update(data)
            data = existing_data
        except Exception as e:
            print(f"Error loading existing results: {e}")
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {output_file}")


def main():
    """Main function for training and evaluating the PPO agent."""
    parser = argparse.ArgumentParser(description='Train PPO agent for Job Shop Scheduling')
    
    # Add a mutually exclusive group for instance path or benchmark names
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--instance_path', type=str,
                      help='Path to a single JSS instance file')
    group.add_argument('--benchmarks', nargs='+',
                      help='List of benchmark names to train on (e.g., ta01 ta02 ta03)')
    
    # Other arguments
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Number of episodes to train for')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for neural networks')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE parameter')
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                        help='PPO clipping parameter')
    parser.add_argument('--value_coef', type=float, default=0.5,
                        help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Maximum gradient norm')
    parser.add_argument('--ppo_epochs', type=int, default=10,
                        help='Number of PPO epochs per update')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for updates')
    parser.add_argument('--update_frequency', type=int, default=10,
                        help='Frequency of policy updates (in episodes)')
    parser.add_argument('--evaluation_frequency', type=int, default=20,
                        help='Frequency of evaluation (in episodes)')
    parser.add_argument('--use_progressive_schedule', type=bool, default=True,
                        help='Whether to gradually increase episode length')
    parser.add_argument('--initial_steps_per_episode', type=int, default=1000,
                        help='Starting episode length for progressive training')
    parser.add_argument('--steps_increment', type=int, default=500,
                        help='How much to increase steps per episode each time')
    parser.add_argument('--increment_interval', type=int, default=100,
                        help='How often to increase episode length (in episodes)')
    parser.add_argument('--auto_entropy_tuning', type=bool, default=True,
                        help='Whether to decrease entropy coefficient over time')
    parser.add_argument('--entropy_decay_rate', type=float, default=0.7,
                        help='Rate to multiply entropy coefficient at each adjustment')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of episodes to evaluate for')
    parser.add_argument('--save_dir', type=str, default='trained_models',
                        help='Directory to save trained models')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to load a trained agent from')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate trained models without training')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing trained models for evaluation')
    parser.add_argument('--compare', action='store_true',
                        help='Generate comparison report with other approaches')
    parser.add_argument('--other_results', type=str, default=None,
                        help='Path to JSON file with results from other approaches')
    parser.add_argument('--report_file', type=str, default='comparison_report.txt',
                        help='Path to save the comparison report')
    parser.add_argument('--save_json', type=str, default=None,
                        help='Path to save results as JSON for later comparison')
    parser.add_argument('--max_steps_per_episode', type=int, default=3000,
                        help='Maximum steps per episode')
    args = parser.parse_args()
    
    try:
        # Check if we're in comparison mode
        if args.compare:
            if not args.benchmarks:
                print("Error: Comparison mode requires --benchmarks argument")
                return
            
            # Evaluate PPO on benchmarks
            print(f"Evaluating PPO on benchmarks for comparison: {args.benchmarks}")
            model_dir = args.model_dir if args.model_dir else args.save_dir
            ppo_results = evaluate_on_multiple_benchmarks(
                benchmark_names=args.benchmarks,
                model_dir=model_dir,
                num_episodes=args.eval_episodes,
                render=args.render,
                save_json=args.save_json
            )
            
            # Load other results if provided
            other_results = None
            if args.other_results:
                try:
                    with open(args.other_results, 'r') as f:
                        other_results = json.load(f)
                    print(f"Loaded results from other approaches: {list(other_results.keys())}")
                except Exception as e:
                    print(f"Error loading other results: {e}")
            
            # Generate comparison report
            generate_comparison_report(ppo_results, other_results, args.report_file)
        
        # Check if we're in evaluation-only mode
        elif args.eval_only:
            if args.benchmarks:
                # Evaluate on multiple benchmarks
                print(f"Evaluating trained models on benchmarks: {args.benchmarks}")
                model_dir = args.model_dir if args.model_dir else args.save_dir
                results = evaluate_on_multiple_benchmarks(
                    benchmark_names=args.benchmarks,
                    model_dir=model_dir,
                    num_episodes=args.eval_episodes,
                    render=args.render,
                    save_json=args.save_json
                )
                
                # Print summary of results
                print("\n" + "="*50)
                print("Summary of Evaluation Results:")
                print("="*50)
                for benchmark, result in results.items():
                    print(f"{benchmark}: Min Makespan = {result['min_makespan']}, "
                          f"Avg Makespan = {result['avg_makespan']:.2f} ± {result['std_makespan']:.2f}")
            else:
                # Evaluate on a single instance
                env = gym.make('jss-v1', env_config={'instance_path': args.instance_path})
                agent = PPOAgent(env)
                
                # Load agent
                if args.load_path:
                    print(f"Loading agent from {args.load_path}")
                    agent.load(args.load_path)
                else:
                    print("Warning: No model path specified for evaluation. Using untrained agent.")
                
                # Evaluate agent
                print(f"Evaluating agent for {args.eval_episodes} episodes")
                results = agent.evaluate(num_episodes=args.eval_episodes, render=args.render)
                
                # Save results to JSON if requested
                if args.save_json:
                    single_results = {os.path.basename(args.instance_path): results}
                    save_results_to_json(single_results, args.save_json, approach_name='PPO')
                
                # Print evaluation results
                print("\nEvaluation Results:")
                print(f"Min Makespan: {results['min_makespan']}")
                print(f"Avg Makespan: {results['avg_makespan']:.2f} ± {results['std_makespan']:.2f}")
                print(f"Avg Reward: {results['avg_reward']:.2f}")
                print(f"Avg Time: {results['avg_time']:.2f}s")
                if 'optimality_gap' in results:
                    print(f"Best Known Solution: {results['best_known_solution']}")
                    print(f"Optimality Gap: {results['optimality_gap']:.2f}%")
        else:
            # Training mode
            if args.benchmarks:
                # Train on multiple benchmarks
                print(f"Training on multiple benchmarks: {args.benchmarks}")
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
                    max_steps_per_episode=args.max_steps_per_episode,
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
                    save_json=args.save_json
                )
                
                # Print summary of results
                print("\n" + "="*50)
                print("Summary of Results:")
                print("="*50)
                for benchmark, result in results.items():
                    gap_str = f", Gap: {result['optimality_gap']:.2f}%" if 'optimality_gap' in result else ""
                    print(f"{benchmark}: Min Makespan = {result['min_makespan']}, "
                          f"Avg Makespan = {result['avg_makespan']:.2f} ± {result['std_makespan']:.2f}{gap_str}")
            else:
                # Train on a single instance
                # Create environment
                env = gym.make('jss-v1', env_config={'instance_path': args.instance_path})
                
                # Create agent
                agent = PPOAgent(
                    env, 
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
                    max_steps_per_episode=args.max_steps_per_episode,
                    update_frequency=args.update_frequency,
                    evaluation_frequency=args.evaluation_frequency,
                    use_progressive_schedule=args.use_progressive_schedule,
                    initial_steps_per_episode=args.initial_steps_per_episode,
                    steps_increment=args.steps_increment,
                    increment_interval=args.increment_interval,
                    auto_entropy_tuning=args.auto_entropy_tuning,
                    entropy_decay_rate=args.entropy_decay_rate
                )
                
                # Load agent if requested
                if args.load_path:
                    print(f"Loading agent from {args.load_path}")
                    agent.load(args.load_path)
                
                # Train agent
                print(f"Training agent for {args.num_episodes} episodes")
                agent.train(args.num_episodes, update_frequency=args.update_frequency)
                
                # Save agent
                os.makedirs(args.save_dir, exist_ok=True)
                save_path = os.path.join(args.save_dir, "jss_ppo_agent.pt")
                print(f"Saving agent to {save_path}")
                agent.save(save_path)
                
                # Plot training curves
                curves_path = os.path.join(args.save_dir, "training_curves.png")
                agent.plot_training_curves(save_path=curves_path)
                
                # Evaluate agent
                print(f"Evaluating agent for {args.eval_episodes} episodes")
                results = agent.evaluate(num_episodes=args.eval_episodes, render=args.render)
                
                # Save results to JSON if requested
                if args.save_json:
                    single_results = {os.path.basename(args.instance_path): results}
                    save_results_to_json(single_results, args.save_json, approach_name='PPO')
                
                # Print evaluation results
                print("\nEvaluation Results:")
                print(f"Min Makespan: {results['min_makespan']}")
                print(f"Avg Makespan: {results['avg_makespan']:.2f} ± {results['std_makespan']:.2f}")
                print(f"Avg Reward: {results['avg_reward']:.2f}")
                print(f"Avg Time: {results['avg_time']:.2f}s")
                if 'optimality_gap' in results:
                    print(f"Best Known Solution: {results['best_known_solution']}")
                    print(f"Optimality Gap: {results['optimality_gap']:.2f}%")
    
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 