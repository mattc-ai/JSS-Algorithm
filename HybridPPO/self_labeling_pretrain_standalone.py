"""
Self-Labeling Pretraining Module for Job Shop Scheduling

Standalone implementation that doesn't rely on pretrained weights.
"""

import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
import time
import os
import sys
import importlib.util

# Add function to set all random seeds
def set_seed(seed):
    """Set seed for all random number generators for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed} for reproducibility")

# Add SelfLabelingJobShop to path
SELF_LABELING_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "Job_Shop_Scheduling/SelfLabelingJobShop")
sys.path.insert(0, SELF_LABELING_PATH)  # Insert at beginning to prioritize

# Import sampling module and utils from SelfLabelingJobShop
try:
    # Load sampling module
    sampling_path = os.path.join(SELF_LABELING_PATH, "sampling.py")
    spec = importlib.util.spec_from_file_location("sampling", sampling_path)
    stg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stg)
    
    # Load utils module
    utils_path = os.path.join(SELF_LABELING_PATH, "utils.py")
    spec = importlib.util.spec_from_file_location("selflabeling_utils", utils_path)
    selflabeling_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selflabeling_utils)
    ObjMeter = selflabeling_utils.ObjMeter
    AverageMeter = selflabeling_utils.AverageMeter
    
    # Load inout module
    inout_path = os.path.join(SELF_LABELING_PATH, "inout.py")
    spec = importlib.util.spec_from_file_location("inout", inout_path)
    inout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inout)
    load_dataset = inout.load_dataset
    
    print(f"Successfully imported modules from: {SELF_LABELING_PATH}")
except Exception as e:
    print(f"Error importing modules from {SELF_LABELING_PATH}: {e}")
    import traceback
    traceback.print_exc()
    print("Please ensure the Job_Shop_Scheduling repository is cloned correctly")
    sys.exit(1)

# Import our standalone models
from models_v2.gat import GraphAttentionNetwork
from models_v2.mha import MultiHeadAttention

# Training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def heuristic_solution(instance, heuristic_type):
    """
    Generate a solution using a simple heuristic.
    
    Args:
        instance: JSP instance
        heuristic_type: Type of heuristic (0: SPT, 1: MWR, 2: FCFS)
    
    Returns:
        Ordered sequence of operations and makespan
    """
    # Extract instance information
    num_jobs = instance['j']
    num_machines = instance['m']
    processing_times = instance['costs']
    machine_order = instance['machines']
    
    # Initialize tracking variables
    job_ptr = [0] * num_jobs  # Current operation index for each job
    machine_available = [0] * num_machines  # Time when each machine becomes available
    job_available = [0] * num_jobs  # Time when each job becomes available for next op
    
    # Solution sequence and current state
    solution = []
    current_time = 0
    remaining_ops = num_jobs * num_machines
    
    while remaining_ops > 0:
        # Find eligible operations (job with operation ready to be processed)
        eligible_jobs = []
        for job in range(num_jobs):
            if job_ptr[job] < num_machines and job_available[job] <= current_time:
                machine = int(machine_order[job][job_ptr[job]])
                if machine_available[machine] <= current_time:
                    eligible_jobs.append(job)
        
        # If no eligible jobs, advance time to next event
        if not eligible_jobs:
            next_time = float('inf')
            for m in range(num_machines):
                if machine_available[m] > current_time:
                    next_time = min(next_time, machine_available[m])
            for j in range(num_jobs):
                if job_ptr[j] < num_machines and job_available[j] > current_time:
                    next_time = min(next_time, job_available[j])
            current_time = next_time
            continue
        
        # Select job based on heuristic
        selected_job = None
        if heuristic_type == 0:  # SPT
            min_time = float('inf')
            for job in eligible_jobs:
                machine = int(machine_order[job][job_ptr[job]])
                proc_time = float(processing_times[job][job_ptr[job]])
                if proc_time < min_time:
                    min_time = proc_time
                    selected_job = job
        
        elif heuristic_type == 1:  # MWR
            max_remaining = -1
            for job in eligible_jobs:
                remaining_time = float(sum(processing_times[job][job_ptr[job]:]))  # Sum remaining ops
                if remaining_time > max_remaining:
                    max_remaining = remaining_time
                    selected_job = job
        
        elif heuristic_type == 2:  # FCFS
            earliest_available = float('inf')
            for job in eligible_jobs:
                if job_available[job] < earliest_available:
                    earliest_available = job_available[job]
                    selected_job = job
        
        # Schedule the selected job
        machine = int(machine_order[selected_job][job_ptr[selected_job]])
        proc_time = float(processing_times[selected_job][job_ptr[selected_job]])
        
        # Update state
        start_time = max(current_time, machine_available[machine], job_available[selected_job])
        end_time = start_time + proc_time
        
        machine_available[machine] = end_time
        job_available[selected_job] = end_time
        job_ptr[selected_job] += 1
        
        # Add to solution
        solution.append(selected_job)  # Note: This stores JOB indices (0 to num_jobs-1), not operation indices
        remaining_ops -= 1
    
    # Calculate makespan
    makespan = max(machine_available)
    
    return solution, makespan


def convert_solution_to_model_format(solution, instance):
    """
    Convert a solution sequence to the format needed by the model for training
    
    Args:
        solution: List of job indices representing the solution
        instance: JSP instance data
        
    Returns:
        Trajectory and state objects for model training
    """
    # Initialize trajectory
    trajectory = torch.tensor(solution, dtype=torch.long, device=device)
    
    # Initialize state
    state = stg.JobShopStates(instance)
    
    return trajectory, state


class EncoderModel(nn.Module):
    """Simple wrapper around the GAT to match the expected interface."""
    def __init__(self, node_dim, hidden_dim, embed_dim, num_heads=3, dropout=0.15):
        super(EncoderModel, self).__init__()
        self.gat = GraphAttentionNetwork(
            in_features=node_dim,
            hidden_dim=hidden_dim,
            out_features=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.out_size = self.gat.out_size
        
    def forward(self, x, edge_index):
        return self.gat(x, edge_index)


class DecoderModel(nn.Module):
    """Simple wrapper around the MHA to match the expected interface."""
    def __init__(self, encoder_size, context_size, hidden_dim=64, dropout=0.15):
        super(DecoderModel, self).__init__()
        self.mha = MultiHeadAttention(
            encoder_size=encoder_size,
            context_size=context_size,
            hidden_size=hidden_dim,
            dropout=dropout
        )
        
    def forward(self, embeddings, state, mask=None):
        # The state is expected to be of shape [batch_size, num_jobs, context_size]
        # But sometimes it comes as [batch_size, context_size] so we need to unsqueeze
        if state.dim() == 2:
            state = state.unsqueeze(1).expand(-1, embeddings.shape[0], -1)
        scores, _ = self.mha(embeddings, state, mask)
        return scores


def self_labeling_pretrain_standalone(instances, 
                                     node_dim=15,
                                     hidden_dim=64, 
                                     embed_dim=128,
                                     num_heads=3,
                                     dropout=0.15,
                                     num_iterations=5, 
                                     batch_size=32, 
                                     lr=1e-4, 
                                     epochs_per_iteration=10,
                                     model_path='./checkpoints/pretrained_model_standalone.pt',
                                     val_set=None,
                                     seed=None):
    """
    Implement self-labeling pre-training with standalone models.
    
    Args:
        instances: List of JSP instances
        node_dim: Dimension of node features
        hidden_dim: Dimension of hidden layers
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        dropout: Dropout rate
        num_iterations: Number of self-labeling iterations
        batch_size: Batch size for training
        lr: Learning rate for pre-training
        epochs_per_iteration: Number of epochs per iteration
        model_path: Path to save the pre-trained model
        val_set: Optional validation set for tracking improvement
        seed: Random seed for reproducibility
        
    Returns:
        encoder, decoder: Trained models
        best_solutions: List of best solutions found
        best_makespans: List of makespans for best solutions
    """
    # Make sure seed is set
    set_seed(seed)
    
    print(f"Starting self-labeling pre-training with {len(instances)} instances")
    print(f"Will run {num_iterations} iterations of self-improvement")
    
    # Debug output for model path
    print(f"Model will be saved to: {model_path}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Create encoder and decoder models
    # First get expected dimensions from the instance format
    sample_instance = instances[0]
    node_features_dim = sample_instance['x'].shape[1]
    state_dim = stg.JobShopStates.size  # Context size from the JSS environment
    
    print(f"Creating encoder with node_dim={node_features_dim}, hidden_dim={hidden_dim}, embed_dim={embed_dim}")
    encoder = EncoderModel(
        node_dim=node_features_dim,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)
    
    print(f"Creating decoder with encoder_size={encoder.out_size}, context_size={state_dim}")
    decoder = DecoderModel(
        encoder_size=encoder.out_size,
        context_size=state_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)
    
    # Place models in training mode
    encoder.train()
    decoder.train()
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=lr
    )
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    
    # Step 1: Generate initial solutions with heuristics
    solutions = []
    makespans = []
    
    for instance in tqdm(instances, desc="Generating initial solutions"):
        # Run multiple heuristics and keep the best solution
        best_makespan = float('inf')
        best_solution = None
        
        for heuristic_idx in range(3):  # SPT, MWR, FCFS
            solution, makespan = heuristic_solution(instance, heuristic_idx)
            
            if makespan < best_makespan:
                best_makespan = makespan
                best_solution = solution
        
        solutions.append(best_solution)
        makespans.append(best_makespan)
    
    print(f"Generated initial solutions with average makespan: {np.mean(makespans):.2f}")
    
    # Step 2: Iterative Self-Improvement
    best_overall_val_gap = float('inf')
    
    for iteration in range(num_iterations):
        print(f"\nSelf-labeling iteration {iteration+1}/{num_iterations}")
        
        # Step 2.1: Train model on current best solutions
        print("Training model on current best solutions...")
        
        for epoch in range(epochs_per_iteration):
            # Shuffle instances for training
            indices = list(range(len(instances)))
            random.shuffle(indices)
            
            epoch_loss = 0.0
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_loss = 0.0
                
                optimizer.zero_grad()
                
                # Process each instance in the batch
                for idx in batch_indices:
                    instance = instances[idx]
                    solution = solutions[idx]
                    
                    # Convert instance to tensors
                    x = instance['x'].to(device)
                    edge_index = instance['edge_index'].to(device)
                    
                    # Forward pass through encoder
                    embeddings = encoder(x, edge_index)
                    
                    # Process each step in the solution
                    jsp = stg.JobShopStates(device)  # Initialize with device
                    state, mask = jsp.init_state(instance)  # Initialize state
                    
                    # Train on all steps except the last one (which is deterministic)
                    for step_idx, job in enumerate(solution[:-1]):  # Skip the last step
                        # Validate job index
                        if job >= instance['j']:  # Check if job index is out of bounds
                            print(f"Warning: Invalid job index {job} for instance {instance.get('name', idx)} with {instance['j']} jobs")
                            continue  # Skip this step
                            
                        # Get decoder output
                        logits = decoder(embeddings[jsp.ops], state) + mask.log()
                        
                        # Calculate loss  
                        target = torch.tensor([job], dtype=torch.long, device=device)
                        loss = criterion(logits, target) 
                        batch_loss += loss / len(batch_indices)  # Average over batch
                        
                        # Update state for next step
                        job_tensor = torch.tensor([job], device=device)
                        state, mask = jsp.update(job_tensor)
                
                # Backpropagation
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()
            
            avg_epoch_loss = epoch_loss * batch_size / len(instances)
            print(f"Epoch {epoch+1}/{epochs_per_iteration}: Loss = {avg_epoch_loss:.4f}")
        
        # Step 2.2: Use the improved model to generate new solutions 
        print("Generating new solutions with the improved model...")
        encoder.eval()
        decoder.eval()
        
        new_solutions = []
        new_makespans = []
        
        with torch.no_grad():
            for instance in tqdm(instances, desc="Sampling solutions"):
                # Try multiple approaches to get the best solution
                
                # 1. Sample multiple solutions
                sols, mss = stg.sampling(instance, encoder, decoder, bs=16, device=device)
                
                # 2. Try greedy solution as well
                greedy_sol, greedy_ms = stg.greedy(instance, encoder, decoder, device=device)
                
                # Determine the best solution from either approach
                if greedy_ms.item() < mss.min().item():
                    best_sol_matrix = greedy_sol.unsqueeze(0)  # Add batch dimension to match sampling
                    best_ms = greedy_ms.item()
                else:
                    best_idx = torch.argmin(mss).item()
                    best_sol_matrix = sols[best_idx].unsqueeze(0)
                    best_ms = mss[best_idx].item()
                
                # For JSP, the actual solution is the machine schedule (matrix form)
                # But we need to convert it to a list of job indices for training
                best_sol = []
                job_indices = []
                
                # Create a valid mapping from operations to job indices
                valid_operations = {}
                
                # First, identify all valid operations in the solution matrix
                for m in range(best_sol_matrix.shape[1]):  # Iterate over job slots
                    for j in range(best_sol_matrix.shape[2]):  # Iterate over machine slots
                        op = best_sol_matrix[0, m, j].item()
                        if op >= 0:  # Only add valid operations (not -1)
                            # Calculate job index from operation index
                            job_idx = op // instance['m']
                            
                            # Ensure job_idx is valid
                            if job_idx < instance['j']:
                                valid_operations[op] = job_idx
                                best_sol.append(op)
                                job_indices.append(job_idx)
                
                # If we don't have exactly num_jobs * num_machines operations
                # We'll construct a valid solution by carefully adding missing operations
                if len(best_sol) != instance['j'] * instance['m']:
                    # Use heuristic to fill in missing operations
                    best_solution, _ = heuristic_solution(instance, 0)  # Use SPT as fallback
                    job_indices = best_solution
                
                # Add the solution to our lists
                new_solutions.append(job_indices)  # Store job indices for training
                new_makespans.append(best_ms)
        
        encoder.train()
        decoder.train()
        
        print(f"Generated new solutions with average makespan: {np.mean(new_makespans):.2f}")
        
        # Step 2.3: Keep better solutions
        improvements = 0
        for i in range(len(instances)):
            if new_makespans[i] < makespans[i]:
                solutions[i] = new_solutions[i]
                makespans[i] = new_makespans[i]
                improvements += 1
        
        print(f"Kept {improvements}/{len(instances)} new solutions that were better")
        print(f"Updated average makespan: {np.mean(makespans):.2f}")
        
        # Validation, if provided
        if val_set:
            val_gap = evaluate_model(encoder, decoder, val_set)
            print(f"Validation gap: {val_gap:.2f}%")
            
            # Save a checkpoint after each iteration
            checkpoint_path = f"{os.path.splitext(model_path)[0]}_latest.pt"
            print(f"Saving checkpoint to: {checkpoint_path}")
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'iteration': iteration,
                'best_val_gap': best_overall_val_gap
            }, checkpoint_path)
            
            # Save if best
            if val_gap < best_overall_val_gap:
                best_overall_val_gap = val_gap
                print(f"Saving best model to: {model_path}")
                torch.save({
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'val_gap': val_gap
                }, model_path)
                print(f"New best model saved with validation gap: {val_gap:.2f}%")
        else:
            # Save at each iteration
            iter_path = f"{os.path.splitext(model_path)[0]}_iter{iteration+1}.pt"
            print(f"Saving iteration checkpoint to: {iter_path}")
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict()
            }, iter_path)
    
    print("\nSelf-labeling pre-training completed.")
    print(f"Final average makespan: {np.mean(makespans):.2f}")
    
    # Save final model if no validation was done
    if not val_set:
        print(f"Saving final model to: {model_path}")
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict()
        }, model_path)
    
    return encoder, decoder, solutions, makespans


@torch.no_grad()
def evaluate_model(encoder, decoder, val_set, num_sols=32):
    """
    Evaluate the model on validation instances.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        val_set: Validation dataset
        num_sols: Number of solutions to sample (increased for better diversity)
        
    Returns:
        float: Average optimality gap (percentage)
    """
    encoder.eval()
    decoder.eval()
    gaps = []
    
    for ins in val_set:
        # Sample solutions with higher batch size for better exploration
        sols, mss = stg.sampling(ins, encoder, decoder, bs=num_sols, device=device)
        
        # Try both sampling and greedy approaches
        greedy_sol, greedy_ms = stg.greedy(ins, encoder, decoder, device=device)
        
        # Use the best of both approaches
        min_ms = min(mss.min().item(), greedy_ms.item())
        
        # Calculate gap relative to known optimal/best makespan
        gap = (min_ms / ins['makespan'] - 1) * 100
        gaps.append(gap)
    
    # Reset to training mode
    encoder.train()
    decoder.train()
    
    return np.mean(gaps)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Self-Labeling Pre-training for JSP (Standalone)')
    parser.add_argument('--data_path', type=str, default='./dataset5k_subset',
                        help='Path to the training data')
    parser.add_argument('--val_path', type=str, default='../Job_Shop_Scheduling/SelfLabelingJobShop/benchmarks/validation',
                        help='Path to validation instances')
    parser.add_argument('--model_path', type=str, default='./checkpoints/pretrained_model_standalone.pt',
                        help='Path to save the pre-trained model')
    parser.add_argument('--num_instances', type=int, default=100,
                        help='Number of instances to use for pre-training (0 for all). This samples from the subset created by prepare_subset.py.')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of self-labeling iterations')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for pre-training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs per iteration')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension of the models')
    parser.add_argument('--embed_dim', type=int, default=128, 
                        help='Embedding dimension of the models')
    parser.add_argument('--num_heads', type=int, default=3,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='Dropout rate')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load datasets
    print(f"Loading training dataset from {args.data_path}")
    if not os.path.exists(args.data_path):
        print(f"Error: Dataset path does not exist: {args.data_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir(os.path.dirname(args.data_path) or '.')}")
        sys.exit(1)
    
    # Check if there's anything in the directory
    files = [f for f in os.listdir(args.data_path) if f.endswith('.jsp')]
    if not files:
        print(f"Error: No .jsp files found in {args.data_path}")
        print(f"Files in directory: {os.listdir(args.data_path)}")
        sys.exit(1)
    
    print(f"Found {len(files)} .jsp files in {args.data_path}")
    train_set = load_dataset(args.data_path, device=device)
    
    # Load validation set if provided
    val_set = None
    if args.val_path:
        print(f"Loading validation dataset from {args.val_path}")
        if not os.path.exists(args.val_path):
            print(f"Error: Validation path does not exist: {args.val_path}")
            print(f"Skipping validation")
        else:
            val_set = load_dataset(args.val_path, device=device)
    
    # Limit number of instances if specified
    if args.num_instances > 0 and args.num_instances < len(train_set):
        print(f"Using a random sample of {args.num_instances} instances out of {len(train_set)}")
        print(f"Note: The full dataset was created with prepare_subset.py for diverse problem sizes")
        print(f"      Increasing --num_instances will use more of this diverse dataset")
        indices = random.sample(range(len(train_set)), args.num_instances)
        train_set = [train_set[i] for i in indices]
        
        # Analyze the sampled dataset diversity
        size_groups = {}
        for instance in train_set:
            size = f"{instance['j']}x{instance['m']}"
            if size not in size_groups:
                size_groups[size] = 0
            size_groups[size] += 1
            
        print(f"Size distribution in sampled dataset:")
        for size, count in sorted(size_groups.items()):
            print(f"  {size}: {count} instances")
    
    # Run self-labeling pre-training
    start_time = time.time()
    self_labeling_pretrain_standalone(train_set,
                                      hidden_dim=args.hidden_dim,
                                      embed_dim=args.embed_dim,
                                      num_heads=args.num_heads,
                                      dropout=args.dropout,
                                      num_iterations=args.iterations,
                                      batch_size=args.batch_size,
                                      lr=args.lr,
                                      epochs_per_iteration=args.epochs,
                                      model_path=args.model_path,
                                      val_set=val_set,
                                      seed=args.seed)
    
    elapsed_time = time.time() - start_time
    print(f"Self-labeling pre-training completed in {elapsed_time:.2f} seconds") 