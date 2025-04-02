#!/usr/bin/env python3
"""
Script to prepare a subset of the dataset for pretraining.
This creates a directory with a diverse subset of JSP files with different sizes.
"""

import os
import random
import shutil
import argparse
import re
from pathlib import Path
from collections import defaultdict

def create_dataset_subset(source_dir, target_dir, num_instances=500):
    """
    Create a diverse subset of JSP files in a new directory.
    
    Args:
        source_dir: Source directory containing JSP files
        target_dir: Target directory to copy files to
        num_instances: Number of instances to include in the subset
    """
    # Ensure source directory exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Get all JSP files (ignore non-jsp files)
    jsp_files = [f for f in os.listdir(source_dir) if f.endswith('.jsp')]
    if not jsp_files:
        raise ValueError(f"No JSP files found in {source_dir}")
    
    print(f"Found {len(jsp_files)} JSP files in {source_dir}")
    
    # Group files by size (extract mxn from filenames like "10x10_1.jsp")
    size_pattern = re.compile(r"(\d+)x(\d+)_")
    size_groups = defaultdict(list)
    
    for file in jsp_files:
        match = size_pattern.search(file)
        if match:
            size = f"{match.group(1)}x{match.group(2)}"
            size_groups[size].append(file)
        else:
            # For files without clear size pattern
            size_groups["unknown"].append(file)
    
    print(f"Found files in these size groups: {list(size_groups.keys())}")
    
    # Calculate how many to take from each group
    total_sizes = len(size_groups)
    files_per_size = num_instances // total_sizes
    
    # Adjust if we can't take evenly from all groups
    if files_per_size == 0:
        files_per_size = 1
        print(f"Warning: Too many size groups. Taking only {total_sizes} files instead of {num_instances}.")
    
    # Select files from each size group
    selected_files = []
    for size, files in size_groups.items():
        # How many to take from this group
        take_count = min(files_per_size, len(files))
        if take_count < files_per_size:
            print(f"Warning: Size group {size} has only {len(files)} files, taking all of them.")
        
        # Randomly select files from this group
        selected = random.sample(files, take_count)
        selected_files.extend(selected)
        print(f"Selected {len(selected)} files from size group {size}")
    
    # If we need more files to reach the target
    remaining = num_instances - len(selected_files)
    if remaining > 0:
        # Flatten all remaining files
        remaining_files = [f for size_files in size_groups.values() 
                          for f in size_files if f not in selected_files]
        
        # Take additional files if available
        if remaining_files:
            take_count = min(remaining, len(remaining_files))
            additional = random.sample(remaining_files, take_count)
            selected_files.extend(additional)
            print(f"Selected {take_count} additional files to reach target count")
    
    print(f"Selected a total of {len(selected_files)} files")
    
    # Create target directory if it doesn't exist, or clean it if it does
    if os.path.exists(target_dir):
        print(f"Cleaning existing target directory: {target_dir}")
        # Remove all JSP files and cached.pt
        for file in os.listdir(target_dir):
            if file.endswith('.jsp') or file.endswith('.pt'):
                file_path = os.path.join(target_dir, file)
                os.remove(file_path)
                print(f"Removed {file_path}")
    else:
        os.makedirs(target_dir, exist_ok=True)
    
    # Copy files
    for file in selected_files:
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(target_dir, file)
        shutil.copy2(source_path, target_path)
    
    print(f"Copied {len(selected_files)} files to {target_dir}")
    return len(selected_files)

def main():
    parser = argparse.ArgumentParser(description='Create a diverse subset of JSP files for pretraining')
    parser.add_argument('--source', type=str, required=True,
                       help='Source directory containing JSP files')
    parser.add_argument('--target', type=str, required=True,
                       help='Target directory to copy files to')
    parser.add_argument('--num', type=int, default=500,
                       help='Number of instances to include in the subset')
    args = parser.parse_args()
    
    create_dataset_subset(args.source, args.target, args.num)
    
if __name__ == '__main__':
    main() 