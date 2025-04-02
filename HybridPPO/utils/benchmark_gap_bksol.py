#!/usr/bin/env python3
"""
Utilities for calculating optimality gaps in job shop scheduling.

This module provides best-known solutions for common job shop scheduling
benchmarks and functions to calculate optimality gaps.
"""

import os
import json

# Load best known solutions from JSON file
def load_best_known_solutions():
    """Load best known solutions from the JSON file"""
    try:
        # Get the directory of this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Try to find the JSON file in various locations
        json_paths = [
            os.path.join(current_dir, '..', '..', 'benchmark_ub_values.json'),  # Project root
            os.path.join(current_dir, '..', 'benchmark_ub_values.json'),        # HybridPPO directory
            os.path.join(current_dir, 'benchmark_ub_values.json')               # utils directory
        ]
        
        for json_path in json_paths:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    return json.load(f)
        
        # If JSON file is not found in any location, raise an error
        print("Error: benchmark_ub_values.json not found in any expected location")
        return {}
    except Exception as e:
        print(f"Error loading best known solutions: {e}")
        return {}

# Load solutions at module import time
BEST_KNOWN_SOLUTIONS = load_best_known_solutions()

def get_best_known_solution(benchmark_name):
    """
    Get the best-known solution for a specific benchmark.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., 'ta01')
        
    Returns:
        The best-known makespan value or None if not found
    """
    return BEST_KNOWN_SOLUTIONS.get(benchmark_name)

def calculate_optimality_gap(makespan, benchmark_name):
    """
    Calculate the optimality gap between the given makespan and the best-known solution.
    
    Gap is calculated as: (makespan / best_known - 1) * 100 (percentage)
    
    Args:
        makespan: The makespan value to evaluate
        benchmark_name: Name of the benchmark (e.g., 'ta01')
        
    Returns:
        The optimality gap as a percentage, or None if best-known solution is not available
    """
    best_known = get_best_known_solution(benchmark_name)
    if best_known is None:
        return None
    
    gap = (makespan / best_known - 1) * 100
    return gap

def calculate_average_gap(makespans, benchmark_names):
    """
    Calculate the average optimality gap for multiple instances.
    
    Args:
        makespans: List of makespan values
        benchmark_names: List of benchmark names corresponding to the makespans
        
    Returns:
        The average gap as a percentage, or None if no gaps could be calculated
    """
    if len(makespans) != len(benchmark_names):
        raise ValueError("Length of makespans and benchmark_names must be the same")
    
    gaps = []
    for makespan, benchmark in zip(makespans, benchmark_names):
        gap = calculate_optimality_gap(makespan, benchmark)
        if gap is not None:
            gaps.append(gap)
    
    if not gaps:
        return None
    
    return sum(gaps) / len(gaps) 