"""
Best known solutions for job shop scheduling benchmark instances.
This module provides utilities for gap calculation against known optimal or best solutions.
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
            os.path.join(current_dir, '..', 'benchmark_ub_values.json'),        # OriginalPPO directory
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
    Get the best known solution for a benchmark.
    
    Args:
        benchmark_name: Name of the benchmark
        
    Returns:
        Best known makespan for the benchmark, or None if not available
    """
    return BEST_KNOWN_SOLUTIONS.get(benchmark_name, None)

def calculate_optimality_gap(makespan, benchmark_name):
    """
    Calculate the optimality gap as a percentage.
    
    Args:
        makespan: The makespan achieved by the algorithm
        benchmark_name: Name of the benchmark
        
    Returns:
        Gap percentage, or None if the best known solution is not available
    """
    best_known = get_best_known_solution(benchmark_name)
    if best_known is None or makespan is None:
        return None
    
    # Calculate gap as a percentage
    gap = (makespan / best_known - 1) * 100
    return gap 