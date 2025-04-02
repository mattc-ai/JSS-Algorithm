#!/usr/bin/env python3
"""
Script to convert LA and DMU benchmark files from SelfLabelingJobShop format to JSSEnv format
and copy them to the JSSEnv instances directory.
"""

import os
import argparse
import shutil

def convert_jsp_to_jssenv_format(jsp_file, output_file):
    """
    Convert a JSP file from SelfLabelingJobShop format to JSSEnv format.
    
    Args:
        jsp_file: Path to the JSP file
        output_file: Path to save the converted file
    
    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        # Read the JSP file
        with open(jsp_file, 'r') as f:
            lines = f.readlines()
        
        # Extract the number of jobs and machines
        first_line = lines[0].strip().split()
        num_jobs = int(first_line[0])
        num_machines = int(first_line[1])
        
        # Extract the job information (machine, processing time)
        job_lines = []
        for i in range(1, num_jobs + 1):
            job_line = lines[i].strip().split()
            job_info = []
            for j in range(0, len(job_line), 2):
                machine = int(job_line[j])
                proc_time = int(job_line[j + 1])
                job_info.append(f"{machine} {proc_time}")
            job_lines.append(" ".join(job_info))
        
        # Create the converted file
        with open(output_file, 'w') as f:
            f.write(f"{num_jobs} {num_machines}\n")
            for job_line in job_lines:
                f.write(f"{job_line}\n")
        
        return True
    
    except Exception as e:
        print(f"Error converting {jsp_file}: {e}")
        return False

def convert_single_benchmark(benchmark_name, benchmark_series, source_dir, target_dir):
    """
    Convert a single benchmark file and save it to the target directory.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., 'ta01')
        benchmark_series: Series of the benchmark (TA, LA, DMU)
        source_dir: Directory containing the benchmark files
        target_dir: Directory to save the converted file
    
    Returns:
        True if conversion was successful, False otherwise
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Construct file paths
    jsp_file = f"{benchmark_name}.jsp"
    source_path = os.path.join(source_dir, jsp_file)
    target_path = os.path.join(target_dir, benchmark_name)
    
    # Check if source file exists
    if not os.path.exists(source_path):
        print(f"Error: Benchmark file {source_path} not found")
        return False
    
    # Convert the file
    print(f"Converting {benchmark_series} benchmark: {benchmark_name}")
    if convert_jsp_to_jssenv_format(source_path, target_path):
        print(f"  Success: {target_path}")
        return True
    else:
        print(f"  Failed: {benchmark_name}")
        return False

def convert_benchmarks(source_dir, target_dir, benchmark_type):
    """
    Convert all benchmark files in the source directory and copy them to the target directory.
    
    Args:
        source_dir: Directory containing the benchmark files
        target_dir: Directory to save the converted files
        benchmark_type: Type of benchmark (LA or DMU)
    
    Returns:
        Number of successfully converted files
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all JSP files in the source directory
    jsp_files = [f for f in os.listdir(source_dir) if f.endswith('.jsp')]
    
    # Sort files to process them in order
    jsp_files.sort()
    
    # Counter for successful conversions
    success_count = 0
    
    # Process each file
    for jsp_file in jsp_files:
        # Get the benchmark name (without extension)
        benchmark_name = os.path.splitext(jsp_file)[0].lower()
        
        # Source and target paths
        source_path = os.path.join(source_dir, jsp_file)
        target_path = os.path.join(target_dir, benchmark_name)
        
        # Convert the file
        print(f"Converting {benchmark_type} benchmark: {benchmark_name}")
        if convert_jsp_to_jssenv_format(source_path, target_path):
            success_count += 1
            print(f"  Success: {target_path}")
        else:
            print(f"  Failed: {benchmark_name}")
    
    return success_count

def main():
    parser = argparse.ArgumentParser(description='Convert LA and DMU benchmark files to JSSEnv format')
    parser.add_argument('--la_dir', type=str, default='Job_Shop_Scheduling/SelfLabelingJobShop/benchmarks/LA',
                      help='Directory containing LA benchmark files')
    parser.add_argument('--dmu_dir', type=str, default='Job_Shop_Scheduling/SelfLabelingJobShop/benchmarks/DMU',
                      help='Directory containing DMU benchmark files')
    parser.add_argument('--ta_dir', type=str, default='Job_Shop_Scheduling/SelfLabelingJobShop/benchmarks/TA',
                      help='Directory containing TA benchmark files')
    parser.add_argument('--target_dir', type=str, default='Job_Shop_Scheduling/JSSEnv/JSSEnv/envs/instances',
                      help='Directory to save the converted files')
    parser.add_argument('--benchmark', type=str, help='Convert a single benchmark (e.g., ta01)')
    parser.add_argument('--benchmark_series', type=str, choices=['TA', 'LA', 'DMU'], 
                      help='Benchmark series for single conversion (TA, LA, DMU)')
    parser.add_argument('--all', action='store_true', help='Convert all benchmarks from all series')
    args = parser.parse_args()
    
    # Convert a single benchmark if specified
    if args.benchmark and args.benchmark_series:
        source_dir = None
        if args.benchmark_series == 'TA':
            source_dir = args.ta_dir
        elif args.benchmark_series == 'LA':
            source_dir = args.la_dir
        elif args.benchmark_series == 'DMU':
            source_dir = args.dmu_dir
        
        if source_dir:
            success = convert_single_benchmark(args.benchmark, args.benchmark_series, source_dir, args.target_dir)
            if success:
                print(f"\nSuccessfully converted benchmark {args.benchmark}")
            else:
                print(f"\nFailed to convert benchmark {args.benchmark}")
            return
    
    # Convert all benchmarks if requested or no specific benchmark was provided
    if args.all or (not args.benchmark and not args.benchmark_series):
        # Convert TA benchmarks
        print(f"\nConverting TA benchmarks from {args.ta_dir}")
        ta_count = convert_benchmarks(args.ta_dir, args.target_dir, 'TA')
        
        # Convert LA benchmarks
        print(f"\nConverting LA benchmarks from {args.la_dir}")
        la_count = convert_benchmarks(args.la_dir, args.target_dir, 'LA')
        
        # Convert DMU benchmarks
        print(f"\nConverting DMU benchmarks from {args.dmu_dir}")
        dmu_count = convert_benchmarks(args.dmu_dir, args.target_dir, 'DMU')
        
        # Print summary
        print(f"\nConversion complete!")
        print(f"Successfully converted {ta_count} TA benchmarks, {la_count} LA benchmarks, and {dmu_count} DMU benchmarks")
        print(f"Files saved to {args.target_dir}")
    else:
        print("Error: Please specify both --benchmark and --benchmark_series for single benchmark conversion,")
        print("       or use --all to convert all benchmarks.")

if __name__ == '__main__':
    main() 