#!/usr/bin/env python3
"""
Run evaluation statistics scripts with multiple feature slice configurations.

This script automatically runs MNIST and/or CIFAR-10 evaluation statistics
with different feature slicing configurations to compare performance.
"""

import subprocess
import sys
import os
from datetime import datetime
import json
import re
import glob

# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
script_dir = os.path.join(parent_dir, "non_linearity_tests")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Which datasets to run
RUN_MNIST = True
RUN_CIFAR10 = True

# Feature slice configurations to test
# Format: (start, stop, description)
# Use None for stop to indicate "all features from start onwards"

MNIST_SLICES = [
    # (0, None, "all_features"),           # All 400 features (100 units * 4)
    (0, 200, "xy_positions_only"),       # x,y positions only (100 units * 2)
    (0, 100, "x_positions_only"),        # x positions only (100 units * 1)
    (100, 200, "y_positions_only"),      # y positions only (100 units * 1)
    (200, 300, "theta_only"),            # theta angles only (100 units * 1)
    (0, 300, "xy_theta"),                # x, y, theta (100 units * 3)
]

CIFAR10_SLICES = [
    # (0, None, "all_features"),           # All 800 features (200 units * 4)
    (0, 400, "xy_positions_only"),      # x,y positions only (200 units * 2)
    (0, 200, "x_positions_only"),        # x positions only (200 units * 1)
    (200, 400, "y_positions_only"),      # y positions only (200 units * 1)
    (400, 600, "theta_only"),            # theta angles only (200 units * 1)
    (0, 600, "xy_theta"),                # x, y, theta (200 units * 3)
]

# ============================================================================
# SCRIPT MODIFICATION AND EXECUTION
# ============================================================================

def modify_script_slicing(script_path, slice_start, slice_stop):
    """
    Modify the feature slicing configuration in a script.
    
    Args:
        script_path: Path to the evaluation statistics script
        slice_start: Start index for slicing
        slice_stop: Stop index for slicing (None for all features from start)
    """
    with open(script_path, 'r') as f:
        lines = f.readlines()
    
    # Find and modify the FEATURE_SLICE_START and FEATURE_SLICE_STOP lines
    modified = False
    for i, line in enumerate(lines):
        if line.strip().startswith('FEATURE_SLICE_START = '):
            lines[i] = f'FEATURE_SLICE_START = {slice_start}\n'
            modified = True
        elif line.strip().startswith('FEATURE_SLICE_STOP = '):
            lines[i] = f'FEATURE_SLICE_STOP = {slice_stop}\n'
            modified = True
    
    if not modified:
        raise ValueError(f"Could not find FEATURE_SLICE_START/STOP in {script_path}")
    
    # Write back
    with open(script_path, 'w') as f:
        f.writelines(lines)
    
    print(f"  Modified {os.path.basename(script_path)}: slice [{slice_start}:{slice_stop}]")


def extract_results_from_file(dataset_name, timestamp_pattern=None):
    """
    Extract test accuracy and std from the most recent results file.
    
    Args:
        dataset_name: 'mnist' or 'cifar10'
        timestamp_pattern: Optional specific timestamp to look for
    
    Returns:
        Tuple of (test_mean, test_std) or (None, None) if not found
    """
    results_dir = os.path.join(parent_dir, "results")
    pattern = f"{dataset_name}_stats_results_*.txt"
    
    # Find all matching files
    files = glob.glob(os.path.join(results_dir, pattern))
    if not files:
        return None, None
    
    # Get the most recent file
    latest_file = max(files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r') as f:
            content = f.read()
            
        # Look for test accuracy line like "  - Test accuracy: 0.9234 ± 0.0056"
        test_match = re.search(r'Test accuracy:\s+([\d.]+)\s+±\s+([\d.]+)', content)
        
        if test_match:
            test_mean = float(test_match.group(1))
            test_std = float(test_match.group(2))
            return test_mean, test_std
    except Exception as e:
        print(f"  Warning: Could not extract results from {latest_file}: {e}")
    
    return None, None


def run_script(script_path, dataset_name, slice_config):
    """
    Run an evaluation statistics script.
    
    Args:
        script_path: Path to the script to run
        dataset_name: Name of the dataset (for logging)
        slice_config: Tuple of (start, stop, description)
    
    Returns:
        Return code from the subprocess
    """
    slice_start, slice_stop, description = slice_config
    
    print(f"\n{'='*70}")
    print(f"Running {dataset_name} with slice configuration: {description}")
    print(f"  Slice: [{slice_start}:{slice_stop}]")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Modify the script with new slicing configuration
    modify_script_slicing(script_path, slice_start, slice_stop)
    
    # Run the script
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=parent_dir,
            check=False,
            capture_output=False  # Let output go to terminal
        )
        
        if result.returncode == 0:
            print(f"\n✓ Successfully completed {dataset_name} - {description}")
            
            # Extract results from the output file
            test_mean, test_std = extract_results_from_file(dataset_name.lower())
            
            if test_mean is not None:
                print(f"  → Test accuracy: {test_mean:.4f} ± {test_std:.4f}")
                return result.returncode, test_mean, test_std
        else:
            print(f"\n✗ Failed {dataset_name} - {description} (exit code: {result.returncode})")
        
        return result.returncode, None, None
    
    except Exception as e:
        print(f"\n✗ Error running {dataset_name} - {description}: {e}")
        return -1, None, None


def main():
    """Main execution function."""
    print(f"\n{'='*70}")
    print("Multiple Feature Slice Experiment Runner")
    print(f"{'='*70}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {
        'start_time': datetime.now().isoformat(),
        'mnist_results': [],
        'cifar10_results': []
    }
    
    # Run MNIST experiments
    if RUN_MNIST:
        mnist_script = os.path.join(script_dir, "mnist_evaluation_statistics.py")
        
        if not os.path.exists(mnist_script):
            print(f"✗ MNIST script not found: {mnist_script}")
        else:
            print(f"\n{'#'*70}")
            print(f"# MNIST EXPERIMENTS ({len(MNIST_SLICES)} configurations)")
            print(f"{'#'*70}")
            
            for i, slice_config in enumerate(MNIST_SLICES, 1):
                print(f"\n[MNIST {i}/{len(MNIST_SLICES)}]")
                
                returncode, test_mean, test_std = run_script(mnist_script, "MNIST", slice_config)
                
                results['mnist_results'].append({
                    'slice_start': slice_config[0],
                    'slice_stop': slice_config[1],
                    'description': slice_config[2],
                    'returncode': returncode,
                    'success': returncode == 0,
                    'test_accuracy_mean': test_mean,
                    'test_accuracy_std': test_std
                })
    
    # Run CIFAR-10 experiments
    if RUN_CIFAR10:
        cifar10_script = os.path.join(script_dir, "cifar10_evaluation_statistics.py")
        
        if not os.path.exists(cifar10_script):
            print(f"✗ CIFAR-10 script not found: {cifar10_script}")
        else:
            print(f"\n{'#'*70}")
            print(f"# CIFAR-10 EXPERIMENTS ({len(CIFAR10_SLICES)} configurations)")
            print(f"{'#'*70}")
            
            for i, slice_config in enumerate(CIFAR10_SLICES, 1):
                print(f"\n[CIFAR-10 {i}/{len(CIFAR10_SLICES)}]")
                
                returncode, test_mean, test_std = run_script(cifar10_script, "cifar10", slice_config)
                
                results['cifar10_results'].append({
                    'slice_start': slice_config[0],
                    'slice_stop': slice_config[1],
                    'description': slice_config[2],
                    'returncode': returncode,
                    'success': returncode == 0,
                    'test_accuracy_mean': test_mean,
                    'test_accuracy_std': test_std
                })
    
    # Save results summary
    results['end_time'] = datetime.now().isoformat()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(parent_dir, "results", f"slice_experiments_summary_{timestamp}.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    
    if RUN_MNIST:
        mnist_success = sum(1 for r in results['mnist_results'] if r['success'])
        print(f"\nMNIST: {mnist_success}/{len(results['mnist_results'])} successful")
        for r in results['mnist_results']:
            status = "✓" if r['success'] else "✗"
            acc_str = f" - Test: {r['test_accuracy_mean']:.4f}±{r['test_accuracy_std']:.4f}" if r['test_accuracy_mean'] is not None else ""
            print(f"  {status} [{r['slice_start']}:{r['slice_stop']}] - {r['description']}{acc_str}")
    
    if RUN_CIFAR10:
        cifar10_success = sum(1 for r in results['cifar10_results'] if r['success'])
        print(f"\nCIFAR-10: {cifar10_success}/{len(results['cifar10_results'])} successful")
        for r in results['cifar10_results']:
            status = "✓" if r['success'] else "✗"
            acc_str = f" - Test: {r['test_accuracy_mean']:.4f}±{r['test_accuracy_std']:.4f}" if r['test_accuracy_mean'] is not None else ""
            print(f"  {status} [{r['slice_start']}:{r['slice_stop']}] - {r['description']}{acc_str}")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
