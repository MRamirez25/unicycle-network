#!/usr/bin/env python3
"""
Evaluate NARMA performance across multiple runs of the same configuration.

This script:
1. Loads multiple robot data files recorded at the same period ratio
2. Evaluates NARMA performance for each run
3. Computes mean and standard deviation of test MSE/NMSE
4. Outputs statistics table and optional plots
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Gyre Termes"],
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 16})

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from plot_robot_data import load_robot_data, extract_robot_ids, get_robot_states

#=============================================================================
# CONFIGURATION - Edit these parameters
#=============================================================================

# Data files: list of files recorded at the same configuration
DATA_FILES = [
    "period80-retakes/time-period-80-retake1.csv.gz",
    "period80-retakes/time-period-80-retake2.csv.gz",
    "period80-retakes/time-period-80-retake3.csv.gz",
    "period80-retakes/time-period-80-retake4.csv.gz",
    "period80-retakes/time-period-80-retake5.csv.gz"
]

# Configuration description (for labeling)
CONFIG_NAME = "Period 80"  # Human-readable name for this configuration

# NARMA orders to evaluate
NARMA_ORDERS = [2, 3, 5]

# Ridge regression parameters
RIDGE_ALPHA = 300  # Regularization strength
RIDGE_SOLVER = 'auto'  # Solver

# Data processing parameters
WASHOUT_FRACTION = 0.1  # Fraction of data to discard as washout (from beginning)
CUTOFF_FRACTION = 0.1  # Fraction of data to discard from the end (0.0 = no cutoff)
TRAIN_FRACTION = 0.8  # Fraction of remaining data for training
N_ROBOTS = 10  # Number of robots to use

# Feature configuration
FEATURE_NAMES = ['pos_x', 'pos_y']
USE_LOCAL_FRAME = False

# Smoothing parameters
SMOOTH_WINDOW_SIZE = None
SMOOTH_METHOD = None

# NARMA generation parameters
NARMA_AMPLITUDE = 0.5
DT = 0.05

# Output
OUTPUT_DIR = "narma_multiple_runs_results"
SAVE_RESULTS = True
SAVE_PLOTS = True  # Set to True to save individual run plots

#=============================================================================
# NARMA generation functions
#=============================================================================

def u(t, period_ratio=2, amplitude=0.2):
    """Input signal: sum of three sinusoids"""
    f1 = 2.11
    f2 = 3.73
    f3 = 4.33
    T = period_ratio
    u_val = amplitude * np.sin(2*np.pi * f1 * t / T) * np.sin(2*np.pi * f2 * t / T) * np.sin(2*np.pi * f3 * t / T)
    return u_val


def generate_narma(n_samples, narma_order=2, period_ratio=2, amplitude=0.2, dt=0.01):
    """
    Generate NARMA-n time series.
    
    Returns:
        u_input: Raw input signal
        u_normalized: Normalized input [0, 0.5] used for NARMA generation
        y_narma: NARMA output series
    """
    # Generate input signal
    t = np.arange(n_samples) * dt
    u_input = np.array([u(ti, period_ratio=period_ratio, amplitude=amplitude) for ti in t])
    
    # Normalize input to [0, 0.5] for NARMA generation
    u_normalized = u_input - u_input.min()
    u_normalized = 0.5 * u_normalized / u_normalized.max()
    
    # Initialize NARMA output
    y_narma = np.zeros(n_samples)
    
    # Generate NARMA series based on order
    if narma_order == 2:
        for k in range(1, n_samples - 1):
            y_narma[k + 1] = (0.4 * y_narma[k] + 
                            0.4 * y_narma[k] * y_narma[k - 1] + 
                            0.6 * u_normalized[k]**3 + 0.1)
                            
    elif narma_order == 3:
        a, b, c, d = 0.3, 0.05, 1.5, 0.1
        for k in range(2, n_samples - 1):
            y_sum = np.sum(y_narma[k - 2:k + 1])
            y_narma[k + 1] = (a * y_narma[k] + 
                            b * y_narma[k] * y_sum + 
                            c * u_normalized[k - 2] * u_normalized[k] + d)
                            
    else:
        # NARMA-n for n >= 4
        n = narma_order
        a, b, c, d = 0.3, 0.05, 1.5, 0.1
        for k in range(n - 1, n_samples - 1):
            y_sum = np.sum(y_narma[k - (n - 1):k + 1])
            y_narma[k + 1] = (a * y_narma[k] + 
                            b * y_narma[k] * y_sum + 
                            c * u_normalized[k - n + 1] * u_normalized[k] + d)
    
    # Check for numerical issues
    if np.max(np.abs(y_narma)) > 10:
        print(f'WARNING: NARMA output exceeded safe range (max={np.max(np.abs(y_narma)):.2f}). Normalizing...')
        y_narma = y_narma / np.max(np.abs(y_narma))
    
    return u_input, u_normalized, y_narma


def extract_robot_features(df, robot_ids, feature_names, use_local_frame=False, washout_idx=0):
    """Extract feature matrix from robot states."""
    all_features = []
    
    for robot_id in robot_ids:
        states = get_robot_states(df, robot_id, verbose=False)
        
        # Apply local frame transformation if requested
        if use_local_frame and washout_idx < len(states['pos_x']):
            states['pos_x'] = states['pos_x'] - states['pos_x'][washout_idx]
            states['pos_y'] = states['pos_y'] - states['pos_y'][washout_idx]
        
        # Extract requested features
        robot_feature_list = [states[fname] for fname in feature_names]
        robot_features = np.stack(robot_feature_list, axis=1)
        all_features.append(robot_features)
    
    # Concatenate all robot features
    features = np.concatenate(all_features, axis=1)
    return features


def smooth_features(features, window_size=5, method='gaussian'):
    """Smooth features to reduce noise."""
    from scipy.ndimage import uniform_filter1d, gaussian_filter1d
    from scipy.signal import savgol_filter
    
    smoothed = features.copy()
    
    if method == 'moving_average':
        for col_idx in range(features.shape[1]):
            smoothed[:, col_idx] = uniform_filter1d(features[:, col_idx], size=window_size, mode='nearest')
    
    elif method == 'gaussian':
        sigma = window_size / 3.0
        for col_idx in range(features.shape[1]):
            smoothed[:, col_idx] = gaussian_filter1d(features[:, col_idx], sigma=sigma, mode='nearest')
    
    elif method == 'savgol':
        if window_size % 2 == 0:
            window_size += 1
        polyorder = min(3, window_size - 2)
        for col_idx in range(features.shape[1]):
            smoothed[:, col_idx] = savgol_filter(features[:, col_idx], window_length=window_size, 
                                                polyorder=polyorder, mode='nearest')
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    return smoothed


#=============================================================================
# Main evaluation function
#=============================================================================

def evaluate_narma_for_run(data_file, narma_order, run_idx):
    """
    Evaluate NARMA prediction for a single run.
    
    Returns:
        results: dict with 'test_nmse', 'test_mse', 'train_nmse', 'train_mse', 'success'
    """
    try:
        print(f"  Run {run_idx+1}: {data_file}")
        
        # Load robot data
        df = load_robot_data(data_file)
        robot_ids = extract_robot_ids(df)
        
        if len(robot_ids) < N_ROBOTS:
            print(f"    WARNING: Only {len(robot_ids)} robots found, need {N_ROBOTS}")
            return {'test_nmse': np.nan, 'test_mse': np.nan, 'train_nmse': np.nan, 'train_mse': np.nan, 'success': False}
        
        robot_ids = robot_ids[:N_ROBOTS]
        
        # Extract features
        features = extract_robot_features(df, robot_ids, FEATURE_NAMES, USE_LOCAL_FRAME, 0)
        n_samples = features.shape[0]
        
        # Extract global_u from robot data
        states = get_robot_states(df, robot_ids[0], verbose=False)
        global_u = states['global_u']
        
        if global_u is None:
            print("    ERROR: global_u not found in data file")
            return {'test_nmse': np.nan, 'test_mse': np.nan, 'train_nmse': np.nan, 'train_mse': np.nan, 'success': False}
        
        # Normalize global_u to [0, 0.5] for NARMA generation
        u_normalized = global_u - global_u.min()
        u_normalized = 0.5 * u_normalized / u_normalized.max()
        
        # Generate NARMA with actual input
        y_narma = np.zeros(n_samples)
        
        if narma_order == 2:
            for k in range(1, n_samples - 1):
                y_narma[k + 1] = (0.4 * y_narma[k] + 
                                0.4 * y_narma[k] * y_narma[k - 1] + 
                                0.6 * u_normalized[k]**3 + 0.1)
        elif narma_order == 3:
            a, b, c, d = 0.3, 0.05, 1.5, 0.1
            for k in range(2, n_samples - 1):
                y_sum = np.sum(y_narma[k - 2:k + 1])
                y_narma[k + 1] = (a * y_narma[k] + 
                                b * y_narma[k] * y_sum + 
                                c * u_normalized[k - 2] * u_normalized[k] + d)
        else:
            n = narma_order
            a, b, c, d = 0.3, 0.05, 1.5, 0.1
            for k in range(n - 1, n_samples - 1):
                y_sum = np.sum(y_narma[k - (n - 1):k + 1])
                y_narma[k + 1] = (a * y_narma[k] + 
                                b * y_narma[k] * y_sum + 
                                c * u_normalized[k - n + 1] * u_normalized[k] + d)
        
        # Split data
        washout_samples = int(WASHOUT_FRACTION * n_samples)
        cutoff_samples = int(CUTOFF_FRACTION * n_samples)
        
        # Determine valid range: after washout, before cutoff
        end_idx = n_samples - cutoff_samples if cutoff_samples > 0 else n_samples
        valid_idx = np.arange(washout_samples, end_idx)
        n_valid = len(valid_idx)
        n_train = int(TRAIN_FRACTION * n_valid)
        
        train_idx = valid_idx[:n_train]
        test_idx = valid_idx[n_train:]
        
        print(f"    Data split: washout={washout_samples}, valid={n_valid}, cutoff={cutoff_samples}, train={len(train_idx)}, test={len(test_idx)}")
        
        # Extract train/test features and targets
        X_train = features[train_idx, :]
        y_train = y_narma[train_idx]
        X_test = features[test_idx, :]
        y_test = y_narma[test_idx]
        
        # Apply smoothing if enabled
        if SMOOTH_WINDOW_SIZE is not None:
            X_train = smooth_features(X_train, window_size=SMOOTH_WINDOW_SIZE, method=SMOOTH_METHOD)
            X_test = smooth_features(X_test, window_size=SMOOTH_WINDOW_SIZE, method=SMOOTH_METHOD)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Ridge regression
        regressor = Ridge(alpha=RIDGE_ALPHA, solver=RIDGE_SOLVER, fit_intercept=True)
        regressor.fit(X_train_scaled, y_train)
        
        # Evaluate on train set
        train_pred = regressor.predict(X_train_scaled)
        train_mse = np.mean((train_pred - y_train)**2)
        train_var = np.var(y_train)
        train_nmse = train_mse / (train_var + 1e-9)
        
        # Evaluate on test set
        test_pred = regressor.predict(X_test_scaled)
        test_mse = np.mean((test_pred - y_test)**2)
        test_var = np.var(y_test)
        test_nmse = test_mse / (test_var + 1e-9)
        
        print(f"    Train NMSE: {train_nmse:.6f}, Test NMSE: {test_nmse:.6f}")
        
        return {
            'test_nmse': test_nmse,
            'test_mse': test_mse,
            'train_nmse': train_nmse,
            'train_mse': train_mse,
            'success': True
        }
        
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'test_nmse': np.nan, 'test_mse': np.nan, 'train_nmse': np.nan, 'train_mse': np.nan, 'success': False}


#=============================================================================
# Main execution
#=============================================================================

def main():
    """Main execution: evaluate all runs and compute statistics."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("NARMA EVALUATION - MULTIPLE RUNS STATISTICS")
    print("="*80)
    print(f"\nConfiguration: {CONFIG_NAME}")
    print(f"  Number of runs: {len(DATA_FILES)}")
    print(f"  NARMA orders: {NARMA_ORDERS}")
    print(f"  Ridge alpha: {RIDGE_ALPHA}")
    print(f"  Features: {FEATURE_NAMES}")
    print(f"  Smoothing: {SMOOTH_METHOD if SMOOTH_WINDOW_SIZE else 'None'}")
    print(f"  Data processing: washout={WASHOUT_FRACTION*100:.1f}%, cutoff={CUTOFF_FRACTION*100:.1f}%, train={TRAIN_FRACTION*100:.1f}%")
    print("="*80)
    
    # Store results for each NARMA order
    all_results = {order: [] for order in NARMA_ORDERS}
    
    # Evaluate all runs for each NARMA order
    for narma_order in NARMA_ORDERS:
        print(f"\nNARMA-{narma_order}:")
        print("-" * 40)
        
        for run_idx, data_file in enumerate(DATA_FILES):
            if not os.path.exists(data_file):
                print(f"  WARNING: File not found: {data_file}")
                all_results[narma_order].append({
                    'test_nmse': np.nan, 'test_mse': np.nan, 
                    'train_nmse': np.nan, 'train_mse': np.nan, 
                    'success': False, 'file': data_file
                })
                continue
            
            result = evaluate_narma_for_run(data_file, narma_order, run_idx)
            result['file'] = data_file
            all_results[narma_order].append(result)
    
    # Compute statistics
    print("\n" + "="*80)
    print("STATISTICS SUMMARY")
    print("="*80)
    
    stats_rows = []
    
    for narma_order in NARMA_ORDERS:
        results = all_results[narma_order]
        
        # Extract successful results
        test_nmse_values = [r['test_nmse'] for r in results if r['success'] and not np.isnan(r['test_nmse'])]
        test_mse_values = [r['test_mse'] for r in results if r['success'] and not np.isnan(r['test_mse'])]
        train_nmse_values = [r['train_nmse'] for r in results if r['success'] and not np.isnan(r['train_nmse'])]
        train_mse_values = [r['train_mse'] for r in results if r['success'] and not np.isnan(r['train_mse'])]
        
        n_successful = len(test_nmse_values)
        
        if n_successful == 0:
            print(f"\nNARMA-{narma_order}: No successful runs")
            stats_rows.append({
                'narma_order': narma_order,
                'n_runs': len(results),
                'n_successful': 0,
                'test_nmse_mean': np.nan,
                'test_nmse_std': np.nan,
                'test_mse_mean': np.nan,
                'test_mse_std': np.nan,
                'train_nmse_mean': np.nan,
                'train_nmse_std': np.nan,
                'train_mse_mean': np.nan,
                'train_mse_std': np.nan
            })
            continue
        
        # Compute statistics
        test_nmse_mean = np.mean(test_nmse_values)
        test_nmse_std = np.std(test_nmse_values)
        test_mse_mean = np.mean(test_mse_values)
        test_mse_std = np.std(test_mse_values)
        train_nmse_mean = np.mean(train_nmse_values)
        train_nmse_std = np.std(train_nmse_values)
        train_mse_mean = np.mean(train_mse_values)
        train_mse_std = np.std(train_mse_values)
        
        print(f"\nNARMA-{narma_order} ({n_successful}/{len(results)} successful runs):")
        print(f"  Test NMSE:  {test_nmse_mean:.6f} ± {test_nmse_std:.6f}")
        print(f"  Test MSE:   {test_mse_mean:.6f} ± {test_mse_std:.6f}")
        print(f"  Train NMSE: {train_nmse_mean:.6f} ± {train_nmse_std:.6f}")
        print(f"  Train MSE:  {train_mse_mean:.6f} ± {train_mse_std:.6f}")
        
        stats_rows.append({
            'narma_order': narma_order,
            'n_runs': len(results),
            'n_successful': n_successful,
            'test_nmse_mean': test_nmse_mean,
            'test_nmse_std': test_nmse_std,
            'test_mse_mean': test_mse_mean,
            'test_mse_std': test_mse_std,
            'train_nmse_mean': train_nmse_mean,
            'train_nmse_std': train_nmse_std,
            'train_mse_mean': train_mse_mean,
            'train_mse_std': train_mse_std
        })
    
    # Save statistics to CSV
    if SAVE_RESULTS:
        stats_df = pd.DataFrame(stats_rows)
        stats_path = os.path.join(OUTPUT_DIR, f'{CONFIG_NAME.replace(" ", "_")}_statistics.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"\n✓ Statistics saved to: {stats_path}")
        
        # Also save individual run results
        all_runs_rows = []
        for narma_order in NARMA_ORDERS:
            for run_idx, result in enumerate(all_results[narma_order]):
                all_runs_rows.append({
                    'narma_order': narma_order,
                    'run': run_idx + 1,
                    'file': result['file'],
                    'success': result['success'],
                    'test_nmse': result['test_nmse'],
                    'test_mse': result['test_mse'],
                    'train_nmse': result['train_nmse'],
                    'train_mse': result['train_mse']
                })
        
        runs_df = pd.DataFrame(all_runs_rows)
        runs_path = os.path.join(OUTPUT_DIR, f'{CONFIG_NAME.replace(" ", "_")}_all_runs.csv')
        runs_df.to_csv(runs_path, index=False)
        print(f"✓ Individual run results saved to: {runs_path}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
