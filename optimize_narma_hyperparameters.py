#!/usr/bin/env python3
"""
Optimize NARMA hyperparameters using Optuna.

This script:
1. Uses Optuna to optimize washout_fraction, cutoff_fraction, and ridge_alpha
2. Evaluates NARMA performance for a given configuration
3. Minimizes test NMSE across specified period ratios and NARMA orders
4. Saves best hyperparameters and optimization history
"""

import os
import sys
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from plot_robot_data import load_robot_data, extract_robot_ids, get_robot_states

#=============================================================================
# CONFIGURATION - Edit these parameters
#=============================================================================

# Data files to use for optimization
# Note: For multiple runs of same configuration, use different keys (can be arbitrary numbers)
DATA_FILES = {
    1: "period80-retakes/time-period-80-retake1.csv.gz",
    2: "period80-retakes/time-period-80-retake2.csv.gz",
    3: "period80-retakes/time-period-80-retake3.csv.gz",
    4: "period80-retakes/time-period-80-retake4.csv.gz",
    5: "period80-retakes/time-period-80-retake5.csv.gz",
}

# NARMA orders to optimize for
NARMA_ORDERS = [2, 3, 5]

# Optuna optimization parameters
N_TRIALS = 200  # Number of optimization trials
STUDY_NAME = "narma_hyperparameter_optimization"
STORAGE = None  # Set to "sqlite:///optuna_narma.db" to persist studies

# Hyperparameter search ranges
WASHOUT_FRACTION_RANGE = (0.05, 0.30)  # Min and max washout fraction
CUTOFF_FRACTION_RANGE = (0.0, 0.20)  # Min and max cutoff fraction
RIDGE_ALPHA_RANGE = (1e-2, 1e4)  # Min and max ridge alpha (log scale)

# Fixed parameters
TRAIN_FRACTION = 0.8  # Fraction of valid data for training
N_ROBOTS = 10  # Number of robots to use
RIDGE_SOLVER = 'auto'  # Ridge solver

# Feature configuration
FEATURE_NAMES = ['pos_x', 'pos_y']
USE_LOCAL_FRAME = False

# Smoothing (disabled for optimization)
SMOOTH_WINDOW_SIZE = None
SMOOTH_METHOD = None

# NARMA parameters
DT = 0.05

# Output
OUTPUT_DIR = "narma_optimization_results"

#=============================================================================
# NARMA generation functions
#=============================================================================

def generate_narma_from_input(u_normalized, narma_order):
    """
    Generate NARMA-n time series from normalized input.
    
    Args:
        u_normalized: Normalized input [0, 0.5]
        narma_order: NARMA order
    
    Returns:
        y_narma: NARMA output series
    """
    n_samples = len(u_normalized)
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
        # NARMA-n for n >= 4
        n = narma_order
        a, b, c, d = 0.3, 0.05, 1.5, 0.1
        for k in range(n - 1, n_samples - 1):
            y_sum = np.sum(y_narma[k - (n - 1):k + 1])
            y_narma[k + 1] = (a * y_narma[k] + 
                            b * y_narma[k] * y_sum + 
                            c * u_normalized[k - n + 1] * u_normalized[k] + d)
    
    return y_narma


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


#=============================================================================
# Optimization objective
#=============================================================================

def objective(trial):
    """
    Optuna objective function to minimize test NMSE.
    
    Args:
        trial: Optuna trial object
    
    Returns:
        mean_test_nmse: Average test NMSE across all configurations
    """
    # Sample hyperparameters
    washout_fraction = trial.suggest_float('washout_fraction', 
                                          WASHOUT_FRACTION_RANGE[0], 
                                          WASHOUT_FRACTION_RANGE[1])
    cutoff_fraction = trial.suggest_float('cutoff_fraction', 
                                         CUTOFF_FRACTION_RANGE[0], 
                                         CUTOFF_FRACTION_RANGE[1])
    ridge_alpha = trial.suggest_float('ridge_alpha', 
                                     RIDGE_ALPHA_RANGE[0], 
                                     RIDGE_ALPHA_RANGE[1], 
                                     log=True)
    
    # Store all test NMSE values
    all_test_nmse = []
    
    # Evaluate across all data files and NARMA orders
    for period_ratio, data_file in DATA_FILES.items():
        if not os.path.exists(data_file):
            continue
        
        # Load data once per file
        try:
            df = load_robot_data(data_file)
            robot_ids = extract_robot_ids(df)
            
            if len(robot_ids) < N_ROBOTS:
                continue
            
            robot_ids = robot_ids[:N_ROBOTS]
            
            # Extract features
            features = extract_robot_features(df, robot_ids, FEATURE_NAMES, USE_LOCAL_FRAME, 0)
            n_samples = features.shape[0]
            
            # Extract global_u
            states = get_robot_states(df, robot_ids[0], verbose=False)
            global_u = states['global_u']
            
            if global_u is None:
                continue
            
            # Normalize global_u to [0, 0.5]
            u_normalized = global_u - global_u.min()
            u_normalized = 0.5 * u_normalized / u_normalized.max()
            
            # Evaluate for each NARMA order
            for narma_order in NARMA_ORDERS:
                # Generate NARMA target
                y_narma = generate_narma_from_input(u_normalized, narma_order)
                
                # Apply washout and cutoff
                washout_samples = int(washout_fraction * n_samples)
                cutoff_samples = int(cutoff_fraction * n_samples)
                
                end_idx = n_samples - cutoff_samples if cutoff_samples > 0 else n_samples
                valid_idx = np.arange(washout_samples, end_idx)
                n_valid = len(valid_idx)
                
                # Check if we have enough data
                if n_valid < 100:  # Minimum samples needed
                    continue
                
                n_train = int(TRAIN_FRACTION * n_valid)
                
                train_idx = valid_idx[:n_train]
                test_idx = valid_idx[n_train:]
                
                # Extract train/test data
                X_train = features[train_idx, :]
                y_train = y_narma[train_idx]
                X_test = features[test_idx, :]
                y_test = y_narma[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train Ridge regression
                regressor = Ridge(alpha=ridge_alpha, solver=RIDGE_SOLVER, fit_intercept=True)
                regressor.fit(X_train_scaled, y_train)
                
                # Evaluate on test set
                test_pred = regressor.predict(X_test_scaled)
                test_mse = np.mean((test_pred - y_test)**2)
                test_var = np.var(y_test)
                test_nmse = test_mse / (test_var + 1e-9)
                
                all_test_nmse.append(test_nmse)
        
        except Exception as e:
            print(f"Error processing {data_file}: {e}")
            continue
    
    # Return mean test NMSE (or inf if no valid results)
    if len(all_test_nmse) == 0:
        return float('inf')
    
    mean_test_nmse = np.mean(all_test_nmse)
    return mean_test_nmse


#=============================================================================
# Main execution
#=============================================================================

def main():
    """Run Optuna optimization."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("NARMA HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data files: {list(DATA_FILES.keys())}")
    print(f"  NARMA orders: {NARMA_ORDERS}")
    print(f"  Number of trials: {N_TRIALS}")
    print(f"  Features: {FEATURE_NAMES}")
    print(f"\nSearch ranges:")
    print(f"  Washout fraction: {WASHOUT_FRACTION_RANGE}")
    print(f"  Cutoff fraction: {CUTOFF_FRACTION_RANGE}")
    print(f"  Ridge alpha: {RIDGE_ALPHA_RANGE} (log scale)")
    print("="*80 + "\n")
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction='minimize',
        storage=STORAGE,
        load_if_exists=True
    )
    
    # Run optimization
    print("Starting optimization...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"\nBest trial:")
    print(f"  Trial number: {study.best_trial.number}")
    print(f"  Test NMSE: {study.best_trial.value:.6f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        if key == 'ridge_alpha':
            print(f"  {key}: {value:.3e}")
        else:
            print(f"  {key}: {value:.4f}")
    print("="*80)
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'optimization_results.csv')
    df_trials = study.trials_dataframe()
    df_trials.to_csv(results_path, index=False)
    print(f"\n✓ All trials saved to: {results_path}")
    
    # Save best parameters
    best_params_path = os.path.join(OUTPUT_DIR, 'best_hyperparameters.txt')
    with open(best_params_path, 'w') as f:
        f.write(f"Best Trial Number: {study.best_trial.number}\n")
        f.write(f"Best Test NMSE: {study.best_trial.value:.6f}\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in study.best_params.items():
            if key == 'ridge_alpha':
                f.write(f"  {key}: {value:.6e}\n")
            else:
                f.write(f"  {key}: {value:.6f}\n")
    print(f"✓ Best hyperparameters saved to: {best_params_path}")
    
    # Print top 5 trials
    print(f"\nTop 5 trials:")
    print("-" * 80)
    df_sorted = df_trials.sort_values('value').head(5)
    for idx, row in df_sorted.iterrows():
        print(f"  Trial {int(row['number'])}: NMSE={row['value']:.6f}, "
              f"washout={row['params_washout_fraction']:.4f}, "
              f"cutoff={row['params_cutoff_fraction']:.4f}, "
              f"alpha={row['params_ridge_alpha']:.3e}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
