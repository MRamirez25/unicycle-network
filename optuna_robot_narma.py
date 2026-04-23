#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for robot NARMA classifier.

This script uses Optuna to find optimal hyperparameters for the NARMA classification
task using robot state data, including:
- Ridge regularization (alpha)
- Train/washout fractions
- Feature selection
- Data trimming
- Smoothing parameters
- Local frame usage
"""

import os
import sys
import numpy as np
import optuna
from optuna.samplers import TPESampler
import traceback
import pickle
from datetime import datetime

# Add parent directory to path
SCRIPT_DIR = '/home/mariano/phd_code/unicycle-network/'
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from robot_narma_classifier import train_narma_classifier
from plot_robot_data import load_robot_data


def objective(trial, df, narma_order=2, data_file_name="robot_data"):
    """
    Optuna objective function for robot NARMA hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        df: DataFrame with robot data
        narma_order: Order of NARMA system
        data_file_name: Name for saving results
    
    Returns:
        test_nmse: Test NMSE (lower is better)
    """
    try:
        # Hyperparameters to optimize
        
        # 1. Ridge regularization (log scale)
        ridge_alpha = trial.suggest_float('ridge_alpha', 1e-8, 1e4, log=True)
        
        # 2. Data split fractions
        washout_fraction = trial.suggest_float('washout_fraction', 0.05, 0.15)
        train_fraction = 0.7#trial.suggest_float('train_fraction', 0.5, 0.8)
        
        # 3. Feature selection
        # Start with a base set and optionally add more
        use_pos_x = True# trial.suggest_categorical('use_pos_x', [True, False])
        use_pos_y = True#trial.suggest_categorical('use_pos_y', [True, False])
        use_linear_x = True#trial.suggest_categorical('use_linear_x', [True, False])
        use_qz = False
        use_qw = False
        use_omega = False
        
        # Build feature list
        feature_names = []
        if use_pos_x:
            feature_names.append('pos_x')
        if use_pos_y:
            feature_names.append('pos_y')
        if use_linear_x:
            feature_names.append('linear_x')
        if use_qz:
            feature_names.append('qz')
        if use_qw:
            feature_names.append('qw')
        if use_omega:
            feature_names.append('omega')
        
        # Ensure at least one feature is selected
        if len(feature_names) == 0:
            # Prune this trial - no features selected
            raise optuna.TrialPruned("No features selected")
        
        # 4. Data trimming (to avoid edge effects or bad data)
        trim_start = trial.suggest_int('trim_start', 0, 500, step=100)
        trim_end = trial.suggest_int('trim_end', 0, 500, step=100)
        start_idx = trim_start if trim_start > 0 else None
        end_idx = -trim_end if trim_end > 0 else None
        
        # 5. Local frame usage
        use_local_frame = False#trial.suggest_categorical('use_local_frame', [True, False])
        
        # 6. Feature smoothing
        smooth_features_flag = False#trial.suggest_categorical('smooth_features', [True, False])
        if smooth_features_flag:
            smooth_window = trial.suggest_int('smooth_window', 1, 101, step=5)  # Odd numbers
            smooth_method = trial.suggest_categorical('smooth_method', ['savgol', 'gaussian', 'moving_average'])
        else:
            smooth_window = 5
            smooth_method = 'savgol'
        
        # Train the classifier with these hyperparameters
        results = train_narma_classifier(
            df,
            narma_order=narma_order,
            period_ratio=70,  # Fixed for robot data
            amplitude=0.5,    # Fixed for robot data
            washout_fraction=washout_fraction,
            train_fraction=train_fraction,
            ridge_alpha=ridge_alpha,
            feature_names=feature_names,
            start_idx=start_idx,
            end_idx=end_idx,
            use_local_frame=use_local_frame,
            exclude_robots=None,  # Could make this a hyperparameter too
            smooth_features_flag=smooth_features_flag,
            smooth_window=smooth_window,
            smooth_method=smooth_method,
            verbose=False  # Suppress output during optimization
        )
        
        # Return test NMSE (minimize this)
        test_nmse = results['test_nmse']
        
        # Log additional metrics for tracking
        trial.set_user_attr('train_nmse', results['train_nmse'])
        trial.set_user_attr('test_mse', results['test_mse'])
        trial.set_user_attr('n_features', len(feature_names))
        trial.set_user_attr('feature_list', ','.join(feature_names))
        
        return test_nmse
    
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"\nTrial failed with error: {e}")
        traceback.print_exc()
        # Return a large penalty value instead of crashing
        return 1e10


def run_optimization(data_file, narma_order=2, n_trials=100, study_name=None, 
                     db_path=None, exclude_robots=None):
    """
    Run Optuna hyperparameter optimization for robot NARMA classifier.
    
    Args:
        data_file: Path to robot data CSV file
        narma_order: Order of NARMA system
        n_trials: Number of optimization trials
        study_name: Name for the optuna study
        db_path: Path to SQLite database for storing results
        exclude_robots: List of robot IDs to exclude (optional)
    
    Returns:
        study: Optuna study object with results
    """
    # Load robot data
    print(f"Loading robot data from: {data_file}")
    df = load_robot_data(data_file)
    print(f"Data loaded: {len(df)} timesteps\n")
    
    # Generate study name if not provided
    if study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_name = os.path.basename(data_file).replace('.csv.gz', '').replace('.csv', '')
        study_name = f"robot_narma{narma_order}_{data_name}_{timestamp}"
    
    # Database path for persistence
    if db_path is None:
        db_path = os.path.join(SCRIPT_DIR, 'optuna_databases', f'{study_name}.db')
    
    # Ensure database directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    storage = f"sqlite:///{db_path}"
    
    print(f"{'='*70}")
    print(f"Starting Optuna Optimization")
    print(f"{'='*70}")
    print(f"Study name: {study_name}")
    print(f"Database: {db_path}")
    print(f"NARMA order: {narma_order}")
    print(f"Number of trials: {n_trials}")
    print(f"{'='*70}\n")
    
    # Create or load study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction='minimize',  # Minimize test NMSE
        load_if_exists=True
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, df, narma_order=narma_order, 
                               data_file_name=os.path.basename(data_file)),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1  # Single job for now (could parallelize if database supports it)
    )
    
    # Print results
    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best test NMSE: {study.best_value:.6f}")
    
    # Best parameters
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Best features
    if 'feature_list' in study.best_trial.user_attrs:
        print(f"\nBest features: {study.best_trial.user_attrs['feature_list']}")
        print(f"Number of features: {study.best_trial.user_attrs['n_features']}")
    
    # Additional metrics
    if 'train_nmse' in study.best_trial.user_attrs:
        print(f"\nBest model performance:")
        print(f"  Train NMSE: {study.best_trial.user_attrs['train_nmse']:.6f}")
        print(f"  Test NMSE: {study.best_value:.6f}")
        print(f"  Test MSE: {study.best_trial.user_attrs['test_mse']:.6f}")
    
    print(f"{'='*70}\n")
    
    # Save study object
    study_file = os.path.join(SCRIPT_DIR, 'optuna_databases', f'{study_name}_study.pkl')
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)
    print(f"Study object saved to: {study_file}")
    
    return study


def analyze_study(study_name_or_path):
    """
    Analyze and visualize results from a completed study.
    
    Args:
        study_name_or_path: Either study name or path to database
    """
    import matplotlib.pyplot as plt
    from optuna.visualization import plot_optimization_history, plot_param_importances
    
    # Load study
    if study_name_or_path.endswith('.db'):
        storage = f"sqlite:///{study_name_or_path}"
        # Extract study name from path
        study_name = os.path.basename(study_name_or_path).replace('.db', '')
    else:
        study_name = study_name_or_path
        db_path = os.path.join(SCRIPT_DIR, 'optuna_databases', f'{study_name}.db')
        storage = f"sqlite:///{db_path}"
    
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    print(f"Analyzing study: {study_name}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best NMSE: {study.best_value:.6f}")
    
    # Plot optimization history
    fig = plot_optimization_history(study)
    fig.write_image(os.path.join(SCRIPT_DIR, f'{study_name}_history.png'))
    print(f"Saved optimization history plot")
    
    # Plot parameter importances
    fig = plot_param_importances(study)
    fig.write_image(os.path.join(SCRIPT_DIR, f'{study_name}_importances.png'))
    print(f"Saved parameter importance plot")
    
    return study


# Main execution
if __name__ == "__main__":
    # Configuration
    DATA_FILE = "/home/mariano/phd_code/unicycle-network/reference10.csv.gz"
    NARMA_ORDER = 2
    N_TRIALS = 100  # Start with 50 trials, increase if needed
    
    # Run optimization
    study = run_optimization(
        data_file=DATA_FILE,
        narma_order=NARMA_ORDER,
        n_trials=N_TRIALS,
        study_name=None,  # Auto-generate
        db_path=None,     # Auto-generate
        exclude_robots=None
    )
    
    # Optionally analyze results
    # analyze_study(study.study_name)
    
    print("\n✓ Optimization complete!")
    print(f"Use the best hyperparameters in robot_narma_classifier.py")
    print(f"Database saved for future analysis")
