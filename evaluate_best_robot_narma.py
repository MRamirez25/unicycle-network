#!/usr/bin/env python3
"""
Evaluate and visualize the best hyperparameters from Optuna study.

This script loads an Optuna study, extracts the best hyperparameters,
trains the model with those parameters, and generates all visualization plots.
"""

import os
import sys
import numpy as np
import optuna
import pickle
from datetime import datetime

# Add parent directory to path
SCRIPT_DIR = '/home/mariano/phd_code/unicycle-network/'
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from robot_narma_classifier import (
    train_narma_classifier, 
    plot_results, 
    plot_features, 
    plot_feature_scaling,
    plot_all_robots_timeseries
)
from plot_robot_data import load_robot_data


def load_study(study_name_or_db_path):
    """
    Load an Optuna study from database.
    
    Args:
        study_name_or_db_path: Either study name or path to .db file
    
    Returns:
        study: Loaded Optuna study object
    """
    if study_name_or_db_path.endswith('.db'):
        # Direct path to database
        db_path = study_name_or_db_path
        storage = f"sqlite:///{db_path}"
        # Extract study name from database
        study_name = os.path.basename(db_path).replace('.db', '')
    else:
        # Study name provided
        study_name = study_name_or_db_path
        db_path = os.path.join(SCRIPT_DIR, 'optuna_databases', f'{study_name}.db')
        storage = f"sqlite:///{db_path}"
    
    print(f"Loading study from: {db_path}")
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    return study


def extract_best_params(study):
    """
    Extract best hyperparameters from study and format them for training.
    
    Args:
        study: Optuna study object
    
    Returns:
        params: Dictionary of parameters ready for train_narma_classifier
    """
    best_params = study.best_params
    best_trial = study.best_trial
    
    print(f"\n{'='*70}")
    print("BEST TRIAL INFORMATION")
    print(f"{'='*70}")
    print(f"Trial number: {best_trial.number}")
    print(f"Test NMSE: {study.best_value:.6f}")
    
    if 'train_nmse' in best_trial.user_attrs:
        print(f"Train NMSE: {best_trial.user_attrs['train_nmse']:.6f}")
    if 'test_mse' in best_trial.user_attrs:
        print(f"Test MSE: {best_trial.user_attrs['test_mse']:.6f}")
    
    print(f"\n{'='*70}")
    print("BEST HYPERPARAMETERS")
    print(f"{'='*70}")
    
    # Extract parameters
    params = {}
    
    # Ridge alpha
    params['ridge_alpha'] = best_params.get('ridge_alpha', 1e-6)
    print(f"Ridge alpha: {params['ridge_alpha']:.2e}")
    
    # Data splits
    params['washout_fraction'] = best_params.get('washout_fraction', 0.1)
    params['train_fraction'] = best_params.get('train_fraction', 0.7)
    print(f"Washout fraction: {params['washout_fraction']:.3f}")
    print(f"Train fraction: {params['train_fraction']:.3f}")
    
    # Feature selection
    feature_names = []
    if True:
        feature_names.append('pos_x')
    if True:
        feature_names.append('pos_y')
    if True:
        feature_names.append('linear_x')
    if best_params.get('use_qz', False):
        feature_names.append('qz')
    if best_params.get('use_qw', False):
        feature_names.append('qw')
    if best_params.get('use_omega', False):
        feature_names.append('omega')
    
    params['feature_names'] = feature_names if len(feature_names) > 0 else None
    print(f"Features: {feature_names if len(feature_names) > 0 else 'ALL'}")
    
    # Data trimming
    trim_start = best_params.get('trim_start', 0)
    trim_end = best_params.get('trim_end', 0)
    params['start_idx'] = trim_start if trim_start > 0 else None
    params['end_idx'] = -trim_end if trim_end > 0 else None
    print(f"Trim start: {trim_start}")
    print(f"Trim end: {trim_end}")
    
    # Local frame
    params['use_local_frame'] = False#best_params.get('use_local_frame', False)
    print(f"Use local frame: {params['use_local_frame']}")
    
    # Smoothing
    params['smooth_features_flag'] = False#best_params.get('smooth_features', False)
    params['smooth_window'] = best_params.get('smooth_window', 5)
    params['smooth_method'] = best_params.get('smooth_method', 'savgol')
    print(f"Smoothing enabled: {params['smooth_features_flag']}")
    if params['smooth_features_flag']:
        print(f"  Window: {params['smooth_window']}")
        print(f"  Method: {params['smooth_method']}")
    
    print(f"{'='*70}\n")
    
    return params


def evaluate_best_model(study_name_or_db_path, data_file, narma_order=2, 
                       exclude_robots=None, save_dir=None):
    """
    Load best hyperparameters from study, train model, and generate plots.
    
    Args:
        study_name_or_db_path: Study name or path to database
        data_file: Path to robot data CSV file
        narma_order: Order of NARMA system
        exclude_robots: List of robot IDs to exclude
        save_dir: Directory to save plots (default: SCRIPT_DIR)
    
    Returns:
        results: Dictionary with training results
        params: Best hyperparameters used
    """
    if save_dir is None:
        save_dir = SCRIPT_DIR
    
    # Load study
    study = load_study(study_name_or_db_path)
    
    # Extract best parameters
    params = extract_best_params(study)
    
    # Load data
    print("Loading robot data...")
    df = load_robot_data(data_file)
    print(f"Data loaded: {len(df)} timesteps\n")
    
    # Train model with best hyperparameters
    print("Training model with best hyperparameters...")
    print(f"{'='*70}\n")
    
    results = train_narma_classifier(
        df,
        narma_order=narma_order,
        period_ratio=70,  # Fixed for robot data
        amplitude=0.5,    # Fixed for robot data
        washout_fraction=params['washout_fraction'],
        train_fraction=params['train_fraction'],
        ridge_alpha=params['ridge_alpha'],
        feature_names=params['feature_names'],
        start_idx=params['start_idx'],
        end_idx=params['end_idx'],
        use_local_frame=params['use_local_frame'],
        exclude_robots=exclude_robots,
        smooth_features_flag=params['smooth_features_flag'],
        smooth_window=params['smooth_window'],
        smooth_method=params['smooth_method'],
        verbose=True
    )
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*70 + "\n")
    
    # Create output directory for this evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = study.study_name
    output_dir = os.path.join(save_dir, f'best_model_{study_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving plots to: {output_dir}\n")
    
    # Plot 1: Robot timeseries
    print("1. Generating robot timeseries plot...")
    plot_all_robots_timeseries(df)
    
    # Plot 2: Predictions
    print("2. Generating predictions plot...")
    plot_results(results, narma_order=narma_order, save_dir=output_dir)
    
    # Plot 3: Features
    print("3. Generating features plot...")
    plot_features(results, narma_order=narma_order, save_dir=output_dir, max_samples=5000)
    
    # Plot 4: Feature scaling
    print("4. Generating feature scaling comparison plot...")
    plot_feature_scaling(results, narma_order=narma_order, save_dir=output_dir, max_samples=2000)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nAll plots saved to: {output_dir}")
    print(f"\nFinal Performance:")
    print(f"  Train NMSE: {results['train_nmse']:.6f}")
    print(f"  Test NMSE: {results['test_nmse']:.6f}")
    print(f"  Test MSE: {results['test_mse']:.6f}")
    print("="*70 + "\n")
    
    # Save hyperparameters to text file
    params_file = os.path.join(output_dir, 'best_hyperparameters.txt')
    with open(params_file, 'w') as f:
        f.write("BEST HYPERPARAMETERS FROM OPTUNA STUDY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Study name: {study.study_name}\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Test NMSE: {study.best_value:.6f}\n\n")
        f.write("Parameters:\n")
        for key, value in params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nPerformance:\n")
        f.write(f"  Train NMSE: {results['train_nmse']:.6f}\n")
        f.write(f"  Test NMSE: {results['test_nmse']:.6f}\n")
        f.write(f"  Test MSE: {results['test_mse']:.6f}\n")
    
    print(f"Hyperparameters saved to: {params_file}\n")
    
    return results, params


def print_summary_table(study):
    """
    Print a summary table of top trials.
    
    Args:
        study: Optuna study object
    """
    print("\n" + "="*70)
    print("TOP 10 TRIALS")
    print("="*70)
    print(f"{'Trial':<8} {'NMSE':<12} {'Features':<30} {'Ridge α':<12}")
    print("-"*70)
    
    # Sort trials by value
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
    
    for trial in sorted_trials[:10]:
        if trial.value is None:
            continue
        
        trial_num = trial.number
        nmse = trial.value
        
        # Get features
        features = trial.user_attrs.get('feature_list', 'N/A')
        if len(features) > 27:
            features = features[:27] + '...'
        
        # Get ridge alpha
        ridge_alpha = trial.params.get('ridge_alpha', 0.0)
        
        print(f"{trial_num:<8} {nmse:<12.6f} {features:<30} {ridge_alpha:<12.2e}")
    
    print("="*70 + "\n")


# Main execution
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate best model from Optuna study')
    parser.add_argument('--study', type=str, default=None,
                       help='Study name or path to .db file')
    parser.add_argument('--data', type=str, 
                       default='/home/mariano/phd_code/unicycle-network/reference10.csv.gz',
                       help='Path to robot data CSV file')
    parser.add_argument('--narma-order', type=int, default=2,
                       help='NARMA order')
    parser.add_argument('--list-studies', action='store_true',
                       help='List available studies and exit')
    
    args = parser.parse_args()
    
    # List studies if requested
    if args.list_studies:
        db_dir = os.path.join(SCRIPT_DIR, 'optuna_databases')
        if os.path.exists(db_dir):
            db_files = [f for f in os.listdir(db_dir) if f.endswith('.db')]
            print("\nAvailable studies:")
            for db_file in sorted(db_files):
                print(f"  - {db_file}")
            print()
        else:
            print("No optuna_databases directory found")
        sys.exit(0)
    
    # Auto-detect most recent study if not provided
    if args.study is None:
        db_dir = os.path.join(SCRIPT_DIR, 'optuna_databases')
        if os.path.exists(db_dir):
            db_files = [os.path.join(db_dir, f) for f in os.listdir(db_dir) if f.endswith('.db')]
            if db_files:
                # Get most recent database
                args.study = max(db_files, key=os.path.getmtime)
                print(f"Auto-detected most recent study: {os.path.basename(args.study)}\n")
            else:
                print("ERROR: No .db files found in optuna_databases/")
                print("Run optuna_robot_narma.py first to create a study.")
                sys.exit(1)
        else:
            print("ERROR: No optuna_databases directory found")
            print("Run optuna_robot_narma.py first to create a study.")
            sys.exit(1)
    
    # Load and evaluate
    try:
        # Load study first to show summary
        study = load_study(args.study)
        print(f"\nStudy loaded successfully!")
        print(f"Total trials: {len(study.trials)}")
        print(f"Best NMSE: {study.best_value:.6f}")
        
        # Print summary table
        print_summary_table(study)
        
        # Evaluate best model
        results, params = evaluate_best_model(
            study_name_or_db_path=args.study,
            data_file=args.data,
            narma_order=args.narma_order,
            exclude_robots=None,
            save_dir=SCRIPT_DIR
        )
        
        print("\n✓ Evaluation complete!")
        print("Check the output directory for all plots and results.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
