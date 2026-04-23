#!/usr/bin/env python3
"""
Evaluate NARMA performance across multiple period ratios and orders.

This script:
1. Loads robot data files recorded at different period ratios
2. Evaluates multiple NARMA orders (e.g., 2, 3, 5, 10)
3. Trains Ridge regression for each (period_ratio, narma_order) combination
4. Plots test NMSE vs period ratio with one line per NARMA order
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
    'font.size': 24,               # Base font size
    'axes.titlesize': 24,          # Title size
    'axes.labelsize': 20,          # X/Y axis labels
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

# Data files: each file corresponds to a specific period ratio
# Format: {period_ratio: "path/to/file.csv.gz"}
DATA_FILES = {
    40: "time-period-40.csv.gz",
    50: "time-period-50.csv.gz",
    60: "time-period-60.csv.gz",
    70: "time-period-70.csv.gz",
    80: "time-period-80.csv.gz",
    90: "time-period-90.csv.gz",
    100: "time-period-100.csv.gz",
    110: "time-period-110.csv.gz",
    120: "time-period-120.csv.gz"
}

# NARMA orders to evaluate
NARMA_ORDERS = [2, 3, 5]

# Ridge regression parameters
RIDGE_ALPHA = 300  # Regularization strength
RIDGE_SOLVER = 'auto'  # Solver: 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'

# Data processing parameters
WASHOUT_FRACTION = 0.15  # Fraction of data to discard as washout (from beginning)
CUTOFF_FRACTION = 0  # Fraction of data to discard from the end (0.0 = no cutoff)
TRAIN_FRACTION = 0.8  # Fraction of remaining data for training (rest for testing)
N_ROBOTS = 10  # Number of robots to use

# Plot configuration for test period trajectories
# Set to None to plot all robots, or provide a list of robot indices (0-based) to plot specific robots
# Example: [0, 1, 2] will plot only the first 3 robots, [0, 4, 9] will plot robots 1, 5, and 10
ROBOTS_TO_PLOT_TEST = [3, 9]  # None = plot all robots, or list like [0, 1, 2, 3, 4]

# Feature configuration
FEATURE_NAMES = ['pos_x', 'pos_y']  # Features to use: ['pos_x', 'pos_y', 'linear_x', 'qz', 'qw', 'omega']
USE_LOCAL_FRAME = False  # Shift positions to local frame centered at post-washout position

# Smoothing parameters (set window_size=None to disable smoothing)
SMOOTH_WINDOW_SIZE = None  # Window size for smoothing (e.g., 5, 7, 11, or None)
SMOOTH_METHOD = None  # 'moving_average', 'gaussian', or 'savgol'

# NARMA generation parameters
NARMA_AMPLITUDE = 0.5  # Amplitude for synthetic NARMA input
DT = 0.05  # Time step

# Output
OUTPUT_DIR = "narma_period_ratio_results"
SAVE_PLOT = True
SHOW_PLOT = True

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
    """
    Extract feature matrix from robot states.
    
    Returns:
        features: numpy array of shape (n_timesteps, n_features)
    """
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

def evaluate_narma_for_period_ratio(data_file, period_ratio, narma_order):
    """
    Evaluate NARMA prediction for a given data file, period ratio, and NARMA order.
    
    Returns:
        results: dict with keys 'nmse', 'mse', 'train_nmse', 'train_mse', 'success'
    """
    try:
        print(f"\n{'='*70}")
        print(f"Period Ratio: {period_ratio}, NARMA Order: {narma_order}")
        print(f"Data file: {data_file}")
        print(f"{'='*70}")
        
        # Load robot data
        df = load_robot_data(data_file)
        robot_ids = extract_robot_ids(df)
        
        if len(robot_ids) < N_ROBOTS:
            print(f"WARNING: Only {len(robot_ids)} robots found, need {N_ROBOTS}")
            return {'nmse': np.nan, 'mse': np.nan, 'train_nmse': np.nan, 'train_mse': np.nan, 'weights': None, 'success': False}
        
        robot_ids = robot_ids[:N_ROBOTS]
        
        # Extract features
        features = extract_robot_features(df, robot_ids, FEATURE_NAMES, USE_LOCAL_FRAME, 0)
        n_samples = features.shape[0]
        
        print(f"Features shape: {features.shape}")
        print(f"Feature names: {FEATURE_NAMES}")
        
        # Extract global_u from robot data
        states = get_robot_states(df, robot_ids[0], verbose=False)
        global_u = states['global_u']
        
        if global_u is None:
            print("ERROR: global_u not found in data file")
            return {'nmse': np.nan, 'mse': np.nan, 'train_nmse': np.nan, 'train_mse': np.nan, 'weights': None, 'success': False}
        
        print(f"Extracted global_u from data - shape: {global_u.shape}, range: [{global_u.min():.4f}, {global_u.max():.4f}]")
        
        # Generate NARMA target using the actual global_u from data
        # Note: We pass global_u directly by using n_samples matching the data length
        # u_input, u_normalized, y_narma = generate_narma(
        #     n_samples, 
        #     narma_order=narma_order, 
        #     period_ratio=period_ratio,  # This parameter is ignored when we use actual data
        #     amplitude=NARMA_AMPLITUDE,  # This parameter is ignored when we use actual data
        #     dt=DT
        # )
        
        # Replace the synthetic input with actual global_u
        # Normalize global_u to [0, 0.5] for NARMA generation
        u_normalized = global_u - global_u.min()
        u_normalized = 0.5 * u_normalized / u_normalized.max()
        
        # Regenerate NARMA with actual input
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
        
        print(f"NARMA-{narma_order} generated from global_u - range: [{y_narma.min():.4f}, {y_narma.max():.4f}]")
        
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
        
        print(f"Data split: washout={washout_samples}, valid={n_valid}, cutoff={cutoff_samples}, train={len(train_idx)}, test={len(test_idx)}")
        
        # Extract train/test features and targets
        X_train = features[train_idx, :]
        y_train = y_narma[train_idx]
        X_test = features[test_idx, :]
        y_test = y_narma[test_idx]
        
        # Apply smoothing if enabled
        if SMOOTH_WINDOW_SIZE is not None:
            print(f"Applying {SMOOTH_METHOD} smoothing with window size {SMOOTH_WINDOW_SIZE}")
            X_train = smooth_features(X_train, window_size=SMOOTH_WINDOW_SIZE, method=SMOOTH_METHOD)
            X_test = smooth_features(X_test, window_size=SMOOTH_WINDOW_SIZE, method=SMOOTH_METHOD)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Scaled features - mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
        
        # Train Ridge regression
        print(f"Training Ridge regression (alpha={RIDGE_ALPHA}, solver={RIDGE_SOLVER})...")
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
        
        print(f"\nResults:")
        print(f"  Train NMSE: {train_nmse:.6f}, MSE: {train_mse:.6f}")
        print(f"  Test NMSE:  {test_nmse:.6f}, MSE: {test_mse:.6f}")
        
        # Save train/test prediction plots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot training predictions
        ax = axes[0]
        t_train = train_idx * DT
        ax.plot(t_train, y_train, 'b-', linewidth=1.5, label='Target', alpha=0.8)
        ax.plot(t_train, train_pred, 'r--', linewidth=1.5, label='Prediction', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('NARMA Output')
        ax.set_title(f'Training Set - NARMA-{narma_order}, Period Ratio {period_ratio} (NMSE: {train_nmse:.4f})', 
                 fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot test predictions
        ax = axes[1]
        t_test = test_idx * DT
        ax.plot(t_test, y_test, 'b-', linewidth=1.5, label='Target', alpha=0.8)
        ax.plot(t_test, test_pred, 'r--', linewidth=1.5, label='Prediction', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('NARMA Output')
        ax.set_title(f'Test Set - NARMA-{narma_order}, Period Ratio {period_ratio} (NMSE: {test_nmse:.4f})', 
                    fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = os.path.join(OUTPUT_DIR, 'prediction_plots')
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'narma{narma_order}_period{period_ratio}.png')
        if narma_order == 3 and period_ratio == 80:
            plt.savefig(plot_path.split('.')[0] + '.pdf', dpi=150, bbox_inches='tight')
        else:
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved plot: {plot_path}")
        
        # Save input signal plot (only once per period ratio, not per NARMA order)
        # Create a unique identifier to ensure we only plot once per period ratio
        input_plot_path = os.path.join(plot_dir, f'input_signal_period{period_ratio}.png')
        if not os.path.exists(input_plot_path):
            fig_input, ax_input = plt.subplots(1, 1, figsize=(14, 5))
            
            t_all = np.arange(n_samples) * DT
            ax_input.plot(t_all, global_u, 'g-', linewidth=1.5, label='Global Input (u)', alpha=0.8)
            ax_input.axvline(washout_samples * DT, color='orange', linestyle='--', linewidth=2, 
                           label='Washout End', alpha=0.7)
            ax_input.axvline(train_idx[-1] * DT, color='red', linestyle='--', linewidth=2, 
                           label='Train/Test Split', alpha=0.7)
            
            ax_input.set_xlabel('Time (s)')
            ax_input.set_ylabel('Input Signal (u)')
            ax_input.set_title(f'Input Signal - Period Ratio {period_ratio}\nRange: [{global_u.min():.4f}, {global_u.max():.4f}]', 
                             fontweight='bold')
            ax_input.legend()
            ax_input.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(input_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig_input)
            print(f"  Saved input plot: {input_plot_path}")
        
        # Save 2D robot trajectories plot (only once per period ratio)
        trajectory_plot_path = os.path.join(plot_dir, f'trajectories_2d_period{period_ratio}.png')
        if not os.path.exists(trajectory_plot_path):
            fig_traj, ax_traj = plt.subplots(1, 1, figsize=(10, 10))
            
            # Plot trajectory for each robot
            colors = plt.cm.tab10(np.linspace(0, 1, len(robot_ids)))
            
            for idx, robot_id in enumerate(robot_ids):
                robot_states = get_robot_states(df, robot_id, verbose=False)
                pos_x = robot_states['pos_x']
                pos_y = robot_states['pos_y']
                
                # Plot trajectory
                ax_traj.plot(pos_x, pos_y, '-', color=colors[idx], linewidth=1.5, 
                           label=f'Robot {idx+1}', alpha=0.7)
                
                # Mark start position
                ax_traj.plot(pos_x[0], pos_y[0], 'o', color=colors[idx], 
                           markersize=10, markeredgecolor='black', markeredgewidth=1.5)
                
                # Mark end position
                ax_traj.plot(pos_x[-1], pos_y[-1], 's', color=colors[idx], 
                           markersize=10, markeredgecolor='black', markeredgewidth=1.5)
            
            ax_traj.set_xlabel('X Position (m)')
            ax_traj.set_ylabel('Y Position (m)')
            ax_traj.set_title(f'Robot Trajectories - Period Ratio {period_ratio}\n(Circle=Start, Square=End)', 
                            fontweight='bold')
            ax_traj.legend(ncol=2)
            ax_traj.grid(True, alpha=0.3)
            ax_traj.set_aspect('equal', adjustable='datalim')
            
            plt.tight_layout()
            plt.savefig(trajectory_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig_traj)
            print(f"  Saved trajectory plot: {trajectory_plot_path}")
        
        # Save time-based trajectories plot (positions and velocities vs time)
        time_traj_plot_path = os.path.join(plot_dir, f'trajectories_time_period{period_ratio}.png')
        if not os.path.exists(time_traj_plot_path):
            fig_time, axes_time = plt.subplots(3, 1, figsize=(14, 12))
            
            # Plot trajectory for each robot
            colors = plt.cm.tab10(np.linspace(0, 1, len(robot_ids)))
            t_all = np.arange(n_samples) * DT
            
            for idx, robot_id in enumerate(robot_ids):
                robot_states = get_robot_states(df, robot_id, verbose=False)
                pos_x = robot_states['pos_x']
                pos_y = robot_states['pos_y']
                linear_x = robot_states['linear_x']
                
                # Plot X position vs time
                axes_time[0].plot(t_all, pos_x, '-', color=colors[idx], linewidth=1.2, 
                                label=f'Robot {idx+1}', alpha=0.7)
                
                # Plot Y position vs time
                axes_time[1].plot(t_all, pos_y, '-', color=colors[idx], linewidth=1.2, 
                                label=f'Robot {idx+1}', alpha=0.7)
                
                # Plot linear velocity vs time
                axes_time[2].plot(t_all, linear_x, '-', color=colors[idx], linewidth=1.2, 
                                label=f'Robot {idx+1}', alpha=0.7)
            
            # Add washout and train/test split markers to all subplots
            for ax in axes_time:
                ax.axvline(washout_samples * DT, color='orange', linestyle='--', 
                          linewidth=2, alpha=0.5)
                ax.axvline(train_idx[-1] * DT, color='red', linestyle='--', 
                          linewidth=2, alpha=0.5)
                ax.grid(True, alpha=0.3)
            
            # Configure subplots
            axes_time[0].set_ylabel('X Position (m)')
            axes_time[0].set_title(f'Robot Trajectories vs Time - Period Ratio {period_ratio}', 
                                   fontweight='bold')
            axes_time[0].legend( ncol=5, loc='upper right')
            
            axes_time[1].set_ylabel('Y Position (m)')
            axes_time[1].legend( ncol=5, loc='upper right')
            
            axes_time[2].set_xlabel('Time (s)')
            axes_time[2].set_ylabel('Linear Velocity (m/s)')
            axes_time[2].legend( ncol=5, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(time_traj_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig_time)
            print(f"  Saved time trajectory plot: {time_traj_plot_path}")
        
        # Save time-based trajectories plot zoomed to test portion
        # Always regenerate to respect ROBOTS_TO_PLOT_TEST changes
        time_traj_test_plot_path = os.path.join(plot_dir, f'trajectories_time_test_period{period_ratio}.png')
        
        fig_test, axes_test = plt.subplots(3, 1, figsize=(14, 12))
        
        # Determine which robots to plot
        if ROBOTS_TO_PLOT_TEST is None:
            # Plot all robots
            robots_to_show = list(range(len(robot_ids)))
        else:
            # Plot only specified robots (validate indices)
            robots_to_show = [idx for idx in ROBOTS_TO_PLOT_TEST if 0 <= idx < len(robot_ids)]
            if len(robots_to_show) == 0:
                print(f"  WARNING: No valid robot indices in ROBOTS_TO_PLOT_TEST. Defaulting to all robots.")
                robots_to_show = list(range(len(robot_ids)))
        
        print(f"  Plotting test trajectories for {len(robots_to_show)} robots: {[idx+1 for idx in robots_to_show]}")
        
        # Plot trajectory for selected robots (test portion only)
        colors = plt.cm.tab10(np.linspace(0, 1, len(robot_ids)))
        t_test = test_idx * DT
        
        for idx in robots_to_show:
            robot_id = robot_ids[idx]
            robot_states = get_robot_states(df, robot_id, verbose=False)
            pos_x_test = robot_states['pos_x'][test_idx]
            pos_y_test = robot_states['pos_y'][test_idx]
            linear_x_test = robot_states['linear_x'][test_idx]
            
            # Plot X position vs time (test only)
            axes_test[0].plot(t_test, pos_x_test, '-', color=colors[idx], linewidth=1.5, 
                            label=f'Robot {idx+1}', alpha=0.7)
            
            # Plot Y position vs time (test only)
            axes_test[1].plot(t_test, pos_y_test, '-', color=colors[idx], linewidth=1.5, 
                            label=f'Robot {idx+1}', alpha=0.7)
            
            # Plot linear velocity vs time (test only)
            axes_test[2].plot(t_test, linear_x_test, '-', color=colors[idx], linewidth=1.5, 
                            label=f'Robot {idx+1}', alpha=0.7)
        
        # Add grid to all subplots
        for ax in axes_test:
            ax.grid(True, alpha=0.3)
        
        # Configure subplots
        n_robots_plotted = len(robots_to_show)
        ncol_legend = min(5, n_robots_plotted)  # Adjust legend columns based on number of robots
        
        axes_test[0].set_ylabel('X Position (m)' )
        title_suffix = f" ({n_robots_plotted}/{len(robot_ids)} robots)" if ROBOTS_TO_PLOT_TEST is not None else ""
        axes_test[0].set_title(f'Robot Trajectories (Test Set) - Period Ratio {period_ratio}{title_suffix}', 
                               fontweight='bold')
        axes_test[0].legend(ncol=ncol_legend, loc='upper right')
        
        axes_test[1].set_ylabel('Y Position (m)')
        axes_test[1].legend(ncol=ncol_legend, loc='upper right')
        
        axes_test[2].set_xlabel('Time (s)')
        axes_test[2].set_ylabel('Linear Velocity (m/s)')
        axes_test[2].legend(ncol=ncol_legend, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(time_traj_test_plot_path, dpi=150, bbox_inches='tight')
        plt.savefig((time_traj_test_plot_path).split('.')[0] + '.pdf', dpi=150, bbox_inches='tight')
        plt.close(fig_test)
        print(f"  Saved test time trajectory plot: {time_traj_test_plot_path}")
        
        # Save regression weights heatmap
        weights = regressor.coef_
        n_features_per_robot = len(FEATURE_NAMES)
        n_robots_used = len(robot_ids)
        
        # Reshape weights to (n_robots, n_features_per_robot)
        weights_reshaped = weights.reshape(n_robots_used, n_features_per_robot)
        
        # Convert to percentages based on absolute magnitude
        total_abs_weight = np.sum(np.abs(weights))
        weights_percentage = (np.abs(weights_reshaped) / total_abs_weight) * 100
        
        # Keep sign information for colormap
        weights_percentage_signed = np.sign(weights_reshaped) * weights_percentage
        
        # Create heatmap
        fig_weights, ax_weights = plt.subplots(1, 1, figsize=(10, 4))
        
        im = ax_weights.imshow(weights_percentage_signed.T, aspect='auto', cmap='RdBu_r', 
                              vmin=-weights_percentage.max(), vmax=weights_percentage.max())
        
        # Set ticks and labels
        ax_weights.set_xticks(np.arange(n_robots_used))
        ax_weights.set_xticklabels([f'R{i+1}' for i in range(n_robots_used)])
        ax_weights.set_yticks(np.arange(n_features_per_robot))
        ax_weights.set_yticklabels([r"$x$ position", r"$y$ position"])  # Use LaTeX for better formatting
        
        ax_weights.set_xlabel('Robot ID')
        # ax_weights.set_ylabel('Feature')
        # ax_weights.set_title(f'Regression Weights (%) - NARMA-{narma_order}, Period Ratio {period_ratio}\nTest NMSE: {test_nmse:.4f}', 
                        #    fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_weights)
        cbar.set_label(r'Weight Contribution (\%)')
        
        # Add text annotations (show percentage with sign)
        for i in range(n_features_per_robot):
            for j in range(n_robots_used):
                text = ax_weights.text(j, i, f'{weights_percentage_signed[j, i]:+.1f}%',
                                      ha="center", va="center", color="black", fontsize=16)
        
        plt.tight_layout()
        
        # Save weights heatmap
        weights_plot_path = os.path.join(plot_dir, f'weights_narma{narma_order}_period{period_ratio}.png')
        plt.savefig(weights_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig_weights)
        print(f"  Saved weights heatmap: {weights_plot_path}")
        
        return {
            'nmse': test_nmse,
            'mse': test_mse,
            'train_nmse': train_nmse,
            'train_mse': train_mse,
            'weights': weights,
            'success': True,
            # Store data for combined plots
            'train_time': train_idx * DT,
            'train_target': y_train,
            'train_pred': train_pred,
            'test_time': test_idx * DT,
            'test_target': y_test,
            'test_pred': test_pred,
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'nmse': np.nan, 'mse': np.nan, 'train_nmse': np.nan, 'train_mse': np.nan, 'weights': None, 'success': False}


#=============================================================================
# Main execution
#=============================================================================

def main():
    """Main execution: evaluate all combinations and plot results."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Store results
    results = {order: {'period_ratios': [], 'nmse': [], 'mse': [], 'train_nmse': [], 'train_mse': []} 
               for order in NARMA_ORDERS}
    
    # Store period 80 results for combined plots
    period80_results = {}
    
    print("\n" + "="*80)
    print("NARMA EVALUATION ACROSS PERIOD RATIOS AND ORDERS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Period ratios: {sorted(DATA_FILES.keys())}")
    print(f"  NARMA orders: {NARMA_ORDERS}")
    print(f"  Ridge alpha: {RIDGE_ALPHA}")
    print(f"  Ridge solver: {RIDGE_SOLVER}")
    print(f"  Features: {FEATURE_NAMES}")
    print(f"  Smoothing: {SMOOTH_METHOD if SMOOTH_WINDOW_SIZE else 'None'} (window={SMOOTH_WINDOW_SIZE})")
    print(f"  Data processing: washout={WASHOUT_FRACTION*100:.0f}%, cutoff={CUTOFF_FRACTION*100:.0f}%, train={TRAIN_FRACTION*100:.0f}%")
    print("="*80)
    
    # Evaluate all combinations
    total_evals = len(DATA_FILES) * len(NARMA_ORDERS)
    pbar = tqdm(total=total_evals, desc="Evaluating")
    
    for period_ratio in sorted(DATA_FILES.keys()):
        data_file = DATA_FILES[period_ratio]
        
        # Check if file exists
        if not os.path.exists(data_file):
            print(f"\nWARNING: File not found: {data_file}")
            # Store NaN for all orders at this period ratio
            for order in NARMA_ORDERS:
                results[order]['period_ratios'].append(period_ratio)
                results[order]['nmse'].append(np.nan)
                results[order]['mse'].append(np.nan)
                results[order]['train_nmse'].append(np.nan)
                results[order]['train_mse'].append(np.nan)
                pbar.update(1)
            continue
        
        for narma_order in NARMA_ORDERS:
            res = evaluate_narma_for_period_ratio(data_file, period_ratio, narma_order)
            
            results[narma_order]['period_ratios'].append(period_ratio)
            results[narma_order]['nmse'].append(res['nmse'])
            results[narma_order]['mse'].append(res['mse'])
            results[narma_order]['train_nmse'].append(res['train_nmse'])
            results[narma_order]['train_mse'].append(res['train_mse'])
            
            # Store period 80 results for combined plots
            if period_ratio == 80 and res['success']:
                period80_results[narma_order] = res
            
            pbar.update(1)
    
    pbar.close()
    
    # Save results to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'narma_results.csv')
    rows = []
    for order in NARMA_ORDERS:
        for i in range(len(results[order]['period_ratios'])):
            rows.append({
                'narma_order': order,
                'period_ratio': results[order]['period_ratios'][i],
                'test_nmse': results[order]['nmse'][i],
                'test_mse': results[order]['mse'][i],
                'train_nmse': results[order]['train_nmse'][i],
                'train_mse': results[order]['train_mse'][i]
            })
    
    df_results = pd.DataFrame(rows)
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Create combined plots for period 80
    if len(period80_results) == len(NARMA_ORDERS):
        print("\nGenerating combined plots for period 80...")
        plot_dir = os.path.join(OUTPUT_DIR, 'prediction_plots')
        
        # Training plot (3 rows for NARMA 2, 3, 5)
        fig_train, axes_train = plt.subplots(3, 1, figsize=(14, 12))
        
        for idx, narma_order in enumerate(NARMA_ORDERS):
            if narma_order in period80_results:
                res = period80_results[narma_order]
                ax = axes_train[idx]
                
                ax.plot(res['train_time'], res['train_target'], 'b-', linewidth=1.5, 
                       label='Target', alpha=0.8)
                ax.plot(res['train_time'], res['train_pred'], 'r--', linewidth=1.5, 
                       label='Prediction', alpha=0.7)
                
                ax.set_ylabel('NARMA Output')
                ax.set_title(f'NARMA-{narma_order} Training (NMSE: {res["train_nmse"]:.4f})', 
                           fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                if idx == len(NARMA_ORDERS) - 1:
                    ax.set_xlabel('Time (s)')
        
        plt.tight_layout()
        train_plot_path = os.path.join(plot_dir, 'period80_combined_training.png')
        plt.savefig(train_plot_path, dpi=150, bbox_inches='tight')
        plt.savefig(train_plot_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close(fig_train)
        print(f"✓ Combined training plot saved: {train_plot_path}")
        
        # Test plot (3 rows for NARMA 2, 3, 5)
        fig_test, axes_test = plt.subplots(3, 1, figsize=(14, 12))
        
        for idx, narma_order in enumerate(NARMA_ORDERS):
            if narma_order in period80_results:
                res = period80_results[narma_order]
                ax = axes_test[idx]
                
                ax.plot(res['test_time'], res['test_target'], 'b-', linewidth=1.5, 
                       label='Target', alpha=0.8)
                ax.plot(res['test_time'], res['test_pred'], 'r--', linewidth=1.5, 
                       label='Prediction', alpha=0.7)
                
                ax.set_ylabel('NARMA Output')
                # ax.set_title(f'NARMA-{narma_order} Test (NMSE: {res["nmse"]:.4f})', 
                        #    fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                if idx == len(NARMA_ORDERS) - 1:
                    ax.set_xlabel('Time (s)')
        
        plt.tight_layout()
        test_plot_path = os.path.join(plot_dir, 'period80_combined_test.png')
        plt.savefig(test_plot_path, dpi=150, bbox_inches='tight')
        plt.savefig(test_plot_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close(fig_test)
        print(f"✓ Combined test plot saved: {test_plot_path}")
    
    # Create combined velocity plot for periods 40, 80, 120
    print("\nGenerating combined velocity plot for periods 40, 80, 120...")
    velocity_periods = [40, 80, 120]
    available_periods = [p for p in velocity_periods if p in DATA_FILES and os.path.exists(DATA_FILES[p])]
    
    if len(available_periods) > 0:
        fig_vel, axes_vel = plt.subplots(len(available_periods), 1, figsize=(14, 4*len(available_periods)))
        
        # Make axes_vel a list if only one period is available
        if len(available_periods) == 1:
            axes_vel = [axes_vel]
        
        for idx, period_ratio in enumerate(available_periods):
            data_file = DATA_FILES[period_ratio]
            ax = axes_vel[idx]
            
            try:
                # Load robot data
                df = load_robot_data(data_file)
                robot_ids = extract_robot_ids(df)
                robot_ids = robot_ids[:min(N_ROBOTS, len(robot_ids))]
                
                # Plot velocities for each robot
                colors = plt.cm.tab10(np.linspace(0, 1, len(robot_ids)))
                
                for robot_idx, robot_id in enumerate(robot_ids):
                    robot_states = get_robot_states(df, robot_id, verbose=False)
                    linear_x = robot_states['linear_x']
                    n_samples = len(linear_x)
                    t = np.arange(n_samples) * DT
                    
                    ax.plot(t, linear_x, '-', color=colors[robot_idx], linewidth=1.2, 
                           label=f'Robot {robot_idx+1}', alpha=0.7)
                
                ax.set_ylabel('Linear Velocity (m/s)')
                ax.set_title(f'Linear Velocities - Period Ratio {period_ratio}', 
                           fontweight='bold')
                ax.legend(ncol=5, loc='upper right', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                if idx == len(available_periods) - 1:
                    ax.set_xlabel('Time (s)')
                    
            except Exception as e:
                print(f"  Warning: Could not load velocities for period {period_ratio}: {e}")
        
        plt.tight_layout()
        vel_plot_path = os.path.join(plot_dir, 'combined_velocities.png')
        plt.savefig(vel_plot_path, dpi=150, bbox_inches='tight')
        plt.savefig(vel_plot_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close(fig_vel)
        print(f"✓ Combined velocity plot saved: {vel_plot_path}")
    
    # Plot results
    print("\nGenerating plots...")
    
    # Plot 1: Test NMSE vs Period Ratio (separate figure)
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    for order in NARMA_ORDERS:
        ax1.plot(results[order]['period_ratios'], results[order]['nmse'], 
                marker='o', linewidth=2, markersize=8, label=f'NARMA-{order}')
    
    ax1.set_xlabel('Input period ratio T [-]')
    ax1.set_ylabel('Test NMSE [-]')
    # ax1.set_title('NARMA Prediction Performance vs Period Ratio', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # ax1.set_yscale('log')  # Log scale for better visualization
    
    # plt.tight_layout()
    ax1.axhline(y=0.2, color='gray', linestyle='--', label='20 % NMSE Threshold')
    if SAVE_PLOT:
        # Save as PNG
        plot_path_png = os.path.join(OUTPUT_DIR, 'narma_performance_vs_period.png')
        plt.savefig(plot_path_png, dpi=150, bbox_inches='tight')
        print(f"✓ Performance plot saved to: {plot_path_png}")
        
        # Save as PDF
        plot_path_pdf = os.path.join(OUTPUT_DIR, 'narma_performance_vs_period.pdf')
        plt.savefig(plot_path_pdf, bbox_inches='tight')
        print(f"✓ Performance plot saved to: {plot_path_pdf}")
    
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig1)
    
    # Plot 2: Train vs Test NMSE for each order (separate figure)
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    
    x_pos = np.arange(len(NARMA_ORDERS))
    train_means = [np.nanmean(results[order]['train_nmse']) for order in NARMA_ORDERS]
    test_means = [np.nanmean(results[order]['nmse']) for order in NARMA_ORDERS]
    
    width = 0.35
    ax2.bar(x_pos - width/2, train_means, width, label='Train NMSE', alpha=0.8)
    ax2.bar(x_pos + width/2, test_means, width, label='Test NMSE', alpha=0.8)
    
    ax2.set_xlabel('NARMA Order')
    ax2.set_ylabel('Mean NMSE')
    ax2.set_title('Average Performance Across All Period Ratios', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'NARMA-{o}' for o in NARMA_ORDERS])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if SAVE_PLOT:
        # Save as PNG
        plot_path_png = os.path.join(OUTPUT_DIR, 'narma_average_performance.png')
        plt.savefig(plot_path_png, dpi=150, bbox_inches='tight')
        print(f"✓ Average performance plot saved to: {plot_path_png}")
        
        # Save as PDF
        plot_path_pdf = os.path.join(OUTPUT_DIR, 'narma_average_performance.pdf')
        plt.savefig(plot_path_pdf, bbox_inches='tight')
        print(f"✓ Average performance plot saved to: {plot_path_pdf}")
    
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig2)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for order in NARMA_ORDERS:
        mean_test_nmse = np.nanmean(results[order]['nmse'])
        std_test_nmse = np.nanstd(results[order]['nmse'])
        print(f"NARMA-{order}: Mean Test NMSE = {mean_test_nmse:.6f} ± {std_test_nmse:.6f}")
    print("="*80)


if __name__ == '__main__':
    main()
