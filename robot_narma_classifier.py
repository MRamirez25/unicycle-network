#!/usr/bin/env python3
"""
Train a NARMA classifier using robot state data.

This script loads robot state data from robot_state_log.csv,
generates NARMA targets from the global input signal,
and trains a Ridge regression classifier using robot states as features.
"""
#%%
#%%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
SCRIPT_DIR = '/home/mariano/phd_code/unicycle-network/'
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
#%%
from plot_robot_data import load_robot_data, extract_robot_ids, get_robot_states
from plot_robot_data import load_robot_data, extract_robot_ids, get_robot_states

#%%
def u(t, period_ratio=2, amplitude=0.2):
    """Input signal: sum of three sinusoids"""
    f1 = 2.11
    f2 = 3.73
    f3 = 4.33
    T = period_ratio
    u_val = amplitude * np.sin(2*np.pi * f1 * t / T) * np.sin(2*np.pi * f2 * t / T) * np.sin(2*np.pi * f3 * t / T)
    return u_val


def generate_narma(n_samples, narma_order=2, period_ratio=2, amplitude=0.2, dt=0.01, u_input=None):
    """
    Generate NARMA-n time series.
    
    Args:
        n_samples: Number of time steps
        narma_order: Order of NARMA system (2, 3, or n >= 4)
        period_ratio: Period scaling for input signal (only used if u_input is None)
        amplitude: Amplitude of input signal (only used if u_input is None)
        dt: Time step (only used if u_input is None)
        u_input: Optional pre-defined input signal array. If provided, this will be used
                instead of generating a synthetic input. Will be normalized to [0, 0.5].
    
    Returns:
        u_input: Raw input signal (either provided or generated)
        u_normalized: Normalized input [0, 0.5] used for NARMA generation
        y_narma: NARMA output series
    """
    # Use provided input or generate synthetic input signal
    if u_input is None:
        # Generate input signal using the sinusoidal function
        t = np.arange(n_samples) * dt
        u_input = np.array([u(ti, period_ratio=period_ratio, amplitude=amplitude) for ti in t])
    else:
        # Use the provided input signal
        if len(u_input) != n_samples:
            raise ValueError(f"Provided u_input has length {len(u_input)}, but n_samples={n_samples}")
        u_input = np.array(u_input)  # Ensure it's a numpy array
    
    # Normalize input to [0, 0.5] for NARMA generation (standard NARMA input range)
    u_normalized = u_input - u_input.min()
    u_normalized = 0.5 * u_normalized / u_normalized.max()
    
    # Initialize NARMA output
    y_narma = np.zeros(n_samples)
    
    # Generate NARMA series based on order using normalized input
    if narma_order == 2:
        # NARMA-2 equations
        for k in range(1, n_samples - 1):
            y_narma[k + 1] = (0.4 * y_narma[k] + 
                            0.4 * y_narma[k] * y_narma[k - 1] + 
                            0.6 * u_normalized[k]**3 + 0.1)
                            
    elif narma_order == 3:
        # NARMA-3 equations
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
    
    # Check for numerical issues and normalize if needed
    if np.max(np.abs(y_narma)) > 10:
        print(f'WARNING: NARMA output exceeded safe range (max={np.max(np.abs(y_narma)):.2f}). Normalizing...')
        y_narma = y_narma / np.max(np.abs(y_narma))
    
    return u_input, u_normalized, y_narma

#%%
def extract_robot_features(df, robot_ids, feature_names=None, start_idx=None, end_idx=None, 
                          use_local_frame=False, washout_idx=0, verbose=False):
    """
    Extract feature matrix from robot states.
    
    Args:
        df: DataFrame with robot data
        robot_ids: List of robot IDs to use
        feature_names: List of feature names to extract. 
                      Options: ['pos_x', 'pos_y', 'linear_x', 'qz', 'qw', 'omega']
                      Default: all features
        start_idx: Index to start from (trim beginning). Default: 0
        end_idx: Index to end at (trim end). Default: None (use all)
        use_local_frame: If True, shift positions to local frame centered at post-washout position
        washout_idx: Index where washout period ends (used for local frame centering)
        verbose: If True, print which column format is being used for each robot
    
    Returns:
        features: numpy array of shape (n_timesteps, n_features)
            Features are concatenated: [robot1_states, robot2_states, ...]
            Each robot contributes selected features in order
    """
    # Default to all features if none specified
    if feature_names is None:
        feature_names = ['pos_x', 'pos_y', 'linear_x', 'qz', 'qw', 'omega']
    
    # Validate feature names
    valid_features = ['pos_x', 'pos_y', 'linear_x', 'qz', 'qw', 'omega']
    for fname in feature_names:
        if fname not in valid_features:
            raise ValueError(f"Invalid feature name: {fname}. Valid options: {valid_features}")
    
    # Check if we have any robots
    if len(robot_ids) == 0:
        raise ValueError("No robot IDs provided. Cannot extract features from empty robot list.")
    
    all_features = []
    
    for robot_id in robot_ids:
        states = get_robot_states(df, robot_id, verbose=verbose)
        
        # Apply local frame transformation if requested
        if use_local_frame and washout_idx < len(states['pos_x']):
            # Center positions at post-washout location
            states['pos_x'] = states['pos_x'] - states['pos_x'][washout_idx]
            states['pos_y'] = states['pos_y'] - states['pos_y'][washout_idx]
        
        # Extract only requested features
        robot_feature_list = [states[fname] for fname in feature_names]
        
        # Stack features for this robot
        robot_features = np.stack(robot_feature_list, axis=1)  # Shape: (n_timesteps, n_selected_features)
        
        all_features.append(robot_features)
    
    # Check that we actually collected features
    if len(all_features) == 0:
        raise ValueError(f"No features extracted. Check that robot_ids is not empty: {robot_ids}")
    
    # Concatenate all robot features
    features = np.concatenate(all_features, axis=1)  # Shape: (n_timesteps, n_robots * n_selected_features)
    
    # Trim data if requested
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = features.shape[0]
    
    features = features[start_idx:end_idx, :]
    
    return features

#%%
def smooth_features(features, window_size=5, method='moving_average'):
    """
    Smooth features to reduce noise and improve regression performance.
    
    WHEN TO SMOOTH:
    - **BEFORE scaling**: Recommended! Preserves the smoothing across all features uniformly.
                         Smoothing raw data removes sensor noise and measurement artifacts.
    - **AFTER scaling**: Can work, but less intuitive. Smoothing changes the mean/std slightly.
    
    WHY SMOOTHING HELPS:
    - Removes high-frequency noise that confuses the regressor
    - Makes features more correlated with smooth NARMA targets
    - Reduces overfitting to measurement noise
    - Can improve generalization (test performance)
    
    CAUTION:
    - Too much smoothing loses dynamic information needed for prediction
    - Can blur important transients/peaks (bad for peak tracking!)
    - Start with small window (3-5) and increase carefully
    
    Args:
        features: numpy array of shape (n_timesteps, n_features)
        window_size: Size of smoothing window (odd number recommended)
                    Typical values: 3, 5, 7, 11
        method: Smoothing method
                'moving_average': Simple moving average (uniform weights)
                'gaussian': Gaussian smoothing (smooth, preserves peaks better)
                'savgol': Savitzky-Golay filter (preserves peaks, polynomial fit)
    
    Returns:
        smoothed_features: numpy array with same shape as input
    """
    from scipy.ndimage import uniform_filter1d, gaussian_filter1d
    from scipy.signal import savgol_filter
    
    smoothed = features.copy()
    
    if method == 'moving_average':
        # Simple moving average - fast, uniform smoothing
        for col_idx in range(features.shape[1]):
            smoothed[:, col_idx] = uniform_filter1d(features[:, col_idx], size=window_size, mode='nearest')
    
    elif method == 'gaussian':
        # Gaussian smoothing - smoother, better for general noise reduction
        sigma = window_size / 3.0  # Rule of thumb: window_size ≈ 3*sigma
        for col_idx in range(features.shape[1]):
            smoothed[:, col_idx] = gaussian_filter1d(features[:, col_idx], sigma=sigma, mode='nearest')
    
    elif method == 'savgol':
        # Savitzky-Golay - best for preserving peaks while smoothing
        # Window must be odd and >= polyorder + 2
        if window_size % 2 == 0:
            window_size += 1  # Make it odd
        polyorder = min(3, window_size - 2)  # Polynomial order
        for col_idx in range(features.shape[1]):
            smoothed[:, col_idx] = savgol_filter(features[:, col_idx], window_length=window_size, 
                                                polyorder=polyorder, mode='nearest')
    else:
        raise ValueError(f"Unknown smoothing method: {method}. Use 'moving_average', 'gaussian', or 'savgol'")
    
    return smoothed

#%%
# Plot all robot states as timeseries
def plot_all_robots_timeseries(df, figsize=(16, 12)):
    robot_ids = extract_robot_ids(df)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f'All Robot States Over Time ({len(robot_ids)} robots)', fontsize=14)
    colors = plt.cm.tab10(np.linspace(0, 1, len(robot_ids)))
    for i, robot_id in enumerate(robot_ids):
        states = get_robot_states(df, robot_id)
        label = f'Robot {robot_id[:8]}'
        color = colors[i]
        axes[0, 0].plot(states['t'], states['pos_x'], label=label, color=color, alpha=0.7)
        axes[0, 1].plot(states['t'], states['pos_y'], label=label, color=color, alpha=0.7)
        axes[1, 0].plot(states['t'], states['linear_x'], label=label, color=color, alpha=0.7)
        axes[1, 1].plot(states['t'], states['omega'], label=label, color=color, alpha=0.7)
        axes[2, 0].plot(states['t'], states['qz'], label=label, color=color, alpha=0.7)
        axes[2, 1].plot(states['t'], states['qw'], label=label, color=color, alpha=0.7)
    axes[0, 0].set_ylabel('X Position')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_ylabel('Y Position')
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].set_ylabel('Linear Velocity')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].set_ylabel('Angular Velocity (ω)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[2, 0].set_ylabel('Quaternion qz')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 1].set_ylabel('Quaternion qw')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot instead of showing
    timeseries_file = os.path.join(SCRIPT_DIR, 'robot_states_timeseries.png')
    plt.savefig(timeseries_file, dpi=150, bbox_inches='tight')
    print(f"Saved robot timeseries plot: {timeseries_file}")
    plt.close()

#%%
def train_narma_classifier(df, narma_order=2, period_ratio=2, amplitude=0.5, 
                          washout_fraction=0.1, train_fraction=0.7, 
                          ridge_alpha=1e-6, verbose=True,
                          feature_names=None, start_idx=None, end_idx=None,
                          use_local_frame=False, exclude_robots=None,
                          smooth_features_flag=False, smooth_window=5, smooth_method='savgol'):
    """
    Train a NARMA classifier using robot state data.
    
    Args:
        df: DataFrame with robot data
        narma_order: Order of NARMA system
        period_ratio: Period ratio for input signal
        amplitude: Amplitude of input signal
        washout_fraction: Fraction of data to use for washout
        train_fraction: Fraction of valid data to use for training
        ridge_alpha: Ridge regression regularization parameter
        verbose: Print progress information
        feature_names: List of feature names to use. 
                      Options: ['pos_x', 'pos_y', 'linear_x', 'qz', 'qw', 'omega']
                      Default: all features
        start_idx: Index to start from (trim beginning). Default: 0
        end_idx: Index to end at (trim end). Default: None (use all)
        use_local_frame: If True, shift positions to local frame centered at post-washout position.
                        This makes each robot's position relative to where it was after washout.
        exclude_robots: List of robot IDs to exclude from training. 
                       Example: ['r00', 'r05'] to exclude robots r00 and r05.
                       Default: None or [] = use all robots
        smooth_features_flag: If True, apply smoothing to features BEFORE scaling.
                             Helps reduce noise and improve regression stability.
                             Recommended for noisy sensor data. Default: False
        smooth_window: Window size for smoothing (3-11 typical). Default: 5
        smooth_method: Smoothing method. Options:
                      'savgol' - Best for preserving peaks (recommended for NARMA)
                      'gaussian' - Good general smoothing
                      'moving_average' - Simple uniform smoothing
    
    Returns:
        Dictionary with:
            - regressor: Trained Ridge regression model
            - scaler: Fitted StandardScaler
            - results: Dictionary with predictions and metrics
    """
    
    # Extract robot IDs
    robot_ids = extract_robot_ids(df)
    
    # Filter out excluded robots
    if exclude_robots is None:
        exclude_robots = []
    
    if exclude_robots:
        original_count = len(robot_ids)
        robot_ids = [rid for rid in robot_ids if rid not in exclude_robots]
        excluded_count = original_count - len(robot_ids)
        if verbose and excluded_count > 0:
            print(f"\nExcluded {excluded_count} robot(s): {exclude_robots}")
            if excluded_count < len(exclude_robots):
                not_found = [r for r in exclude_robots if r not in extract_robot_ids(df)]
                if not_found:
                    print(f"  Note: {not_found} were not found in the data")
    
    n_robots = len(robot_ids)
    
    # Check if we found any robots
    if n_robots == 0:
        print(f"\n{'='*60}")
        print("ERROR: No robot IDs found in DataFrame!")
        print(f"{'='*60}")
        print(f"DataFrame columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        print(f"Total columns: {len(df.columns)}")
        print("\nExpected column format: 'robot_XXXXX_linear_x'")
        print("Please check if your CSV file has the correct column naming.")
        raise ValueError("No robot IDs found in DataFrame. Check column naming format.")
    
    # Default to all features if none specified
    if feature_names is None:
        feature_names = ['pos_x', 'pos_y', 'linear_x', 'qz', 'qw', 'omega']
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training NARMA-{narma_order} Classifier with Robot Data")
        print(f"{'='*60}")
        print(f"Number of robots: {n_robots}")
        print(f"Robot IDs: {[rid[:8] for rid in robot_ids]}")
        print(f"Selected features: {feature_names}")
        if start_idx is not None or end_idx is not None:
            print(f"Data trimming: start_idx={start_idx}, end_idx={end_idx}")
        if use_local_frame:
            print(f"Using local frame: positions centered at post-washout location")
    
    # Calculate washout index (before trimming)
    # We need to know where washout ends in the original data
    if start_idx is None:
        initial_start = 0
    else:
        initial_start = start_idx
    
    # Washout applies to the trimmed data
    washout_idx_in_trimmed = int(washout_fraction * ((end_idx if end_idx else len(get_robot_states(df, robot_ids[0])['t'])) - initial_start))
    washout_idx_absolute = initial_start + washout_idx_in_trimmed
    
    # Extract features from robot states
    if verbose:
        print("\nExtracting robot features...")
    features = extract_robot_features(df, robot_ids, feature_names=feature_names, 
                                     start_idx=start_idx, end_idx=end_idx,
                                     use_local_frame=use_local_frame, 
                                     washout_idx=washout_idx_absolute,
                                     verbose=verbose)
    n_samples, n_features = features.shape
    
    # Apply smoothing BEFORE scaling if requested
    if smooth_features_flag:
        if verbose:
            print(f"\nApplying {smooth_method} smoothing (window={smooth_window})...")
            print(f"  Smoothing helps reduce noise and can improve regression stability")
        features = smooth_features(features, window_size=smooth_window, method=smooth_method)
    
    if verbose:
        print(f"Feature matrix shape: {features.shape}")
        print(f"Features per robot: {len(feature_names)} {feature_names}")
        print(f"Total features: {n_features} ({n_robots} robots × {len(feature_names)} states)")
        if smooth_features_flag:
            print(f"  (Smoothed with {smooth_method}, window={smooth_window})")
    
    # Get time and global input from first robot (also trim to match features)
    states = get_robot_states(df, robot_ids[0])
    
    # Apply same trimming to time and global_u
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(states['t'])
    
    time_vector = states['t'][start_idx:end_idx]
    global_u = states['global_u'][start_idx:end_idx] if states['global_u'] is not None else None
    
    if global_u is None:
        raise ValueError("No global_u found in data! Cannot generate NARMA targets.")
    
    # Calculate dt from time vector
    dt = np.mean(np.diff(time_vector))
    
    if verbose:
        print(f"\nTime series information:")
        print(f"  Total timesteps: {n_samples}")
        print(f"  Time step (dt): {dt:.6f} s")
        print(f"  Total time: {time_vector[-1] - time_vector[0]:.2f} s")
        print(f"  Global input range: [{global_u.min():.6f}, {global_u.max():.6f}]")
    
    # Generate NARMA targets using the global input
    if verbose:
        print(f"\nGenerating NARMA-{narma_order} targets...")
        print(f"Using global_u from robot data as NARMA input")
    u_input, u_normalized, y_narma = generate_narma(
        n_samples=n_samples,
        narma_order=narma_order,
        period_ratio=period_ratio,
        amplitude=amplitude,
        dt=dt,
        u_input=global_u  # Use actual global_u from robot data
    )
    
    if verbose:
        print(f"NARMA input normalization:")
        print(f"  Original global_u range: [{global_u.min():.6f}, {global_u.max():.6f}]")
        print(f"  Normalized to: [0.0, 0.5] for NARMA generation")
        print(f"NARMA target stats:")
        print(f"  Range: [{y_narma.min():.6f}, {y_narma.max():.6f}]")
        print(f"  Mean: {y_narma.mean():.6f}, Std: {y_narma.std():.6f}")
    
    # Split data into train/test with washout
    washout_samples = int(washout_fraction * n_samples)
    valid_idx = np.arange(washout_samples, n_samples)
    n_valid = len(valid_idx)
    n_train = int(train_fraction * n_valid)
    
    train_idx = valid_idx[:n_train]
    test_idx = valid_idx[n_train:]
    
    if verbose:
        print(f"\nData split:")
        print(f"  Washout: {washout_samples} samples ({washout_fraction * 100:.1f}%)")
        print(f"  Training: {len(train_idx)} samples ({train_fraction * 100:.1f}%)")
        print(f"  Testing: {len(test_idx)} samples ({(1 - train_fraction) * 100:.1f}%)")
    
    # Extract train/test features and targets
    X_train = features[train_idx, :]
    X_test = features[test_idx, :]
    y_train = y_narma[train_idx]
    y_test = y_narma[test_idx]
    
    # Scale features
    if verbose:
        print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if verbose:
        print(f"Scaled train stats - mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
        
        # Check for features with unusual magnitudes after scaling
        feature_magnitudes = np.abs(X_train_scaled).max(axis=0)
        mag_mean = feature_magnitudes.mean()
        mag_std = feature_magnitudes.std()
        outlier_features = np.where(feature_magnitudes > mag_mean + 3*mag_std)[0]
        
        if len(outlier_features) > 0:
            print(f"\n⚠ Warning: {len(outlier_features)} features have unusually large magnitudes after scaling:")
            for feat_idx in outlier_features[:5]:  # Show first 5
                robot_idx = feat_idx // len(feature_names)
                feat_type_idx = feat_idx % len(feature_names)
                feat_name = feature_names[feat_type_idx]
                robot_label = robot_ids[robot_idx][:8] if len(robot_ids[robot_idx]) > 8 else robot_ids[robot_idx]
                print(f"  Feature {feat_idx}: Robot {robot_label}, {feat_name} - max magnitude: {feature_magnitudes[feat_idx]:.2f}")
            print(f"  (Mean magnitude: {mag_mean:.2f}, Std: {mag_std:.2f})")
            print(f"  This suggests outliers or extreme values in the original data.")
    
    # Train Ridge regression
    if verbose:
        print(f"\nTraining Ridge regression (alpha={ridge_alpha:.2e})...")
    regressor = Ridge(alpha=ridge_alpha, max_iter=10000, solver='auto', fit_intercept=True)
    regressor.fit(X_train_scaled, y_train)
    
    # Print weight statistics
    weights = regressor.coef_.ravel()
    intercept = regressor.intercept_
    intercept_val = intercept[0] if isinstance(intercept, np.ndarray) and intercept.size > 0 else float(intercept)
    
    if verbose:
        print(f"\nRegressor statistics:")
        print(f"  Weights - mean: {weights.mean():.6f}, std: {weights.std():.6f}")
        print(f"  Weights - max: {weights.max():.6f}, min: {weights.min():.6f}")
        print(f"  Weights - L2 norm: {np.linalg.norm(weights):.6f}")
        print(f"  Intercept: {intercept_val:.6f}")
        
        # Check if weights might be too small
        avg_abs_weight = np.abs(weights).mean()
        if avg_abs_weight < 0.001:
            print(f"\n  ⚠ Warning: Average absolute weight is very small ({avg_abs_weight:.6e})")
            print(f"    This could indicate:")
            print(f"    - Over-regularization (try reducing RIDGE_ALPHA)")
            print(f"    - Weak relationship between features and target")
            print(f"    - Features need better engineering")
        elif avg_abs_weight > 1.0:
            print(f"\n  ⚠ Warning: Average absolute weight is large ({avg_abs_weight:.6f})")
            print(f"    This could indicate under-regularization (try increasing RIDGE_ALPHA)")
    
    # Make predictions
    train_predictions = regressor.predict(X_train_scaled)
    test_predictions = regressor.predict(X_test_scaled)
    
    # Calculate metrics
    train_mse = np.mean(np.square(train_predictions - y_train))
    test_mse = np.mean(np.square(test_predictions - y_test))
    
    train_var = np.var(y_train)
    test_var = np.var(y_test)
    
    train_nmse = train_mse / (train_var + 1e-9)
    test_nmse = test_mse / (test_var + 1e-9)
    
    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Training MSE: {train_mse:.6f}")
        print(f"Training NMSE: {train_nmse:.6f}")
        print(f"Testing MSE: {test_mse:.6f}")
        print(f"Testing NMSE: {test_nmse:.6f}")
        print(f"{'='*60}\n")
    
    # Package results
    results = {
        'regressor': regressor,
        'scaler': scaler,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions,
        'train_targets': y_train,
        'test_targets': y_test,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_nmse': train_nmse,
        'test_nmse': test_nmse,
        'weights': weights,
        'intercept': intercept_val,
        'n_robots': n_robots,
        'robot_ids': robot_ids,
        'time': time_vector,
        'global_u': global_u,
        'u_input': u_input,
        'y_narma': y_narma,
        'features': features,
        'feature_names': feature_names  # Add feature names to results
    }
    
    return results

#%%
def plot_results(results, narma_order=2, save_dir=None):
    """
    Create visualization plots for NARMA classifier results.
    
    Args:
        results: Dictionary returned by train_narma_classifier
        narma_order: Order of NARMA system
        save_dir: Directory to save plots (default: current directory)
    """
    if save_dir is None:
        save_dir = SCRIPT_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    time = results['time']
    train_idx = results['train_idx']
    test_idx = results['test_idx']
    train_pred = results['train_predictions']
    test_pred = results['test_predictions']
    train_target = results['train_targets']
    test_target = results['test_targets']
    u_input = results['u_input']
    test_nmse = results['test_nmse']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot 1: Input signal (full)
    axes[0].plot(time, u_input, label='Input Signal', alpha=0.7, color='gray')
    axes[0].axvspan(time[train_idx[0]], time[train_idx[-1]], alpha=0.2, color='blue', label='Train')
    axes[0].axvspan(time[test_idx[0]], time[test_idx[-1]], alpha=0.2, color='orange', label='Test')
    axes[0].set_ylabel('Input u(t)', fontsize=12)
    axes[0].set_title(f'NARMA-{narma_order} Input Signal and Data Split', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Training predictions vs targets
    axes[1].plot(time[train_idx], train_target, label='Target', alpha=0.7, linewidth=2)
    axes[1].plot(time[train_idx], train_pred, label='Prediction', alpha=0.7, linewidth=2)
    axes[1].set_ylabel('Output y(t)', fontsize=12)
    axes[1].set_title(f'Training Set (NMSE={results["train_nmse"]:.4f})', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Test predictions vs targets
    axes[2].plot(time[test_idx], test_target, label='Target', alpha=0.7, linewidth=2)
    axes[2].plot(time[test_idx], test_pred, label='Prediction', alpha=0.7, linewidth=2)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Output y(t)', fontsize=12)
    axes[2].set_title(f'Test Set (NMSE={test_nmse:.4f})', fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(save_dir, f'robot_narma{narma_order}_predictions.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved predictions plot: {plot_file}")
    plt.close()
    
    # Create weight analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    weights = results['weights']
    n_robots = results['n_robots']
    robot_ids = results['robot_ids']
    
    # Reshape weights by robot and state
    feature_labels = results.get('feature_names', ['pos_x', 'pos_y', 'linear_x', 'qz', 'qw', 'omega'])
    n_states_per_robot = len(feature_labels)
    weights_reshaped = weights.reshape(n_robots, n_states_per_robot)
    
    state_names = feature_labels
    
    # Plot 1: Heatmap with weight values
    im = axes[0, 0].imshow(weights_reshaped.T, aspect='auto', cmap='RdBu_r',
                           vmin=-np.abs(weights).max(), vmax=np.abs(weights).max())
    axes[0, 0].set_xlabel('Robot Index', fontsize=12)
    axes[0, 0].set_ylabel('State Type', fontsize=12)
    axes[0, 0].set_yticks(range(n_states_per_robot))
    axes[0, 0].set_yticklabels(state_names)
    axes[0, 0].set_title('Ridge Regressor Weights Heatmap', fontsize=12)
    
    # Add weight values as text on heatmap
    for i in range(n_states_per_robot):
        for j in range(n_robots):
            weight_val = weights_reshaped[j, i]
            # Choose text color based on background
            text_color = 'white' if abs(weight_val) > np.abs(weights).max() * 0.5 else 'black'
            axes[0, 0].text(j, i, f'{weight_val:.3f}', 
                          ha='center', va='center', color=text_color, fontsize=8)
    plt.colorbar(im, ax=axes[0, 0], label='Weight value')
    
    # Plot 2: Weight distribution
    axes[0, 1].hist(weights, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[0, 1].axvline(weights.mean(), color='green', linestyle='--', linewidth=2, label='Mean')
    axes[0, 1].set_xlabel('Weight value', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].set_title('Weight Distribution', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Weights by state type
    state_weights = [weights_reshaped[:, i] for i in range(n_states_per_robot)]
    bp = axes[1, 0].boxplot(state_weights, labels=state_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.6)
    axes[1, 0].set_ylabel('Weight value', fontsize=12)
    axes[1, 0].set_xlabel('State type', fontsize=12)
    axes[1, 0].set_title('Weight Distribution by State Type', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 4: Cumulative weight contribution
    sorted_abs_weights = np.sort(np.abs(weights))[::-1]
    cumulative_contrib = np.cumsum(sorted_abs_weights) / np.sum(sorted_abs_weights)
    axes[1, 1].plot(cumulative_contrib, linewidth=2)
    axes[1, 1].axhline(0.9, color='red', linestyle='--', alpha=0.7, label='90%')
    axes[1, 1].axhline(0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
    axes[1, 1].set_xlabel('Number of weights (sorted by magnitude)', fontsize=12)
    axes[1, 1].set_ylabel('Cumulative contribution', fontsize=12)
    axes[1, 1].set_title('Cumulative Weight Contribution', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Ridge Regressor Weight Analysis ({n_robots} robots)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    weights_file = os.path.join(save_dir, f'robot_narma{narma_order}_weights.png')
    plt.savefig(weights_file, dpi=150, bbox_inches='tight')
    print(f"Saved weights plot: {weights_file}")
    plt.close()

#%%
def plot_features(results, narma_order=2, save_dir=None, max_samples=500):
    """
    Visualize selected features from robot data.
    
    Args:
        results: Dictionary returned by train_narma_classifier
        narma_order: Order of NARMA system
        save_dir: Directory to save plots (default: current directory)
        max_samples: Maximum number of samples to plot (for readability)
    """
    if save_dir is None:
        save_dir = SCRIPT_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    
    import matplotlib.pyplot as plt
    
    # Extract data
    features = results['features']
    feature_names = results['feature_names']
    n_robots = results['n_robots']
    robot_ids = results['robot_ids']
    time = results['time']
    y_narma = results['y_narma']
    train_idx = results['train_idx']
    test_idx = results['test_idx']
    
    n_features = len(feature_names)
    n_samples = min(max_samples, len(features))
    
    # Limit data for plotting
    plot_features = features[:n_samples]
    plot_time = time[:n_samples]
    plot_target = y_narma[:n_samples]
    
    # Create figure with subplots for each feature type
    fig, axes = plt.subplots(n_features + 1, 1, figsize=(15, 3*(n_features + 1)))
    if n_features == 0:
        axes = [axes]
    
    # Plot NARMA target at top
    axes[0].plot(plot_time, plot_target, color='black', linewidth=2, label='NARMA target')
    axes[0].set_ylabel('NARMA output', fontsize=12)
    axes[0].set_title(f'NARMA-{narma_order} Target Signal', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot each feature type across all robots
    for feat_idx, feat_name in enumerate(feature_names):
        ax = axes[feat_idx + 1]
        
        # Plot this feature for each robot
        for robot_idx in range(n_robots):
            col_idx = robot_idx * n_features + feat_idx
            robot_label = robot_ids[robot_idx][:8] if len(robot_ids[robot_idx]) > 8 else robot_ids[robot_idx]
            ax.plot(plot_time, plot_features[:, col_idx], alpha=0.7, linewidth=1, 
                   label=f'Robot {robot_label}')
        
        ax.set_ylabel(feat_name, fontsize=12)
        ax.set_title(f'Feature: {feat_name} (all robots)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Only show legend if not too many robots
        if n_robots <= 5:
            ax.legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    
    plt.suptitle(f'Robot Features for NARMA-{narma_order} Classification\n'
                 f'{n_robots} robots, {n_features} features per robot',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    features_file = os.path.join(save_dir, f'robot_narma{narma_order}_features.png')
    plt.savefig(features_file, dpi=150, bbox_inches='tight')
    print(f"Saved features plot: {features_file}")
    plt.close()

#%%
def plot_feature_scaling(results, narma_order=2, save_dir=None, max_samples=1000):
    """
    Visualize features before and after scaling on the same plot.
    
    Args:
        results: Dictionary returned by train_narma_classifier
        narma_order: Order of NARMA system
        save_dir: Directory to save plots (default: current directory)
        max_samples: Maximum number of samples to plot (for readability)
    """
    if save_dir is None:
        save_dir = SCRIPT_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    
    # Extract data
    features = results['features']
    feature_names = results['feature_names']
    n_robots = results['n_robots']
    robot_ids = results['robot_ids']
    time = results['time']
    train_idx = results['train_idx']
    scaler = results['scaler']
    
    n_features = len(feature_names)
    n_samples = min(max_samples, len(features))
    
    # Get training data for scaling reference
    X_train = features[train_idx, :]
    X_train_scaled = scaler.transform(X_train)
    
    # Scale all features for plotting
    features_scaled = scaler.transform(features)
    
    # Limit data for plotting
    plot_features_raw = features[:n_samples]
    plot_features_scaled = features_scaled[:n_samples]
    plot_time = time[:n_samples]
    
    # Create figure with subplots for each feature type
    # Each subplot shows before (left) and after (right) scaling
    fig, axes = plt.subplots(n_features, 2, figsize=(18, 3*n_features))
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each feature type across all robots
    for feat_idx, feat_name in enumerate(feature_names):
        # Left column: Raw features
        ax_raw = axes[feat_idx, 0]
        # Right column: Scaled features
        ax_scaled = axes[feat_idx, 1]
        
        # Plot this feature for each robot
        for robot_idx in range(n_robots):
            col_idx = robot_idx * n_features + feat_idx
            robot_label = robot_ids[robot_idx][:8] if len(robot_ids[robot_idx]) > 8 else robot_ids[robot_idx]
            
            # Raw feature
            ax_raw.plot(plot_time, plot_features_raw[:, col_idx], alpha=0.7, linewidth=1, 
                       label=f'Robot {robot_label}')
            
            # Scaled feature
            ax_scaled.plot(plot_time, plot_features_scaled[:, col_idx], alpha=0.7, linewidth=1,
                          label=f'Robot {robot_label}')
        
        # Configure raw features axis
        ax_raw.set_ylabel(f'{feat_name}\n(raw)', fontsize=12)
        ax_raw.set_title(f'Before Scaling: {feat_name}', fontsize=12, fontweight='bold')
        ax_raw.grid(True, alpha=0.3)
        if n_robots <= 5:
            ax_raw.legend(loc='upper right', fontsize=8)
        
        # Configure scaled features axis
        ax_scaled.set_ylabel(f'{feat_name}\n(scaled)', fontsize=12)
        ax_scaled.set_title(f'After Scaling: {feat_name}\n(mean=0, std=1)', fontsize=12, fontweight='bold')
        ax_scaled.grid(True, alpha=0.3)
        ax_scaled.axhline(0, color='red', linestyle='--', alpha=0.3, linewidth=1)
        if n_robots <= 5:
            ax_scaled.legend(loc='upper right', fontsize=8)
        
        # X-label only on bottom row
        if feat_idx == n_features - 1:
            ax_raw.set_xlabel('Time (s)', fontsize=12)
            ax_scaled.set_xlabel('Time (s)', fontsize=12)
    
    plt.suptitle(f'Feature Scaling Comparison for NARMA-{narma_order}\n'
                 f'{n_robots} robots, StandardScaler applied',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    scaling_file = os.path.join(save_dir, f'robot_narma{narma_order}_feature_scaling.png')
    plt.savefig(scaling_file, dpi=150, bbox_inches='tight')
    print(f"Saved feature scaling plot: {scaling_file}")
    plt.close()

#%%
# Example usage
if __name__ == "__main__":
    # Configuration
    NARMA_ORDER = 5
    PERIOD_RATIO = 70
    AMPLITUDE = 0.5
    WASHOUT_FRACTION = 0.15
    TRAIN_FRACTION = 0.7
    RIDGE_ALPHA = 6
    
    # Feature selection - choose which features to use from each robot
    # Options: 'pos_x', 'pos_y', 'linear_x', 'qz', 'qw', 'omega'
    # Example: Use only position and velocity features
    # FEATURE_NAMES = ['pos_x', 'pos_y', 'linear_x', 'omega']
    # Default: use all features
    FEATURE_NAMES = ['pos_x', 'pos_y', 'linear_x']  # None = use all features
    
    # Local frame option - center positions at post-washout location
    # If True, each robot's position will be relative to where it was after washout
    # This removes global position offsets and focuses on relative movements
    USE_LOCAL_FRAME = False  # Set to False to use absolute positions
    
    # Robot exclusion - specify robot IDs to exclude from training
    # Example: EXCLUDE_ROBOTS = ['r05', 'r12']  # Exclude robots r05 and r12
    # Default: None or [] = use all robots
    EXCLUDE_ROBOTS = []  # List of robot IDs to exclude, e.g., ['r00', 'r03']
    
    # Feature smoothing - reduce noise in sensor data
    # Smoothing is applied BEFORE scaling (recommended)
    # Helps with: noisy sensors, measurement artifacts, high-frequency noise
    # Caution: Too much smoothing can blur important dynamics!
    SMOOTH_FEATURES = False  # Set to True to enable smoothing
    SMOOTH_WINDOW = 100       # Window size: 3-5 for light smoothing, 7-11 for heavy
    SMOOTH_METHOD = 'savgol'  # 'savgol' (best for peaks), 'gaussian', or 'moving_average'

    
    # Data trimming - cut off beginning/end of time series
    # Example: Skip first 100 samples and last 50 samples
    # START_IDX = 100
    # END_IDX = -50  # Negative indexing works too
    # Default: use all data
    START_IDX = None  # None = start from beginning
    END_IDX = None    # None = use until end
    
    # Load robot data
    print("Loading robot data...")
    DATA_FILE = "/home/mariano/phd_code/unicycle-network/reference10.csv.gz"
    df = load_robot_data(DATA_FILE)
    
    # Train classifier
    results = train_narma_classifier(
        df,
        narma_order=NARMA_ORDER,
        period_ratio=PERIOD_RATIO,
        amplitude=AMPLITUDE,
        washout_fraction=WASHOUT_FRACTION,
        train_fraction=TRAIN_FRACTION,
        ridge_alpha=RIDGE_ALPHA,
        feature_names=FEATURE_NAMES,
        start_idx=START_IDX,
        end_idx=END_IDX,
        use_local_frame=USE_LOCAL_FRAME,
        exclude_robots=EXCLUDE_ROBOTS,
        smooth_features_flag=SMOOTH_FEATURES,
        smooth_window=SMOOTH_WINDOW,
        smooth_method=SMOOTH_METHOD,
        verbose=True
    )
    
    # Create plots
    print("\nGenerating plots...")
    plot_all_robots_timeseries(df)
    plot_results(results, narma_order=NARMA_ORDER, save_dir=SCRIPT_DIR)
    plot_features(results, narma_order=NARMA_ORDER, save_dir=SCRIPT_DIR, max_samples=5000)
    plot_feature_scaling(results, narma_order=NARMA_ORDER, save_dir=SCRIPT_DIR, max_samples=2000)
    
    print("\n✓ Training complete!")
    
    # Tips for improving performance on amplitude peaks:
    # 1. Increase RIDGE_ALPHA (e.g., 1e-5 to 1e-3) for more regularization
    # 2. Add more training data or reduce WASHOUT_FRACTION
    # 3. Normalize/clip NARMA target to prevent extreme values
    # 4. Use weighted regression to emphasize peak regions
    # 5. Add velocity features (linear_x, omega) which may respond better to dynamics
    # 6. Try polynomial features or feature engineering (e.g., velocity^2)

# %%
