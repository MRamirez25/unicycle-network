from functools import partial
import torch
from torch import nn, optim
from tqdm import tqdm
import optuna
from unicycle_network_class import UnicycleReservoir
import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn import preprocessing
import traceback
from plot_robot_data import load_robot_data, extract_robot_ids, get_robot_states


def u(t, period_ratio=2, amplitude=0.2):
    """Input signal: sum of three sinusoids"""
    f1 = 2.11  # frequencies
    f2 = 3.73
    f3 = 4.33
    T = period_ratio
    u_val = amplitude * np.sin(2*np.pi * f1 * t / T) * np.sin(2*np.pi * f2 * t / T) * np.sin(2*np.pi * f3 * t / T)
    return u_val


def generate_narma(n_samples, narma_order=10, period_ratio=2, amplitude=0.2):
    """
    Generate NARMA-n time series.
    
    Args:
        n_samples: Number of time steps
        narma_order: Order of NARMA system (2, 3, or n >= 4)
        amplitude: Amplitude of input signal (will be normalized to [0, 0.5] for NARMA)
    
    Returns:
        u_input: Raw input signal from u(t) function
        u_normalized: Normalized input [0, 0.5] used for NARMA generation
        y_narma: NARMA output series
    """
    # Generate input signal using the sinusoidal function
    t = np.arange(n_samples)*0.01
    u_input = np.array([u(ti, period_ratio=period_ratio, amplitude=amplitude) for ti in t])
    
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
    
    # Return BOTH raw input AND normalized input (the one used for NARMA)
    return u_input, u_normalized, y_narma


def split_narma_data(u_input, y_narma, washout_fraction=0.1, train_fraction=0.7):
    """
    Split NARMA data into train/test sets with washout period.
    
    Args:
        u_input: Input signal
        y_narma: NARMA output
        washout_fraction: Fraction of data to discard as washout
        train_fraction: Fraction of remaining data to use for training
    
    Returns:
        train_u, train_y: Training data
        test_u, test_y: Test data
        washout_samples: Number of washout samples
    """
    n_samples = len(u_input)
    
    # Washout: discard initial transient samples
    washout_samples = int(washout_fraction * n_samples)
    valid_idx = np.arange(washout_samples, n_samples)
    
    # Split remaining data into train/test
    n_valid = len(valid_idx)
    n_train = int(train_fraction * n_valid)
    
    train_idx = valid_idx[:n_train]
    test_idx = valid_idx[n_train:]
    
    print(f'\n=== Data Split ===')
    print(f'Total samples: {n_samples}')
    print(f'Washout: {washout_samples} samples ({washout_fraction * 100:.1f}%)')
    print(f'Training: {len(train_idx)} samples ({train_fraction * 100:.1f}%)')
    print(f'Testing: {len(test_idx)} samples ({(1 - train_fraction) * 100:.1f}%)')
    
    # Extract train/test sets
    train_u = u_input[train_idx]
    train_y = y_narma[train_idx]
    test_u = u_input[test_idx]
    test_y = y_narma[test_idx]
    
    return train_u, train_y, test_u, test_y, washout_samples


# Single run objective function (for one random initialization)
def objective_single_run(trial, aligned_orientations=None, ang_input=None, ang_connections=None, 
              narma_order=10, n_samples=10000, seed=None, 
              robot_data_file=None, robot_data_start_idx=0, use_initial_distances=False,
              position_noise_std=0.0):
    """
    Run objective function with a specific random seed.
    
    Args:
        seed: Random seed for this run. If None, uses random initialization.
        robot_data_file: Path to robot data CSV file for initial positions
        robot_data_start_idx: Index in robot data to extract initial positions
        use_initial_distances: If True, set spring equilibrium distances from initial positions
        position_noise_std: Standard deviation of Gaussian noise added to positions (m)
    """
    # Set random seed if provided
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    # Suggest hyperparameters for Optuna to search
    aligned_orientations = trial.suggest_categorical("aligned_orientations", [True, False]) if aligned_orientations is None else aligned_orientations
    if not aligned_orientations:
        ang_input = trial.suggest_categorical("ang_input", [True, False]) if ang_input is None else ang_input
        ang_connections = trial.suggest_categorical("ang_connections", [True, False]) if ang_connections is None else ang_connections
    else:
        ang_input = False
        ang_connections = False

    # Position noise is now a fixed parameter (not optimized by Optuna)
    # It's passed as an argument to the function

    n_units = 10
    lin_stiff_min = trial.suggest_float('lin_stiff_min', 0.1, 10.0)  # Restored to MNIST ranges
    lin_stiff_max = trial.suggest_float('lin_stiff_max', lin_stiff_min, 20.0)  # Restored to MNIST ranges
    ang_stiff_min = trial.suggest_float('ang_stiff_min', 0.1, 5.0)  # Restored to MNIST ranges
    ang_stiff_max = trial.suggest_float('ang_stiff_max', ang_stiff_min, 10.0)  # Restored to MNIST ranges
    lin_damping_min = trial.suggest_float('lin_damping_min', 0.1, 2.0)
    lin_damping_max = trial.suggest_float('lin_damping_max', lin_damping_min, 10.0)
    ang_damping_min = trial.suggest_float('ang_damping_min', 0.1, 10.0)
    ang_damping_max = trial.suggest_float('ang_damping_max', ang_damping_min, 20.0)
    dt = 0.01#trial.suggest_float("dt", 0.001, 0.1, log=True)  # Increased range: larger dt = slower dynamics
    inp_bias = 0
    period_ratio = trial.suggest_int("period_ratio", 10, 100, step=10)  # Increased minimum: slower input variations
    
    # Input scaling for reservoir (separate from NARMA generation)
    reservoir_input_scale = trial.suggest_float("reservoir_input_scale", 1.0, 100.0)  # Log scale for better exploration
    
    # Ridge regression hyperparameter - extended range to allow weaker regularization
    ridge_alpha = trial.suggest_float("ridge_alpha", 1e-6, 1e2, log=True)
    
    # Data split parameters
    washout_fraction = trial.suggest_float("washout_fraction", 0.0, 0.1, step=0.05)
    train_fraction = 0.8  # Fixed: 70% train, 30% test
    
    # Washup period: let system equilibrate with zero input before data
    washup_steps = trial.suggest_int('washup_steps', 0, 2000, step=500)
    
    # Generate NARMA series with fixed amplitude of 0.5
    print(f'\n=== Generating NARMA-{narma_order} target series ===')
    print(f'NARMA amplitude: 0.5, Period ratio: {period_ratio}, Reservoir input scale: {reservoir_input_scale:.3f}')
    u_raw, u_norm, y_narma = generate_narma(n_samples, narma_order=narma_order, period_ratio=period_ratio, amplitude=0.5)
    
    # Split data
    train_u, train_y, test_u, test_y, washout_samples = split_narma_data(
        u_norm, y_narma, washout_fraction, train_fraction
    )
    
    # Input map parameters (1D input for NARMA)
    n_inp = 1
    non_zero_fraction = trial.suggest_float("non_zero_fraction", 0.3, 1.0, step=0.1)
    total_elements = n_inp * n_units
    num_non_zero = int(total_elements * non_zero_fraction)
    
    # Create sparse input maps
    lin_input_map = torch.zeros(n_inp, n_units)
    flat_indices = torch.randperm(total_elements)[:num_non_zero]
    row_indices = flat_indices // n_units
    col_indices = flat_indices % n_units
    
    # magnitude_min = trial.suggest_float("magnitude_min", -5.0, 0.0)
    magnitude_max = trial.suggest_float("magnitude_max", 1.0, 10.0)  # Increased like MNIST
    random_values = torch.rand(num_non_zero) * 2*(magnitude_max) - magnitude_max
    lin_input_map[row_indices, col_indices] = random_values

    # Angular input map
    ang_input_map = torch.zeros(n_inp, n_units)
    if ang_input:
        non_zero_fraction_ang = trial.suggest_float("non_zero_fraction_ang", 0.1, 1.0, step=0.1)
        num_non_zero_ang = int(total_elements * non_zero_fraction_ang)
        flat_indices_ang = torch.randperm(total_elements)[:num_non_zero_ang]
        row_indices_ang = flat_indices_ang // n_units
        col_indices_ang = flat_indices_ang % n_units
        
        magnitude_min_ang = trial.suggest_float("magnitude_min_ang", -20.0, 0.0)  # Restored to MNIST ranges
        magnitude_max_ang = trial.suggest_float("magnitude_max_ang", magnitude_min_ang, 20.0)  # Restored to MNIST ranges
        random_values_ang = torch.rand(num_non_zero_ang) * (magnitude_max_ang - magnitude_min_ang) + magnitude_min_ang
        ang_input_map[row_indices_ang, col_indices_ang] = random_values_ang

    # Network connectivity parameters
    n_connections_fraction = trial.suggest_float("n_connections_fraction", 0.2, 1.0, step=0.1)
    n_connections = int(n_units * n_connections_fraction)
    n_connections_anchor_fraction = trial.suggest_float("anchor_con_fraction", 0, 1.0, step=0.1)
    n_connections_anchor = int(n_connections_anchor_fraction * n_units)
    
    n_steps_readout = 0
    
    if not ang_connections:
        n_connections_ang = 0
        n_connections_anchor_ang = 0
    else:
        n_connections_ang_fraction = trial.suggest_float("n_connections_ang_fraction", 0.1, 1.0, step=0.1)
        n_connections_ang = int(n_units * n_connections_ang_fraction)
        n_connections_anchor_fraction_ang = trial.suggest_float("anchor_con_fraction_ang", 0, 0.8, step=0.1)
        n_connections_anchor_ang = int(n_connections_anchor_fraction_ang * n_units)

    eq_dist_min = trial.suggest_float("eq_dist_min", 0.01, 0.1)  # Restored to MNIST ranges
    eq_dist_max = trial.suggest_float("eq_dist_max", eq_dist_min, 0.3)  # Restored to MNIST ranges
    eq_dist_min_ang = trial.suggest_float("eq_dist_min_ang", -2*torch.pi, 0.0)
    eq_dist_max_ang = trial.suggest_float("eq_dist_max_ang", eq_dist_min_ang, 2*torch.pi)

    # Initialize the model
    model = UnicycleReservoir(
        n_inp=n_inp, n_units=n_units, dt=dt, n_out=1,  # Single output for NARMA prediction
        lin_stiff_min=lin_stiff_min, lin_stiff_max=lin_stiff_max,
        ang_stiff_min=ang_stiff_min, ang_stiff_max=ang_stiff_max,
        lin_damping_min=lin_damping_min, lin_damping_max=lin_damping_max,
        ang_damping_min=ang_damping_min, ang_damping_max=ang_damping_max,
        eq_dist_min=eq_dist_min, eq_dist_max=eq_dist_max, 
        eq_dist_min_ang=eq_dist_min_ang, eq_dist_max_ang=eq_dist_max_ang,
        n_connections=n_connections, inp_bias=inp_bias, lin_input_map=lin_input_map, 
        n_connections_anchor=n_connections_anchor, ang_input_map=ang_input_map,
        n_connections_ang=n_connections_ang, n_connections_anchor_ang=n_connections_anchor_ang, 
        n_past_steps_readout=n_steps_readout, use_capped_dynamics=True, max_speed=0.3, max_acceleration=3.0,
        position_noise_std=position_noise_std).to(device)

    # Set initial states (batch size = 1 for time series)
    batch_size = 1
    
    # Option 1: Use robot data for initial positions
    if robot_data_file is not None:
        print(f"\n=== Loading robot data for initial positions ===")
        print(f"File: {robot_data_file}")
        print(f"Start index: {robot_data_start_idx}")
        
        df = load_robot_data(robot_data_file)
        robot_ids = extract_robot_ids(df)
        
        # Use only first n_units robots
        if len(robot_ids) < n_units:
            raise ValueError(f"Need {n_units} robots but only {len(robot_ids)} found in data")
        robot_ids = robot_ids[:n_units]
        
        # Extract initial states from robot data
        x_positions = np.zeros(n_units)
        z_positions = np.zeros(n_units)
        linear_velocities = np.zeros(n_units)
        angular_velocities = np.zeros(n_units)
        orientations = np.zeros(n_units)
        
        for i, robot_id in enumerate(robot_ids):
            states = get_robot_states(df, robot_id)
            x_positions[i] = states['pos_x'][robot_data_start_idx]
            z_positions[i] = states['pos_y'][robot_data_start_idx]
            linear_velocities[i] = states['linear_x'][robot_data_start_idx]
            angular_velocities[i] = states['omega'][robot_data_start_idx]
            
            # Compute orientation from quaternion
            qz = states['qz'][robot_data_start_idx]
            qw = states['qw'][robot_data_start_idx]
            theta = 2 * np.arctan2(qz, qw)
            orientations[i] = theta
        
        # Set initial states from robot data
        model.set_init_states(batch_size, x_positions, z_positions, orientations, 
                             linear_velocities, angular_velocities)
        
        print(f"Loaded initial positions from robot data")
        print(f"  X range: [{x_positions.min():.3f}, {x_positions.max():.3f}]")
        print(f"  Z range: [{z_positions.min():.3f}, {z_positions.max():.3f}]")
        
        # Optionally set equilibrium distances based on initial positions
        if use_initial_distances:
            print(f"Setting spring equilibrium distances from initial positions...")
            model.set_eq_distances_from_initial_positions()
    
    # Option 2: Random initialization (default)
    else:
        model.set_init_states_random(batch_size)
    
    model.x_init = model.x_init.to(device)
    model.z_init = model.z_init.to(device)
    model.theta_init = model.theta_init.to(device)
    model.s_init = model.s_init.to(device)
    model.omega_init = model.omega_init.to(device)
    model.lin_input_map = model.lin_input_map.to(device)
    model.ang_input_map = model.ang_input_map.to(device)
    
    # Move network parameters to device
    model.unicycle_network.lin_damping = model.unicycle_network.lin_damping.to(device)
    model.unicycle_network.ang_damping = model.unicycle_network.ang_damping.to(device)
    model.unicycle_network.mass_vector = model.unicycle_network.mass_vector.to(device)
    model.unicycle_network.j_vector = model.unicycle_network.j_vector.to(device)

    # Configure initial states
    model.s_init[:,:] = 0
    model.omega_init[:,:] = 0
    if not aligned_orientations:
        model.theta_init[:,:] = torch.rand(model.theta_init.size()) * (4*torch.pi) - 2*torch.pi
        model.theta_init[:,0] = 0
    else:
        model.theta_init[:,:] = torch.rand(1) * (4*torch.pi) - 2*torch.pi

    # Washup period: Run model with zero input to let it settle near equilibrium
    if washup_steps > 0:
        print(f"\n=== Running washup period ({washup_steps} steps) ===")
        x = model.x_init.clone()
        z = model.z_init.clone()
        theta = model.theta_init.clone()
        s = model.s_init.clone()
        omega = model.omega_init.clone()
        
        u_washup = torch.zeros((1, washup_steps, 1), device=device)
        
        with torch.no_grad():
            for t in range(washup_steps):
                linear_input = (u_washup[:, t]) @ model.lin_input_map
                angular_input = (u_washup[:, t]) @ model.ang_input_map
                
                x, z, theta, s, omega = model.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)
        
        # Update model's initial states to the equilibrated states
        model.set_init_states(batch_size, x, z, theta, s, omega)
        print(f"Washup complete - State ranges: x=[{x.min():.3f}, {x.max():.3f}], s=[{s.min():.3f}, {s.max():.3f}]")

    # Extract reservoir activations for ALL samples (including washout)
    try:
        # CRITICAL: Use the SAME normalized input that was used to generate NARMA!
        # Apply reservoir_input_scale to the normalized [0, 0.5] input
        full_input = torch.from_numpy(u_raw * reservoir_input_scale).float().reshape(1, -1, 1).to(device)
        
        print(f"\n=== Extracting reservoir activations ===")
        print(f"Full input shape: {full_input.shape}")
        print(f"Input range after scaling: [{full_input.min():.4f}, {full_input.max():.4f}]")
        print(f"Washout samples: {washout_samples}")
        
        # Get reservoir activations (no gradients needed)
        with torch.no_grad():
            states_list, _, _ = model(full_input, full_input)
        
        # Process activations - concatenate all states from states_list
        # states_list is a list of length time_steps, each element is (batch=1, n_units*5)
        activations_all = torch.stack(states_list, dim=1)  # (1, time_steps, n_units*5)
        activations_all = activations_all.squeeze(0).cpu().numpy()  # (time_steps, n_units*5)
        
        # Extract velocities (s) for velocity constraint - they are in column indices [n_units*3:n_units*4]
        velocities_all = activations_all[:, n_units*3:n_units*4]  # (time_steps, n_units)
        
        # Velocity constraints (hard failures):
        # 1. Velocities should not exceed 0.3 m/s too much
        # 2. Velocities should reach at least 0.05 m/s (not all stuck near zero)
        velocity_magnitudes = np.abs(velocities_all)
        max_velocity = velocity_magnitudes.max()
        mean_velocity = velocity_magnitudes.mean()
        
        print(f"Velocity stats - mean: {mean_velocity:.6f}, std: {velocity_magnitudes.std():.6f}, max: {max_velocity:.6f}")
        
        # Check if velocities saturate at 0.3 too often (hitting the cap)
        # Define "saturated" as being very close to 0.3 (within 1%)
        saturation_threshold = 0.295  # 0.3 * 0.99
        saturated_mask = velocity_magnitudes >= saturation_threshold
        saturation_fraction = np.mean(saturated_mask)
        
        if saturation_fraction > 0.3:  # More than 30% of velocities saturated
            print(f"❌ FAILED: Velocities saturate at 0.3 too often - saturation fraction: {saturation_fraction:.3f} (>{0.3})")
            return float('inf')
        
        # Check if velocities never go above 0.05 (stuck/no movement)
        if max_velocity < 0.05:
            print(f"❌ FAILED: Velocities too low - max velocity: {max_velocity:.6f} < 0.05 m/s (no significant movement)")
            return float('inf')
        
        print(f"✓ Velocity constraints passed - max: {max_velocity:.3f} m/s, saturation fraction: {saturation_fraction:.3f}")
        
        activations_all = activations_all[:, :n_units*3]  # Use only position (x, y) and angle (theta) states
        
        print(f"All activations shape: {activations_all.shape}")
        print(f"All activations stats - mean: {activations_all.mean():.6f}, std: {activations_all.std():.6f}")
        # Check for NaN values
        if np.isnan(activations_all).any():
            print("NaN values detected in activations, ending trial")
            return float('inf'), float('inf'), float('inf')  # Return high error for failed trial
        
        # Check for zero variance (all activations identical)
        if activations_all.std() < 1e-10:
            print("WARNING: Activations have zero variance, reservoir not responding")
            return float('inf'), float('inf'), float('inf')
        
        # Check if reservoir dynamics die out towards the end
        # Compare activation changes in first half vs last half
        n_timesteps = len(activations_all)
        mid_point = n_timesteps // 2
        
        # Calculate mean absolute change per timestep for each half
        first_half_changes = np.mean(np.abs(np.diff(activations_all[:mid_point], axis=0)))
        last_half_changes = np.mean(np.abs(np.diff(activations_all[mid_point:], axis=0)))
        
        print(f"Activation dynamics - First half changes: {first_half_changes:.6f}, Last half changes: {last_half_changes:.6f}")
        
        # If last half has much less change than first half, dynamics are dying
        if first_half_changes > 1e-8:  # Only check if there was initial dynamics
            change_ratio = last_half_changes / first_half_changes
            if change_ratio < 0.1:
                print(f"❌ FAILED: Reservoir dynamics died out (last/first change ratio: {change_ratio:.4f})")
                return float('inf'), float('inf'), float('inf')
        
        # Also check if the last portion has near-zero changes (stuck/frozen states)
        last_quarter_start = 3 * n_timesteps // 4
        last_quarter_changes = np.mean(np.abs(np.diff(activations_all[last_quarter_start:], axis=0)))
        
        if last_quarter_changes < 1e-8:
            print(f"❌ FAILED: Reservoir frozen in last quarter (mean change: {last_quarter_changes:.2e})")
            return float('inf'), float('inf'), float('inf')
        
        # Split activations using same indices as targets (after washout)
        valid_idx = np.arange(washout_samples, n_samples)
        n_valid = len(valid_idx)
        n_train = int(train_fraction * n_valid)
        
        train_idx = valid_idx[:n_train]
        test_idx = valid_idx[n_train:]
        
        X_train = activations_all[train_idx, :]
        y_train_full = y_narma[train_idx]  # Keep as 1D array
        X_test = activations_all[test_idx, :]
        y_test_full = y_narma[test_idx]  # Keep as 1D array
        
        print(f"\n=== Train/Test Split ===")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train_full.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test_full.shape}")
        print(f"Train target stats - mean: {y_train_full.mean():.6f}, std: {y_train_full.std():.6f}")
        
        # Train Ridge regression on reservoir activations
        # Standard per-feature scaling (each column independently)
        from sklearn.preprocessing import StandardScaler
        
        X_train_features = X_train[:, :]
        X_test_features = X_test[:, :]
        
        # Fit scaler on training data and transform both train and test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
        X_test_scaled = scaler.transform(X_test_features)
        
        print(f"Feature scaling - each coordinate normalized independently")
        print(f"Scaled train stats - mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
        
        # Train Ridge regression with intercept (more standard approach)
        # StandardScaler already centered features, so intercept captures target mean
        regressor = Ridge(alpha=ridge_alpha, max_iter=1000, solver='lsqr', fit_intercept=True).fit(X_train_scaled, y_train_full)
        
        # Print regressor weights statistics
        weights = regressor.coef_.ravel()  # Flatten to 1D array
        intercept = regressor.intercept_
        # Handle both scalar and array intercepts
        intercept_val = intercept[0] if isinstance(intercept, np.ndarray) and intercept.size > 0 else float(intercept)
        print(f"Regressor weights - mean: {weights.mean():.6f}, std: {weights.std():.6f}, max: {weights.max():.6f}, min: {weights.min():.6f}")
        print(f"Regressor intercept: {intercept_val:.6f}")
        
        # Check if weights are all near zero (failed to learn)
        if np.abs(weights).max() < 1e-6:
            print("❌ FAILED: All regressor weights are essentially zero - no learning occurred")
            return float('inf'), float('inf'), float('inf')
        
        # Check if weights have low variance (all similar magnitude)
        weight_std = np.std(weights)
        if weight_std < 1e-6:
            print(f"❌ FAILED: Weights have no variance (std: {weight_std:.2e}) - degenerate solution")
            return float('inf'), float('inf'), float('inf')
        
        # Check if most weights are zero (sparse but potentially degenerate)
        non_zero_weights = np.sum(np.abs(weights) > 1e-6)
        weight_fraction = non_zero_weights / len(weights)
        if non_zero_weights < 5:
            print(f"❌ FAILED: Only {non_zero_weights} non-zero weights - degenerate solution")
            return float('inf'), float('inf'), float('inf')
        if weight_fraction < 0.1:
            print(f"❌ FAILED: Only {weight_fraction*100:.1f}% of weights are non-zero - too sparse")
            return float('inf'), float('inf'), float('inf')
        
        # Check if regressor learned anything (predictions vary)
        train_predictions = regressor.predict(X_train_scaled).ravel()  # Ensure 1D
        train_std = train_predictions.std()
        train_pred_range = train_predictions.max() - train_predictions.min()
        print(f"Train predictions stats - mean: {train_predictions.mean():.6f}, std: {train_std:.6f}, range: {train_pred_range:.6f}")
        print(f"Train predictions shape: {train_predictions.shape}")
        
        # Check variance relative to target
        target_std = y_train_full.std()
        pred_to_target_std_ratio = train_std / (target_std + 1e-9)
        
        if train_std < 1e-6:
            print("❌ FAILED: Regressor predicting constant (near-zero variance)")
            return float('inf'), float('inf'), float('inf')
        
        # Check if predictions are just constant (flat prediction)
        if train_pred_range < 1e-6:
            print(f"❌ FAILED: Predictions are flat (range: {train_pred_range:.2e})")
            return float('inf'), float('inf'), float('inf')
        
        # Check if predictions are too flat compared to target (variance ratio too small)
        if pred_to_target_std_ratio < 0.01:
            print(f"❌ FAILED: Predictions have {pred_to_target_std_ratio*100:.3f}% of target variance - too flat")
            return float('inf'), float('inf'), float('inf')
        
        # Predict on test set
        predictions = regressor.predict(X_test_scaled).ravel()  # Ensure 1D
        
        print(f"Test predictions shape: {predictions.shape}, Target shape: {y_test_full.shape}")
        
        # Check if test predictions are also flat
        test_pred_std = predictions.std()
        test_pred_range = predictions.max() - predictions.min()
        test_target_std = y_test_full.std()
        test_pred_to_target_std_ratio = test_pred_std / (test_target_std + 1e-9)
        print(f"Test predictions - std: {test_pred_std:.6f}, range: {test_pred_range:.6f}, std_ratio: {test_pred_to_target_std_ratio:.4f}")
        
        if test_pred_std < 1e-6:
            print(f"❌ FAILED: Test predictions have near-zero variance (std: {test_pred_std:.2e})")
            return float('inf')
        
        if test_pred_range < 1e-6:
            print(f"❌ FAILED: Test predictions are flat (range: {test_pred_range:.2e})")
            return float('inf')
        
        # Check if test predictions are too flat compared to target
        if test_pred_to_target_std_ratio < 0.01:
            print(f"❌ FAILED: Test predictions have {test_pred_to_target_std_ratio*100:.3f}% of target variance - too flat")
            return float('inf')
        
        # Calculate NMSE (Normalized Mean Squared Error)
        mse = np.mean(np.square(predictions - y_test_full))
        target_var = np.var(y_test_full)
        nmse = mse / (target_var + 1e-9)
        
        # Calculate NRMSE
        rmse = np.sqrt(mse)
        target_std = np.std(y_test_full)
        nrmse = rmse / (target_std + 1e-9)
        
        print(f"Test MSE: {mse:.6f}, NMSE: {nmse:.6f}, NRMSE: {nrmse:.6f}")
        
        # Check if NMSE is too high (worse than basic prediction)
        if nmse > 0.8:
            print(f"❌ FAILED: NMSE too high ({nmse:.4f}) - very poor prediction")
            return float('inf')
        
        # Check if NMSE is close to 1.0 (predicting mean or near-mean)
        # NMSE = 1.0 means MSE = target variance, which happens when predicting the mean
        if 0.95 <= nmse <= 1.05:
            print(f"❌ FAILED: NMSE ≈ 1.0 ({nmse:.4f}) - model is essentially predicting the mean")
            return float('inf')
        
        if nmse > 10:
            print(f"⚠️  WARNING: High NMSE ({nmse:.2f}) - poor configuration")
        
        # Return single objective: NMSE (lower is better)
        # Velocity and oscillation constraints are now hard failures (handled above)
        print(f"\n{'='*60}")
        print(f"Trial objective - NMSE: {nmse:.6f}")
        print(f"{'='*60}")
        return nmse
        
    except Exception as e:
        print(f"Error in trial: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return float('inf')  # Return high error for failed trials


def objective(trial, aligned_orientations=None, ang_input=None, ang_connections=None, 
              narma_order=10, n_samples=10000, n_runs=3,
              robot_data_file=None, robot_data_start_idx=0, use_initial_distances=False,
              position_noise_std=0.0):
    """
    Wrapper objective function that runs multiple times and returns mean score.
    
    Args:
        n_runs: Number of times to run with different random seeds (default: 3)
        robot_data_file: Path to robot data CSV file for initial positions
        robot_data_start_idx: Index in robot data to extract initial positions
        use_initial_distances: If True, set spring equilibrium distances from initial positions
        position_noise_std: Standard deviation of Gaussian noise added to positions (m)
    
    Returns:
        Mean NMSE across all runs
    """
    print(f"\n{'#'*80}")
    print(f"# Running trial {trial.number} with {n_runs} independent runs")
    print(f"{'#'*80}\n")
    
    results = []
    failed_runs = 0
    
    for run_idx in range(n_runs):
        seed = trial.number * 1000 + run_idx  # Unique seed per run
        print(f"\n{'='*70}")
        print(f"RUN {run_idx + 1}/{n_runs} (seed={seed})")
        print(f"{'='*70}")
        
        nmse = objective_single_run(
            trial, 
            aligned_orientations=aligned_orientations,
            ang_input=ang_input, 
            ang_connections=ang_connections,
            narma_order=narma_order,
            n_samples=n_samples,
            seed=seed,
            robot_data_file=robot_data_file,
            robot_data_start_idx=robot_data_start_idx,
            use_initial_distances=use_initial_distances,
            position_noise_std=position_noise_std
        )
        
        # Check if run failed
        if np.isinf(nmse):
            failed_runs += 1
            print(f"❌ Run {run_idx + 1} FAILED")
        else:
            results.append(nmse)
            print(f"✓ Run {run_idx + 1} completed: NMSE={nmse:.6f}")
    
    # If all runs failed, return inf
    if len(results) == 0:
        print(f"\n{'!'*80}")
        print(f"! ALL {n_runs} RUNS FAILED - returning infinite penalty")
        print(f"{'!'*80}\n")
        return float('inf')
    
    # Calculate mean across successful runs
    mean_nmse = np.mean(results)
    
    # Calculate std for reporting
    std_nmse = np.std(results) if len(results) > 1 else 0.0
    
    print(f"\n{'='*80}")
    print(f"TRIAL {trial.number} SUMMARY ({len(results)}/{n_runs} successful runs)")
    print(f"{'='*80}")
    print(f"NMSE: {mean_nmse:.6f} ± {std_nmse:.6f}")
    print(f"{'='*80}\n")
    
    return mean_nmse


# Define the search space and start the optimization
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Optuna optimization for NARMA prediction with unicycle reservoir')
    parser.add_argument('--narma-order', type=int, default=2, help='NARMA order (2, 3, or higher)')
    parser.add_argument('--n-samples', type=int, default=10000, help='Number of time samples')
    parser.add_argument('--n-runs', type=int, default=3, help='Number of runs per trial (for robustness)')
    parser.add_argument('--timeout', type=int, default=3600*8, help='Optimization timeout in seconds')
    parser.add_argument('--robot-data', type=str, default='/home/mariano/phd_code/unicycle-network/reference10.csv.gz', 
                       help='Path to robot data CSV file for initial positions')
    parser.add_argument('--robot-start-idx', type=int, default=100,
                       help='Index in robot data to extract initial positions')
    parser.add_argument('--use-initial-distances', action='store_true',
                       help='Set spring equilibrium distances from initial robot positions')
    parser.add_argument('--position-noise-std', type=float, default=0.0,
                       help='Standard deviation of Gaussian noise added to positions (m)')
    
    args = parser.parse_args()
    
    # Define the GPU or CPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # NARMA configuration
    narma_order = args.narma_order
    n_samples = args.n_samples
    n_runs_per_trial = args.n_runs
    
    print(f"\n{'='*80}")
    print(f"CONFIGURATION:")
    print(f"  NARMA order: {narma_order}")
    print(f"  Samples: {n_samples}")
    print(f"  Runs per trial: {n_runs_per_trial} (averaging results across random seeds)")
    print(f"  Position noise std: {args.position_noise_std} m")
    if args.robot_data:
        print(f"  Robot data: {args.robot_data}")
        print(f"  Robot start index: {args.robot_start_idx}")
        print(f"  Use initial distances: {args.use_initial_distances}")
    else:
        print(f"  Initial positions: Random")
    print(f"{'='*80}\n")
    
    database_name = "unicycle_nets_narma_real_data_init"
    study_name = f"narma{narma_order}_noise_{args.position_noise_std}"
    storage_name = "sqlite:///optuna_databases/{}.db".format(database_name)
    study = optuna.create_study(
        storage=storage_name, 
        study_name=study_name, 
        direction='minimize',  # Single-objective: minimize NMSE
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(
        partial(objective, aligned_orientations=False, ang_input=False, ang_connections=False,
                narma_order=narma_order, n_samples=n_samples, n_runs=n_runs_per_trial,
                robot_data_file=args.robot_data, robot_data_start_idx=args.robot_start_idx,
                use_initial_distances=args.use_initial_distances,
                position_noise_std=args.position_noise_std), 
        timeout=args.timeout
    )

    # Get the best solution (single-objective)
    print(f"\n{'='*60}")
    print(f"Single-objective optimization complete!")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"{'='*60}")
    
    # Print best trial
    print(f"\nBest solution:")
    print(f"  NMSE: {study.best_value:.6f}")
    print(f"  Trial number: {study.best_trial.number}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    with open(f'narma{narma_order}_unicycle_single_objective_results.txt', 'w') as f:
        f.write(f"NARMA-{narma_order} Single-Objective Prediction Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Configuration:\n")
        f.write(f"  Runs per trial: {n_runs_per_trial} (scores averaged across random seeds)\n")
        f.write(f"  Samples: {n_samples}\n")
        f.write(f"Objective:\n")
        f.write(f"  Minimize NMSE\n")
        f.write(f"Constraints:\n")
        f.write(f"  Velocity max: 0.3 m/s (mean excess < 0.1)\n")
        f.write(f"  Velocity min: 0.05 m/s (max velocity must exceed this)\n")
        f.write(f"\nBest solution:\n")
        f.write(f"  Trial: {study.best_trial.number}\n")
        f.write(f"  NMSE: {study.best_value:.6f}\n")
        f.write(f"\nBest hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
