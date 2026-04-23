"""
Sweep period_ratio parameter and measure NMSE performance.
Minimal script to test how input signal frequency affects NARMA prediction.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import optuna

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from unicycle_network_class import UnicycleReservoir


def u(t, period_ratio=2, amplitude=0.2):
    """Input signal: sum of three sinusoids"""
    f1 = 2.11
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
        period_ratio: Period scaling for input signal
        amplitude: Amplitude of input signal (will be normalized to [0, 0.5] for NARMA)
    
    Returns:
        u_input: Raw input signal from u(t) function
        u_normalized: Normalized input [0, 0.5] used for NARMA generation
        y_narma: NARMA output series
    """
    t = np.arange(n_samples) * 0.01  # Time vector with dt=0.01
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


def evaluate_period_ratio(period_ratio, study_name, database_name, narma_order=2, 
                          n_samples=10000, device='cpu', verbose=False, seed=40):
    """
    Evaluate model performance for a specific period_ratio.
    
    Args:
        seed: Random seed for initialization (default: 40)
    
    Returns:
        nmse: Normalized Mean Squared Error on test set
    """
    # Load best trial from study
    storage_name = f"sqlite:///{parent_dir}/optuna_databases/{database_name}.db"
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    
    # Get best trial (lowest NMSE)
    best_trial = study.best_trials[10]#min(study.best_trials, key=lambda t: t.values[0])
    params = best_trial.params
    
    if verbose:
        print(f"\nPeriod ratio: {period_ratio}, Seed: {seed}")
        print(f"Using trial {best_trial.number} (NMSE: {best_trial.values[0]:.6f})")
    
    # Extract hyperparameters
    n_units = 10
    dt = 0.01
    lin_stiff_min = params['lin_stiff_min']
    lin_stiff_max = params['lin_stiff_max']
    ang_stiff_min = params['ang_stiff_min']
    ang_stiff_max = params['ang_stiff_max']
    lin_damping_min = params['lin_damping_min']
    lin_damping_max = params['lin_damping_max']
    ang_damping_min = params['ang_damping_min']
    ang_damping_max = params['ang_damping_max']
    eq_dist_min = params['eq_dist_min']
    eq_dist_max = params['eq_dist_max']
    eq_dist_min_ang = params['eq_dist_min_ang']
    eq_dist_max_ang = params['eq_dist_max_ang']
    ridge_alpha = params.get('ridge_alpha', 1e-6)
    washout_fraction = params['washout_fraction']
    reservoir_input_scale = params.get('reservoir_input_scale', 1.0)
    washup_steps = params.get('washup_steps', 0)
    
    # Network connectivity
    aligned_orientations = False#params['aligned_orientations']
    ang_input = params.get('ang_input', True)
    ang_connections = params.get('ang_connections', True)
    n_connections_fraction = params['n_connections_fraction']
    n_connections = int(n_units * n_connections_fraction)
    anchor_con_fraction = params['anchor_con_fraction']
    n_connections_anchor = int(n_units * anchor_con_fraction)
    
    if ang_connections:
        n_connections_ang_fraction = params["n_connections_ang_fraction"]
        n_connections_ang = int(n_connections_ang_fraction * n_units)
        anchor_con_fraction_ang = params['anchor_con_fraction_ang']
        n_connections_anchor_ang = int(anchor_con_fraction_ang * n_units)
    else:
        n_connections_ang = 0
        n_connections_anchor_ang = 0
    
    # Set random seed for reproducibility (same as narma_ang_input_coupled.py)
    # This ensures input maps and network parameters are consistent
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Input maps
    n_inp = 1
    non_zero_fraction = params['non_zero_fraction']
    magnitude_max = params['magnitude_max']
    
    total_elements = n_inp * n_units
    num_non_zero = int(total_elements * non_zero_fraction)
    
    lin_input_map = torch.zeros(n_inp, n_units)
    flat_indices = torch.randperm(total_elements)[:num_non_zero]
    row_indices = flat_indices // n_units
    col_indices = flat_indices % n_units
    
    # Get magnitude_min from params or calculate from magnitude_max
    magnitude_min_val = 0.0  # Default
    non_zero_weights = lin_input_map[lin_input_map != 0]
    if len(non_zero_weights) > 0:
        magnitude_min_val = non_zero_weights.min().item()
    
    random_values = torch.rand(num_non_zero) * 2 * (magnitude_max) - magnitude_max
    lin_input_map[row_indices, col_indices] = random_values
    
    ang_input_map = torch.zeros(n_inp, n_units)
    if ang_input:
        non_zero_fraction_ang = params['non_zero_fraction_ang']
        magnitude_min_ang = params['magnitude_min_ang']
        magnitude_max_ang = params['magnitude_max_ang']
        num_non_zero_ang = int(total_elements * non_zero_fraction_ang)
        flat_indices_ang = torch.randperm(total_elements)[:num_non_zero_ang]
        row_indices_ang = flat_indices_ang // n_units
        col_indices_ang = flat_indices_ang % n_units
        random_values_ang = torch.rand(num_non_zero_ang) * (magnitude_max_ang - magnitude_min_ang) + magnitude_min_ang
        ang_input_map[row_indices_ang, col_indices_ang] = random_values_ang
    
    # Generate NARMA data with this period_ratio
    u_raw, u_norm, y_narma = generate_narma(n_samples, narma_order=narma_order, 
                                            period_ratio=period_ratio, amplitude=0.5)
    
    # Create model
    model = UnicycleReservoir(
        n_inp=n_inp, n_units=n_units, dt=dt, n_out=1,
        lin_stiff_min=lin_stiff_min, lin_stiff_max=lin_stiff_max,
        ang_stiff_min=ang_stiff_min, ang_stiff_max=ang_stiff_max,
        lin_damping_min=lin_damping_min, lin_damping_max=lin_damping_max,
        ang_damping_min=ang_damping_min, ang_damping_max=ang_damping_max,
        eq_dist_min=eq_dist_min, eq_dist_max=eq_dist_max,
        eq_dist_min_ang=eq_dist_min_ang, eq_dist_max_ang=eq_dist_max_ang,
        n_connections=n_connections, inp_bias=0, lin_input_map=lin_input_map,
        n_connections_anchor=n_connections_anchor, ang_input_map=ang_input_map,
        n_connections_ang=n_connections_ang, n_connections_anchor_ang=n_connections_anchor_ang,
        n_past_steps_readout=0, use_capped_dynamics=True, max_speed=0.2, max_acceleration=2.0
    ).to(device)
    
    # Initialize states
    batch_size = 1
    model.set_init_states_random(batch_size)
    
    # Move all model components to device
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
    
    # Set initial velocities
    model.s_init[:, :] = 0
    model.omega_init[:, :] = 0
    
    # Set initial orientations (after moving to device to ensure correct device)
    if not aligned_orientations:
        model.theta_init[:, :] = torch.rand(model.theta_init.size(), device=device) * (4*torch.pi) - 2*torch.pi
        model.theta_init[:, 0] = 0
    else:
        model.theta_init[:, :] = torch.rand(1, device=device) * (4*torch.pi) - 2*torch.pi
    
    # Washup period
    if washup_steps > 0:
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
        
        model.set_init_states(batch_size, x, z, theta, s, omega)
    
    # Get reservoir activations
    full_input = torch.from_numpy(u_raw * reservoir_input_scale).float().reshape(1, -1, 1).to(device)
    
    with torch.no_grad():
        states_list, _, _ = model(full_input, full_input)
    
    activations_all = torch.stack(states_list, dim=1).squeeze(0).cpu().numpy()
    
    # Split data
    washout_samples = int(washout_fraction * n_samples)
    valid_idx = np.arange(washout_samples, n_samples)
    n_valid = len(valid_idx)
    train_fraction = 0.7
    n_train = int(train_fraction * n_valid)
    
    train_idx = valid_idx[:n_train]
    test_idx = valid_idx[n_train:]
    
    X_train = activations_all[train_idx, :3*n_units]  # Use only x, z, theta
    y_train = y_narma[train_idx]
    X_test = activations_all[test_idx, :3*n_units]
    y_test = y_narma[test_idx]
    
    # Train Ridge regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    regressor = Ridge(alpha=ridge_alpha, max_iter=1000, solver='lsqr', fit_intercept=True)
    regressor.fit(X_train_scaled, y_train)
    
    # Predict and calculate NMSE
    predictions = regressor.predict(X_test_scaled)
    mse = np.mean(np.square(predictions - y_test))
    target_var = np.var(y_test)
    nmse = mse / (target_var + 1e-9)
    
    if verbose:
        print(f"  NMSE: {nmse:.6f}")
    
    return nmse


def sweep_period_ratios(period_ratios, study_name, database_name, narma_order=2,
                       n_samples=10000, output_dir=None, verbose=True, seeds=None):
    """
    Sweep through different period_ratio values and measure NMSE.
    
    Args:
        period_ratios: List or array of period_ratio values to test
        study_name: Name of Optuna study to load parameters from
        database_name: Name of database file (without .db extension)
        narma_order: Order of NARMA system
        n_samples: Number of samples to generate
        output_dir: Directory to save results (default: parent_dir/plots)
        verbose: Print progress
        seeds: List of random seeds to run (default: [40]). Use multiple seeds for error bars.
    
    Returns:
        period_ratios: Array of tested period ratios
        nmse_mean: Array of mean NMSE values across seeds
        nmse_std: Array of NMSE standard deviations across seeds
        nmse_all: 2D array of shape (n_period_ratios, n_seeds) with all individual results
    """
    if output_dir is None:
        output_dir = f"{parent_dir}/plots"
    
    if seeds is None:
        seeds = [40]  # Default single seed
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Device: {device}")
        print(f"Testing {len(period_ratios)} period_ratio values: {period_ratios}")
        print(f"Using {len(seeds)} random seeds: {seeds}")
        print(f"Study: {study_name} from {database_name}.db")
        print("="*60)
    
    # Store results for all seeds
    nmse_all = np.zeros((len(period_ratios), len(seeds)))
    nmse_all[:] = np.nan  # Initialize with NaN
    
    for i, pr in enumerate(period_ratios):
        if verbose:
            print(f"\n[{i+1}/{len(period_ratios)}] Testing period_ratio = {pr}")
        
        for j, seed in enumerate(seeds):
            if verbose and len(seeds) > 1:
                print(f"  Seed {j+1}/{len(seeds)}: {seed}")
            
            try:
                nmse = evaluate_period_ratio(
                    pr, study_name, database_name, 
                    narma_order=narma_order,
                    n_samples=n_samples,
                    device=device,
                    verbose=verbose and len(seeds) == 1,  # Only verbose if single seed
                    seed=seed
                )
                nmse_all[i, j] = nmse
                
                if verbose and len(seeds) > 1:
                    print(f"    NMSE: {nmse:.6f}")
                    
            except Exception as e:
                print(f"  ERROR (seed {seed}): {e}")
                nmse_all[i, j] = np.nan
        
        # Print summary for this period_ratio if using multiple seeds
        if verbose and len(seeds) > 1:
            valid_nmse = nmse_all[i, ~np.isnan(nmse_all[i, :])]
            if len(valid_nmse) > 0:
                print(f"  Summary: {valid_nmse.mean():.6f} ± {valid_nmse.std():.6f} ({len(valid_nmse)}/{len(seeds)} successful)")
    
    # Calculate statistics across seeds
    nmse_mean = np.nanmean(nmse_all, axis=1)
    nmse_std = np.nanstd(nmse_all, axis=1)
    
    # Save results
    results_file = f"{output_dir}/narma{narma_order}_period_ratio_sweep.csv"
    with open(results_file, 'w') as f:
        # Header
        f.write("period_ratio,nmse_mean,nmse_std")
        for j, seed in enumerate(seeds):
            f.write(f",nmse_seed{seed}")
        f.write("\n")
        
        # Data
        for i, pr in enumerate(period_ratios):
            f.write(f"{pr},{nmse_mean[i]},{nmse_std[i]}")
            for j in range(len(seeds)):
                f.write(f",{nmse_all[i, j]}")
            f.write("\n")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Results saved to: {results_file}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Remove NaN values for plotting
    valid_mask = ~np.isnan(nmse_mean)
    valid_pr = np.array(period_ratios)[valid_mask]
    valid_nmse_mean = nmse_mean[valid_mask]
    valid_nmse_std = nmse_std[valid_mask]
    
    if len(seeds) > 1:
        # Plot with error bars
        ax.errorbar(valid_pr, valid_nmse_mean, yerr=valid_nmse_std, 
                   fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
                   label=f'NMSE (mean ± std, n={len(seeds)} seeds)')
    else:
        # Plot without error bars
        ax.plot(valid_pr, valid_nmse_mean, 'o-', linewidth=2, markersize=8, label='NMSE')
    
    ax.set_xlabel('Period Ratio', fontsize=12)
    ax.set_ylabel('NMSE (Normalized Mean Squared Error)', fontsize=12)
    
    title = f'NARMA-{narma_order} Performance vs Input Signal Period'
    if len(seeds) > 1:
        title += f' ({len(seeds)} seeds)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Annotate best performance
    if len(valid_nmse_mean) > 0:
        best_idx = np.argmin(valid_nmse_mean)
        best_pr = valid_pr[best_idx]
        best_nmse = valid_nmse_mean[best_idx]
        best_std = valid_nmse_std[best_idx]
        
        if len(seeds) > 1:
            label = f'Best: PR={best_pr}, NMSE={best_nmse:.4f}±{best_std:.4f}'
        else:
            label = f'Best: PR={best_pr}, NMSE={best_nmse:.4f}'
        
        ax.plot(best_pr, best_nmse, 'r*', markersize=15, label=label)
        ax.legend()
    
    plt.tight_layout()
    plot_file = f"{output_dir}/narma{narma_order}_period_ratio_sweep.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"Plot saved to: {plot_file}")
    plt.close()
    
    return period_ratios, nmse_mean, nmse_std, nmse_all


if __name__ == '__main__':
    # Configuration
    N_SAMPLES = 10000
    
    # Study to load parameters from
    DATABASE_NAME = "unicycle_nets_narma_fixed_dt_higher_period_ratio"
    
    # Period ratios to test
    period_ratios = [15, 20, 25, 30, 35, 40]
    
    # Random seeds for statistical robustness
    seeds = [40, 41, 42, 111, 345]
    
    # NARMA orders to test
    narma_orders = [2, 5]
    
    # Store results for comparison plot
    results = {}
    
    # Run sweep for each NARMA order
    for narma_order in narma_orders:
        study_name = f"narma{narma_order}_multiobjective_velocity_oscillation_more_stiffness"
        
        print(f"\n{'='*60}")
        print(f"NARMA-{narma_order} Period Ratio Sweep")
        print(f"{'='*60}\n")
        
        pr, nmse_mean, nmse_std, nmse_all = sweep_period_ratios(
            period_ratios=period_ratios,
            study_name=study_name,
            database_name=DATABASE_NAME,
            narma_order=narma_order,
            n_samples=N_SAMPLES,
            seeds=seeds,
            verbose=True
        )
        
        results[narma_order] = {
            'period_ratios': pr,
            'nmse_mean': nmse_mean,
            'nmse_std': nmse_std,
            'nmse_all': nmse_all
        }
        
        # Print summary for this NARMA order
        print(f"\n{'='*60}")
        print(f"NARMA-{narma_order} SUMMARY")
        print(f"{'='*60}")
        print(f"{'Period Ratio':>12} | {'NMSE Mean':>12} | {'NMSE Std':>10} | {'N Seeds':>8}")
        print("-"*60)
        for i, pr_val in enumerate(pr):
            valid_seeds = np.sum(~np.isnan(nmse_all[i, :]))
            if np.isnan(nmse_mean[i]):
                print(f"{pr_val:12.0f} | {'FAILED':>12} | {'-':>10} | {valid_seeds:>8}/{len(seeds)}")
            else:
                print(f"{pr_val:12.0f} | {nmse_mean[i]:12.6f} | {nmse_std[i]:10.6f} | {valid_seeds:>8}/{len(seeds)}")
        
        valid_mask = ~np.isnan(nmse_mean)
        if np.sum(valid_mask) > 0:
            best_idx = np.nanargmin(nmse_mean)
            print(f"\nBest: period_ratio={pr[best_idx]}, NMSE={nmse_mean[best_idx]:.6f}±{nmse_std[best_idx]:.6f}")
    
    # Create comparison plot with both NARMA-2 and NARMA-5
    print(f"\n{'='*60}")
    print("Creating comparison plot...")
    print(f"{'='*60}\n")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue for NARMA-2, Orange for NARMA-5
    markers = ['o', 's']  # Circle for NARMA-2, Square for NARMA-5
    
    for idx, narma_order in enumerate(narma_orders):
        res = results[narma_order]
        pr = res['period_ratios']
        nmse_mean = res['nmse_mean']
        nmse_std = res['nmse_std']
        
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(nmse_mean)
        valid_pr = np.array(pr)[valid_mask]
        valid_nmse_mean = nmse_mean[valid_mask]
        valid_nmse_std = nmse_std[valid_mask]
        
        if len(seeds) > 1:
            # Plot with error bars
            ax.errorbar(valid_pr, valid_nmse_mean, yerr=valid_nmse_std,
                       fmt=markers[idx] + '-', linewidth=2.5, markersize=10,
                       capsize=5, capthick=2, alpha=0.8,
                       color=colors[idx],
                       label=f'NARMA-{narma_order} (n={len(seeds)} seeds)')
        else:
            # Plot without error bars
            ax.plot(valid_pr, valid_nmse_mean, markers[idx] + '-',
                   linewidth=2.5, markersize=10, alpha=0.8,
                   color=colors[idx],
                   label=f'NARMA-{narma_order}')
    
    ax.set_xlabel('Period Ratio', fontsize=14)
    ax.set_ylabel('NMSE (Normalized Mean Squared Error)', fontsize=14)
    
    title = 'NARMA Performance Comparison vs Input Signal Period'
    if len(seeds) > 1:
        title += f' ({len(seeds)} seeds)'
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    output_dir = f"{parent_dir}/plots"
    comparison_file = f"{output_dir}/narma_comparison_period_ratio_sweep.png"
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {comparison_file}")
    plt.close()
    
    print("\n" + "="*60)
    print("All sweeps complete!")
    print("="*60)

