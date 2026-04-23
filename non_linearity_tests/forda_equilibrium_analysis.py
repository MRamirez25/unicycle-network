#%%
"""
Analyze final reservoir states in relation to equilibria.

This script:
1. Loads a trained FordA model configuration from Optuna
2. Runs the reservoir on FordA data
3. Takes the final states for samples of each class
4. Uses those final (x, y) positions and thetas to find nearby equilibria
5. Analyzes if different classes tend to be near different equilibria
"""

import os
import sys
import numpy as np
import torch
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import optuna

#%%
# Add the parent directory to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import get_FordA_data
from unicycle_network_class import UnicycleNetwork, UnicycleReservoir

#%%
# ============================================================
# CONFIGURATION
# ============================================================

# Optuna study configuration
DATABASE_NAME = "unicycle_nets_forda_logreg"
STUDY_NAME = "only_last_state_as_readout_20_units_not_aligned_less_ang_magnitude_no_tanh"

# Seed for reproducibility
SEED = 33

# Equilibrium finding parameters
N_EQUILIBRIUM_TRIALS = 2000  # Random restarts to find equilibria
EQUILIBRIUM_TOL = 1e-6  # Tolerance for equilibrium
UNIQUE_TOL = 1e-3  # Tolerance for considering equilibria as same

# Data source: "train" or "test"
USE_DATA_SOURCE = "train"  # Change to "test" to use test set samples

#%%
def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#%%
def load_optuna_params(database_name, study_name):
    """Load best parameters from Optuna study."""
    storage_name = f"sqlite:///{parent_dir}/optuna_databases/{database_name}.db"
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    return study.best_params, study.best_value

#%%
# ============================================================
# EQUILIBRIUM EQUATIONS
# ============================================================

def make_equilibrium_equations(K_mat, A_mat, e, p0, N):
    """
    Create equilibrium equation function for the unicycle network.
    
    The equilibrium is found by solving F(s) = 0, where s are the 
    line parameters (positions along each unicycle's heading direction).
    
    Args:
        K_mat: (N, N) stiffness coupling matrix
        A_mat: (N, N) equilibrium distances matrix
        e: (N, 2) unit heading vectors
        p0: (N, 2) reference positions
        N: number of units
    
    Returns:
        F: equilibrium equation function
        positions: function to compute positions from s
    """
    def positions(s):
        """Compute positions from line parameters s."""
        s_full = np.zeros(N)
        s_full[1:] = s  # First unicycle fixed at s=0
        return p0 + s_full[:, None] * e
    
    def F(s):
        """Equilibrium equations: F(s) = 0 at equilibrium."""
        p = positions(s)
        d = p[:, None, :] - p[None, :, :]  # (N, N, 2)
        r = np.linalg.norm(d, axis=2) + np.eye(N)  # Avoid div by zero
        F_contrib = K_mat * (A_mat - r) * np.einsum('ijk,ik->ij', d, e) / r
        F_vec = np.sum(F_contrib, axis=1)
        return F_vec[1:]  # Drop first unicycle (fixed)
    
    return F, positions


def find_equilibria_from_state(K_mat, A_mat, theta, p_init, N, n_trials=100, tol=1e-6, unique_tol=1e-3):
    """
    Find equilibria starting from a given state.
    
    Args:
        K_mat: Stiffness matrix
        A_mat: Equilibrium distances matrix
        theta: (N,) heading angles from the final state
        p_init: (N, 2) positions from the final state (used as p0)
        N: number of units
        n_trials: number of random restarts around the initial guess
        tol: tolerance for equilibrium
        unique_tol: tolerance for deduplication
    
    Returns:
        List of unique equilibria (as s vectors)
    """
    # Use the final theta to define heading directions
    e = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    
    # Use the final position as reference point p0
    p0 = p_init.copy()
    
    # Create equilibrium equations
    F, positions = make_equilibrium_equations(K_mat, A_mat, e, p0, N)
    
    solutions = []
    
    # Initial guess: s = 0 (positions at p0)
    s0 = np.zeros(N - 1)
    
    # Try from the exact initial state
    try:
        sol = root(F, s0, method='hybr')
        if sol.success and np.linalg.norm(sol.fun) < tol:
            solutions.append(sol.x)
    except:
        pass
    
    # Try random perturbations
    for _ in range(n_trials):
        s_init = np.random.randn(N - 1) * 0.5  # Small perturbations
        try:
            sol = root(F, s_init, method='hybr')
            if sol.success and np.linalg.norm(sol.fun) < tol:
                solutions.append(sol.x)
        except:
            continue
    
    # Also try larger range random restarts
    for _ in range(n_trials // 2):
        s_init = np.random.randn(N - 1) * 5  # Larger perturbations
        try:
            sol = root(F, s_init, method='hybr')
            if sol.success and np.linalg.norm(sol.fun) < tol:
                solutions.append(sol.x)
        except:
            continue
    
    if len(solutions) == 0:
        return [], F, positions
    
    # Remove duplicates
    unique = []
    for s in solutions:
        if not any(np.linalg.norm(s - u) < unique_tol for u in unique):
            unique.append(s)
    
    return unique, F, positions


def distance_to_equilibria(p_state, equilibria, positions_func):
    """
    Compute distance from a state to each equilibrium.
    
    Args:
        p_state: (N, 2) current positions
        equilibria: list of s vectors for equilibria
        positions_func: function to convert s to positions
    
    Returns:
        distances: array of distances to each equilibrium
    """
    if len(equilibria) == 0:
        return np.array([np.inf])
    
    distances = []
    for s_eq in equilibria:
        p_eq = positions_func(s_eq)
        dist = np.linalg.norm(p_state[1:] - p_eq[1:])  # Ignore fixed first unit
        distances.append(dist)
    
    return np.array(distances)

#%%
def main():
    print("="*70)
    print(f"EQUILIBRIUM ANALYSIS FOR FordA RESERVOIR STATES ({USE_DATA_SOURCE.upper()} DATA)")
    print("="*70)
    
    set_seed(SEED)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load Optuna parameters
    print(f"\nLoading parameters from: {DATABASE_NAME}/{STUDY_NAME}")
    params, best_value = load_optuna_params(DATABASE_NAME, STUDY_NAME)
    print(f"Best validation accuracy: {best_value:.4f}")
    
    # Extract parameters
    n_units = params.get('n_units', 20)
    aligned_orientations = params.get('aligned_orientations', False)
    ang_input = params.get('ang_input', True)
    ang_connections = params.get('ang_connections', True)
    
    bs_train = params['batch_size']
    dt = params['dt']
    washup = params['washup_steps']
    
    # Connection parameters
    n_connections_fraction = params['n_connections_fraction']
    n_connections = int(n_units * n_connections_fraction)
    anchor_con_fraction = params['anchor_con_fraction']
    n_connections_anchor = int(n_units * anchor_con_fraction)
    
    if ang_connections:
        n_connections_ang_fraction = params['n_connections_ang_fraction']
        anchor_con_fraction_ang = params['anchor_con_fraction_ang']
        n_connections_anchor_ang = int(anchor_con_fraction_ang * n_units)
        n_connections_ang = int(n_connections_ang_fraction * n_units)
    else:
        n_connections_anchor_ang = 0
        n_connections_ang = 0
    
    print(f"\nNetwork: n_units={n_units}, ang_input={ang_input}, ang_connections={ang_connections}")
    
    #%%
    # Create input maps
    lin_input_map = torch.zeros(1, n_units)
    num_non_zero = params['non_zero_elements']
    non_zero_indices = torch.randperm(n_units)[:num_non_zero]
    magnitude_min = params['magnitude_min']
    magnitude_max = params['magnitude_max']
    lin_input_map[0, non_zero_indices] = torch.rand(num_non_zero) * (magnitude_max - magnitude_min) + magnitude_min
    
    ang_input_map = torch.zeros(1, n_units)
    if ang_input:
        num_non_zero_ang = params['non_zero_elements_ang']
        non_zero_indices_ang = torch.randperm(n_units)[:num_non_zero_ang]
        magnitude_min_ang = params['magnitude_min_ang']
        magnitude_max_ang = params['magnitude_max_ang']
        ang_input_map[0, non_zero_indices_ang] = torch.rand(num_non_zero_ang) * (magnitude_max_ang - magnitude_min_ang) + magnitude_min_ang
    
    #%%
    # Create model
    model = UnicycleReservoir(
        n_inp=1, n_units=n_units, dt=dt, n_out=2,
        lin_input_map=lin_input_map,
        lin_stiff_min=params['lin_stiff_min'],
        lin_stiff_max=params['lin_stiff_max'],
        ang_stiff_min=params['ang_stiff_min'],
        ang_stiff_max=params['ang_stiff_max'],
        lin_damping_min=params['lin_damping_min'],
        lin_damping_max=params['lin_damping_max'],
        ang_damping_min=params['ang_damping_min'],
        ang_damping_max=params['ang_damping_max'],
        eq_dist_min=params['eq_dist_min'],
        eq_dist_max=params['eq_dist_max'],
        eq_dist_min_ang=params['eq_dist_min_ang'],
        eq_dist_max_ang=params['eq_dist_max_ang'],
        n_connections=n_connections,
        n_connections_anchor=n_connections_anchor,
        n_past_steps_readout=0,
        n_connections_ang=n_connections_ang,
        n_connections_anchor_ang=n_connections_anchor_ang,
        inp_bias=params['inp_bias'],
        ang_input_map=ang_input_map
    ).to(device)
    
    # Get stiffness and equilibrium distance matrices for equilibrium analysis
    K_mat = model.unicycle_network.stiffness_coupling_matrix.detach().cpu().numpy()
    A_mat = model.unicycle_network.eq_distances_matrix.detach().cpu().numpy().squeeze(-1)
    
    print(f"Stiffness matrix shape: {K_mat.shape}")
    print(f"Eq distances matrix shape: {A_mat.shape}")
    
    #%%
    # Load data
    print("\nLoading FordA data...")
    train_loader, valid_loader, test_loader = get_FordA_data(bs_train, bs_train)
    
    # Select data source based on configuration
    if USE_DATA_SOURCE == "test":
        data_loader = test_loader
        data_source_name = "TEST"
    else:
        data_loader = train_loader
        data_source_name = "TRAIN"
    
    print(f"Using {data_source_name} data for equilibrium analysis")
    
    #%%
    # Initialize model states
    model.set_init_states_random(bs_train)
    model.x_init = model.x_init.to(device)
    model.z_init = model.z_init.to(device)
    model.theta_init = model.theta_init.to(device)
    model.s_init = model.s_init.to(device)
    model.omega_init = model.omega_init.to(device)
    model.lin_input_map = model.lin_input_map.to(device)
    model.ang_input_map = model.ang_input_map.to(device)
    model.unicycle_network.lin_damping = model.unicycle_network.lin_damping.to(device)
    model.unicycle_network.ang_damping = model.unicycle_network.ang_damping.to(device)
    model.unicycle_network.mass_vector = model.unicycle_network.mass_vector.to(device)
    model.unicycle_network.j_vector = model.unicycle_network.j_vector.to(device)
    
    model.s_init[:, 0] = 0
    model.omega_init[:, :] = 0
    
    if not aligned_orientations:
        model.theta_init[:, :] = torch.rand(model.theta_init.size()) * (4*torch.pi) - 2*torch.pi
        model.theta_init[:, 0] = 0
    else:
        model.theta_init[:, :] = torch.rand(1) * (4*torch.pi) - 2*torch.pi
    
    #%%
    # Washup phase
    print("\nRunning washup phase...")
    x = model.x_init[0:1, :]
    z = model.z_init[0:1, :]
    theta = model.theta_init[0:1, :]
    s = model.s_init[0:1, :]
    omega = model.omega_init[0:1, :]
    
    u_lin = torch.zeros((1, washup, 1), device=device)
    u_ang = torch.zeros_like(u_lin, device=device)
    
    for t in range(washup):
        linear_input = u_lin[:, t] @ model.lin_input_map
        angular_input = u_ang[:, t] @ model.ang_input_map
        x, z, theta, s, omega = model.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)
    
    model.set_init_states(bs_train, x, z, theta, s, omega)
    
    #%%
    # Run on selected data and collect final states
    print(f"\nRunning reservoir on {data_source_name} data...")
    all_states_list = []
    all_labels = []
    
    for batch_x, batch_labels in tqdm(data_loader, desc="Processing batches"):
        batch_x = batch_x.to(device)
        batch_labels = batch_labels.to(device)
        
        states_list, output, mid_states = model(batch_x, batch_x)
        
        # Get final state: states_list[-1] has shape (batch, 5*n_units)
        # [x, y, theta, s, omega] for all units
        final_states = states_list[-1].detach().cpu().numpy()
        all_states_list.append(final_states)
        all_labels.append(batch_labels.cpu().numpy())
    
    all_final_states = np.vstack(all_states_list)
    all_labels = np.concatenate(all_labels).flatten()
    
    print(f"Final states shape: {all_final_states.shape}")
    print(f"Labels shape: {all_labels.shape}")
    print(f"Class 0: {np.sum(all_labels == 0)}, Class 1: {np.sum(all_labels == 1)}")
    
    #%%
    # Extract x, y, theta from final states
    all_x = all_final_states[:, 0:n_units]
    all_y = all_final_states[:, n_units:2*n_units]
    all_theta = all_final_states[:, 2*n_units:3*n_units]
    
    print(f"\nExtracted: x shape={all_x.shape}, y shape={all_y.shape}, theta shape={all_theta.shape}")
    
    #%%
    # ============================================================
    # TRAIN CLASSIFIER ON FINAL STATES
    # ============================================================
    print("\n" + "="*70)
    print(f"TRAINING CLASSIFIER ON FINAL (x, y) STATES ({data_source_name} DATA)")
    print("="*70)
    
    # Features: x and y for each unit -> 2*n_units features
    X_train = np.column_stack([all_x, all_y])  # (n_samples, 2*n_units)
    y_train = all_labels
    
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train_scaled, y_train)
    
    # Get weights: clf.coef_ has shape (1, 2*n_units) for binary classification
    # First n_units are x weights, next n_units are y weights
    weights = clf.coef_[0]
    x_weights = weights[:n_units]
    y_weights = weights[n_units:]
    
    # Compute total weight magnitude per unit (combining x and y weights)
    unit_weight_magnitude = np.sqrt(x_weights**2 + y_weights**2)
    
    # Rank units by weight magnitude
    unit_ranking = np.argsort(unit_weight_magnitude)[::-1]  # Descending order
    n_top = n_units // 2
    top_units = unit_ranking[:n_top]
    bottom_units = unit_ranking[n_top:]
    
    print(f"\nClassifier accuracy on training data: {clf.score(X_train_scaled, y_train):.4f}")
    print(f"Top {n_top} units by weight magnitude: {top_units}")
    print(f"Bottom {n_units - n_top} units: {bottom_units}")
    print(f"\nWeight magnitudes - Top: {unit_weight_magnitude[top_units].mean():.4f} ± {unit_weight_magnitude[top_units].std():.4f}")
    print(f"Weight magnitudes - Bottom: {unit_weight_magnitude[bottom_units].mean():.4f} ± {unit_weight_magnitude[bottom_units].std():.4f}")

    # Get classifier predictions for sample selection
    y_pred = clf.predict(X_train_scaled)
    print(f"\nClassifier predictions: {np.sum(y_pred == 0)} class 0, {np.sum(y_pred == 1)} class 1")

    #%%
    # ============================================================
    # FIND EQUILIBRIA FOR SAMPLE STATES FROM EACH CLASS
    # ============================================================
    print("\n" + "="*70)
    print("FINDING EQUILIBRIA NEAR FINAL STATES")
    print("="*70)
    
    # Select sample indices based on CLASSIFIER PREDICTIONS (not ground truth labels)
    n_samples_per_class = 50  # Analyze this many samples per class
    
    idxs_pred_0 = np.where(y_pred == 0)[0]
    idxs_pred_1 = np.where(y_pred == 1)[0]
    
    sample_idxs_0 = idxs_pred_0[:n_samples_per_class]
    sample_idxs_1 = idxs_pred_1[:n_samples_per_class]
    
    print(f"Using samples classified as class 0: {len(sample_idxs_0)} samples")
    print(f"Using samples classified as class 1: {len(sample_idxs_1)} samples")
    
    # Store results
    equilibria_info_0 = []  # For predicted class 0
    equilibria_info_1 = []  # For predicted class 1
    
    #%%
    # Find equilibria for Class 0 samples (predicted)
    print(f"\nFinding equilibria for {len(sample_idxs_0)} predicted Class 0 samples...")
    for idx in tqdm(sample_idxs_0):
        # Get position and theta for this sample
        p_state = np.column_stack([all_x[idx], all_y[idx]])  # (n_units, 2)
        theta_state = all_theta[idx]  # (n_units,)
        
        # Find equilibria using this state's theta and position as p0
        equilibria, F, positions_func = find_equilibria_from_state(
            K_mat, A_mat, theta_state, p_state, n_units,
            n_trials=N_EQUILIBRIUM_TRIALS // 10,  # Fewer trials per sample
            tol=EQUILIBRIUM_TOL,
            unique_tol=UNIQUE_TOL
        )
        
        # Compute distances to equilibria
        if len(equilibria) > 0:
            distances = distance_to_equilibria(p_state, equilibria, positions_func)
            closest_eq_idx = np.argmin(distances)
            closest_dist = distances[closest_eq_idx]
        else:
            closest_dist = np.inf
            closest_eq_idx = -1
        
        equilibria_info_0.append({
            'idx': idx,
            'n_equilibria': len(equilibria),
            'closest_distance': closest_dist,
            'equilibria': equilibria,
        })
    
    #%%
    # Find equilibria for Class 1 samples (predicted)
    print(f"\nFinding equilibria for {len(sample_idxs_1)} predicted Class 1 samples...")
    for idx in tqdm(sample_idxs_1):
        p_state = np.column_stack([all_x[idx], all_y[idx]])
        theta_state = all_theta[idx]
        
        equilibria, F, positions_func = find_equilibria_from_state(
            K_mat, A_mat, theta_state, p_state, n_units,
            n_trials=N_EQUILIBRIUM_TRIALS // 10,
            tol=EQUILIBRIUM_TOL,
            unique_tol=UNIQUE_TOL
        )
        
        if len(equilibria) > 0:
            distances = distance_to_equilibria(p_state, equilibria, positions_func)
            closest_eq_idx = np.argmin(distances)
            closest_dist = distances[closest_eq_idx]
        else:
            closest_dist = np.inf
            closest_eq_idx = -1
        
        equilibria_info_1.append({
            'idx': idx,
            'n_equilibria': len(equilibria),
            'closest_distance': closest_dist,
            'equilibria': equilibria,
        })
    
    #%%
    # ============================================================
    # ANALYZE RESULTS
    # ============================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY (based on classifier predictions)")
    print("="*70)
    
    n_eq_0 = [info['n_equilibria'] for info in equilibria_info_0]
    n_eq_1 = [info['n_equilibria'] for info in equilibria_info_1]
    dist_0 = [info['closest_distance'] for info in equilibria_info_0 if info['closest_distance'] < np.inf]
    dist_1 = [info['closest_distance'] for info in equilibria_info_1 if info['closest_distance'] < np.inf]
    
    print(f"\nPredicted Class 0:")
    print(f"  Average equilibria found: {np.mean(n_eq_0):.2f} ± {np.std(n_eq_0):.2f}")
    if len(dist_0) > 0:
        print(f"  Average closest distance: {np.mean(dist_0):.4f} ± {np.std(dist_0):.4f}")
    else:
        print(f"  No equilibria found for any sample")
    
    print(f"\nPredicted Class 1:")
    print(f"  Average equilibria found: {np.mean(n_eq_1):.2f} ± {np.std(n_eq_1):.2f}")
    if len(dist_1) > 0:
        print(f"  Average closest distance: {np.mean(dist_1):.4f} ± {np.std(dist_1):.4f}")
    else:
        print(f"  No equilibria found for any sample")
    
    #%%
    # ============================================================
    # CLASSIFIER WEIGHT ANALYSIS: TOP UNITS VS EQUILIBRIUM DISTANCE
    # ============================================================
    print("\n" + "="*70)
    print("CLASSIFIER WEIGHT VS EQUILIBRIUM DISTANCE ANALYSIS")
    print("="*70)
    
    # Using classifier weights computed earlier
    # top_units, bottom_units, unit_weight_magnitude already defined
    
    # For each sample with equilibria, compare |s| values for top vs bottom units
    # |s| = distance along heading to reach equilibrium (s=0 means already at equilibrium)
    top_s_values = []  # |s| for top units
    bottom_s_values = []  # |s| for bottom units
    
    all_equilibria_info = equilibria_info_0 + equilibria_info_1
    
    for info in all_equilibria_info:
        if len(info['equilibria']) == 0:
            continue
        
        # Find the closest equilibrium (by total distance)
        idx = info['idx']
        p_state = np.column_stack([all_x[idx], all_y[idx]])
        theta_state = all_theta[idx]
        
        _, F, positions_func = find_equilibria_from_state(
            K_mat, A_mat, theta_state, p_state, n_units, n_trials=0
        )
        
        distances = distance_to_equilibria(p_state, info['equilibria'], positions_func)
        closest_eq_idx = np.argmin(distances)
        s_eq = info['equilibria'][closest_eq_idx]  # (n_units-1,) - s values for units 1 to n_units-1
        
        # s_eq doesn't include unit 0 (fixed at s=0)
        # Create full s vector
        s_full = np.zeros(n_units)
        s_full[1:] = s_eq
        
        # Get |s| for top and bottom units (excluding unit 0 from bottom if present)
        for unit_idx in top_units:
            if unit_idx > 0:  # Unit 0 is fixed
                top_s_values.append(np.abs(s_full[unit_idx]))
        
        for unit_idx in bottom_units:
            if unit_idx > 0:
                bottom_s_values.append(np.abs(s_full[unit_idx]))
    
    if len(top_s_values) > 0 and len(bottom_s_values) > 0:
        print(f"\n|s| values (distance to equilibrium along heading):")
        print(f"  Top units:    {np.mean(top_s_values):.4f} ± {np.std(top_s_values):.4f} (n={len(top_s_values)})")
        print(f"  Bottom units: {np.mean(bottom_s_values):.4f} ± {np.std(bottom_s_values):.4f} (n={len(bottom_s_values)})")
        
        # Statistical test
        t_stat, p_value_ttest = stats.ttest_ind(top_s_values, bottom_s_values)
        print(f"\n  t-test: t={t_stat:.3f}, p={p_value_ttest:.4f}")
        
        if np.mean(top_s_values) < np.mean(bottom_s_values):
            print(f"  ✓ Top classifier units are CLOSER to equilibrium (lower |s|)")
        else:
            print(f"  ✗ Top classifier units are FARTHER from equilibrium (higher |s|)")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 1. Weight magnitude by unit
        ax = axes[0]
        colors = ['green' if i in top_units else 'gray' for i in range(n_units)]
        ax.bar(range(n_units), unit_weight_magnitude, color=colors, alpha=0.7)
        ax.set_xlabel('Unit index')
        ax.set_ylabel('Classifier weight magnitude')
        ax.set_title(f'Unit importance (green = top {n_top})')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. |s| distribution comparison
        ax = axes[1]
        ax.hist(top_s_values, bins=30, alpha=0.6, label=f'Top {n_top} units', color='green')
        ax.hist(bottom_s_values, bins=30, alpha=0.6, label=f'Bottom {n_units-n_top} units', color='gray')
        ax.axvline(np.mean(top_s_values), color='green', linestyle='--', linewidth=2)
        ax.axvline(np.mean(bottom_s_values), color='gray', linestyle='--', linewidth=2)
        ax.set_xlabel('|s| (distance to equilibrium along heading)')
        ax.set_ylabel('Count')
        ax.set_title(f'Equilibrium distance by unit importance\n(p={p_value_ttest:.4f})')
        ax.legend()
        
        # 3. Bar comparison
        ax = axes[2]
        categories = [f'Top {n_top}\nunits', f'Bottom {n_units-n_top}\nunits']
        means = [np.mean(top_s_values), np.mean(bottom_s_values)]
        stds = [np.std(top_s_values), np.std(bottom_s_values)]
        colors = ['green', 'gray']
        
        ax.bar(categories, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax.set_ylabel('Mean |s| (distance to equilibrium)')
        ax.set_title('Do important units settle closer to equilibrium?')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{parent_dir}/plots/forda_classifier_weights_vs_equilibrium.png', dpi=150)
        plt.show()
    else:
        print("\nNot enough equilibria data for weight analysis")

    #%%
    # ============================================================
    # EQUILIBRIUM IMPORTANCE TEST: CLASSIFY FROM EQUILIBRIUM POSITIONS
    # ============================================================
    print("\n" + "="*70)
    print("EQUILIBRIUM IMPORTANCE TEST")
    print("="*70)
    
    # Test 1: Train classifier on equilibrium positions instead of final states
    # If equilibria carry class information, accuracy should be similar
    
    # Collect equilibrium positions for samples that have equilibria
    eq_positions_list = []
    eq_labels_list = []
    final_positions_with_eq = []
    
    all_equilibria_info = equilibria_info_0 + equilibria_info_1
    
    for info in all_equilibria_info:
        if len(info['equilibria']) == 0:
            continue
        
        idx = info['idx']
        p_state = np.column_stack([all_x[idx], all_y[idx]])
        theta_state = all_theta[idx]
        e = np.stack([np.cos(theta_state), np.sin(theta_state)], axis=1)
        
        # Find closest equilibrium
        _, F, positions_func = find_equilibria_from_state(
            K_mat, A_mat, theta_state, p_state, n_units, n_trials=0
        )
        distances = distance_to_equilibria(p_state, info['equilibria'], positions_func)
        closest_eq_idx = np.argmin(distances)
        s_eq = info['equilibria'][closest_eq_idx]
        
        # Get equilibrium position
        s_full = np.zeros(n_units)
        s_full[1:] = s_eq
        p_eq = p_state + s_full[:, None] * e
        
        # Flatten for classifier: [x_0, x_1, ..., y_0, y_1, ...]
        eq_flat = np.concatenate([p_eq[:, 0], p_eq[:, 1]])
        final_flat = np.concatenate([p_state[:, 0], p_state[:, 1]])
        
        eq_positions_list.append(eq_flat)
        final_positions_with_eq.append(final_flat)
        eq_labels_list.append(y_pred[idx])  # Use classifier predictions as labels
    
    if len(eq_positions_list) > 10:
        X_eq = np.array(eq_positions_list)
        X_final_subset = np.array(final_positions_with_eq)
        y_eq = np.array(eq_labels_list)
        
        # Scale using the same scaler
        X_eq_scaled = scaler.transform(X_eq)
        X_final_subset_scaled = scaler.transform(X_final_subset)
        
        # Train classifier on equilibrium positions
        clf_eq = LogisticRegression(max_iter=1000, random_state=SEED)
        clf_eq.fit(X_eq_scaled, y_eq)
        acc_from_eq = clf_eq.score(X_eq_scaled, y_eq)
        
        # For comparison: accuracy on final states (subset with equilibria)
        acc_from_final_subset = clf.score(X_final_subset_scaled, y_eq)
        
        # Cross-test: Train on equilibria, test on final states
        acc_eq_to_final = clf_eq.score(X_final_subset_scaled, y_eq)
        
        # Cross-test: Train on final (original clf), test on equilibria
        acc_final_to_eq = clf.score(X_eq_scaled, y_eq)
        
        print(f"\nTest 1: Classification from different representations")
        print(f"  Samples with equilibria found: {len(y_eq)}")
        print(f"  ")
        print(f"  Accuracy on FINAL states (subset):      {acc_from_final_subset:.4f}")
        print(f"  Accuracy on EQUILIBRIUM positions:      {acc_from_eq:.4f}")
        print(f"  ")
        print(f"  Cross-test (train eq → test final):     {acc_eq_to_final:.4f}")
        print(f"  Cross-test (train final → test eq):     {acc_final_to_eq:.4f}")
        
        if acc_from_eq > 0.9 * acc_from_final_subset:
            print(f"\n  ✓ Equilibrium positions carry most class information!")
        elif acc_from_eq > 0.7 * acc_from_final_subset:
            print(f"\n  ~ Equilibrium positions carry some class information")
        else:
            print(f"\n  ✗ Equilibrium positions lose significant class information")
        
        # Test 2: Perturb final states AWAY from equilibrium
        # If equilibria matter, this should hurt classification
        print(f"\nTest 2: Perturbation away from equilibrium")
        
        perturbation_magnitudes = [0.1, 0.5, 1.0, 2.0, 5.0]
        print(f"  Perturbing states in direction AWAY from equilibrium...")
        
        for pert_mag in perturbation_magnitudes:
            X_perturbed = []
            for i, info in enumerate([inf for inf in all_equilibria_info if len(inf['equilibria']) > 0]):
                idx = info['idx']
                p_state = np.column_stack([all_x[idx], all_y[idx]])
                theta_state = all_theta[idx]
                e = np.stack([np.cos(theta_state), np.sin(theta_state)], axis=1)
                
                # Get closest equilibrium
                _, F, positions_func = find_equilibria_from_state(
                    K_mat, A_mat, theta_state, p_state, n_units, n_trials=0
                )
                distances = distance_to_equilibria(p_state, info['equilibria'], positions_func)
                closest_eq_idx = np.argmin(distances)
                s_eq = info['equilibria'][closest_eq_idx]
                
                s_full = np.zeros(n_units)
                s_full[1:] = s_eq
                p_eq = p_state + s_full[:, None] * e
                
                # Direction from equilibrium to final state (away from eq)
                direction = p_state - p_eq
                direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
                direction_norm[direction_norm == 0] = 1  # Avoid div by zero
                direction_unit = direction / direction_norm
                
                # Perturb away from equilibrium
                p_perturbed = p_state + pert_mag * direction_unit
                
                # Flatten
                perturbed_flat = np.concatenate([p_perturbed[:, 0], p_perturbed[:, 1]])
                X_perturbed.append(perturbed_flat)
            
            X_perturbed = np.array(X_perturbed)
            X_perturbed_scaled = scaler.transform(X_perturbed)
            acc_perturbed = clf.score(X_perturbed_scaled, y_eq)
            
            print(f"    Perturbation magnitude {pert_mag:.1f}: accuracy = {acc_perturbed:.4f} (Δ = {acc_perturbed - acc_from_final_subset:+.4f})")
        
        print(f"\n  If accuracy drops with larger perturbations, distance from equilibrium matters.")
    else:
        print("\nNot enough samples with equilibria for importance test")

    #%%
    # ============================================================
    # CROSS-CLASS EQUILIBRIA COMPARISON
    # ============================================================
    print("\n" + "="*70)
    print("CROSS-CLASS EQUILIBRIA COMPARISON")
    print("="*70)
    
    # Collect all equilibria converted to (x, y) positions (not s-space!)
    # Each sample has different theta and p0, so s-vectors are not comparable.
    # We must convert to actual positions for meaningful comparison.
    all_equilibria_0 = []  # List of (sample_idx, p_eq) tuples where p_eq is (n_units, 2)
    all_equilibria_1 = []
    
    for info in equilibria_info_0:
        idx = info['idx']
        # Reconstruct the positions function for this sample
        p0 = np.column_stack([all_x[idx], all_y[idx]])
        theta = all_theta[idx]
        e = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        
        for s_eq in info['equilibria']:
            # Convert s to actual positions
            s_full = np.zeros(n_units)
            s_full[1:] = s_eq
            p_eq = p0 + s_full[:, None] * e  # (n_units, 2)
            all_equilibria_0.append((idx, p_eq))
    
    for info in equilibria_info_1:
        idx = info['idx']
        p0 = np.column_stack([all_x[idx], all_y[idx]])
        theta = all_theta[idx]
        e = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        
        for s_eq in info['equilibria']:
            s_full = np.zeros(n_units)
            s_full[1:] = s_eq
            p_eq = p0 + s_full[:, None] * e
            all_equilibria_1.append((idx, p_eq))
    
    print(f"\nTotal equilibria found - Class 0: {len(all_equilibria_0)}, Class 1: {len(all_equilibria_1)}")
    
    # Scale equilibrium positions using the same scaler as the classifier
    # The scaler was fit on X_train = [all_x, all_y] with shape (n_samples, 2*n_units)
    # We need to transform equilibrium positions the same way
    def scale_equilibrium_positions(p_eq, scaler, n_units):
        """Scale equilibrium positions using the classifier's scaler."""
        # p_eq is (n_units, 2) -> flatten to match scaler input format
        # scaler expects (1, 2*n_units) where first n_units are x, next n_units are y
        flat = np.concatenate([p_eq[:, 0], p_eq[:, 1]])  # (2*n_units,)
        scaled = scaler.transform(flat.reshape(1, -1))  # (1, 2*n_units)
        # Reshape back to (n_units, 2)
        return np.column_stack([scaled[0, :n_units], scaled[0, n_units:]])
    
    # Scale all equilibria
    all_equilibria_0_scaled = [(idx, scale_equilibrium_positions(p_eq, scaler, n_units)) 
                               for idx, p_eq in all_equilibria_0]
    all_equilibria_1_scaled = [(idx, scale_equilibrium_positions(p_eq, scaler, n_units)) 
                               for idx, p_eq in all_equilibria_1]
    
    print(f"Equilibrium positions scaled using classifier's StandardScaler")
    
    if len(all_equilibria_0_scaled) > 1 and len(all_equilibria_1_scaled) > 1:
        # Use only the top units by classifier weight for distance computation
        # top_units is already defined from classifier training
        # Filter out unit 0 if present (it's fixed)
        top_units_filtered = [u for u in top_units if u > 0]
        
        print(f"\nUsing top {len(top_units_filtered)} units (by classifier weight) for equilibrium comparison:")
        print(f"  Units: {top_units_filtered}")
        
        # ============================================================
        # BASELINE: Expected distance for random points in this space
        # ============================================================
        # Compute distances between random samples from the scaled training data
        # This gives us a baseline for "how far apart are random points?"
        n_random_samples = 1000
        random_indices = np.random.choice(len(X_train_scaled), size=n_random_samples, replace=True)
        random_points_scaled = X_train_scaled[random_indices]
        
        # Reshape to (n_samples, n_units, 2) for consistent indexing
        random_points_reshaped = np.stack([
            random_points_scaled[:, :n_units],  # x coordinates
            random_points_scaled[:, n_units:]   # y coordinates
        ], axis=2)  # (n_samples, n_units, 2)
        
        # Compute pairwise distances for random points (using top units only)
        n_random_pairs = min(5000, n_random_samples * (n_random_samples - 1) // 2)
        random_distances = []
        for _ in range(n_random_pairs):
            i, j = np.random.choice(n_random_samples, size=2, replace=False)
            p_i = random_points_reshaped[i]  # (n_units, 2)
            p_j = random_points_reshaped[j]  # (n_units, 2)
            dist = np.linalg.norm(p_i[top_units_filtered] - p_j[top_units_filtered])
            random_distances.append(dist)
        
        random_baseline = np.mean(random_distances)
        random_std = np.std(random_distances)
        
        # Also compute theoretical expectation for standard normal
        # For d dimensions, E[||X-Y||] ≈ sqrt(2d) when X,Y ~ N(0,I)
        d_effective = len(top_units_filtered) * 2  # x and y for each unit
        theoretical_baseline = np.sqrt(2 * d_effective)
        
        print(f"\n--- BASELINE: Random point distances in scaled space ---")
        print(f"  Empirical (from training data): {random_baseline:.4f} ± {random_std:.4f}")
        print(f"  Theoretical (std normal, d={d_effective}): {theoretical_baseline:.4f}")
        
        # Compute within-class distances (how similar are equilibria within same class)
        # Using SCALED positions of TOP UNITS ONLY
        def compute_pairwise_distances(equilibria_list):
            """Compute pairwise distances between scaled equilibria positions."""
            n = len(equilibria_list)
            if n < 2:
                return []
            distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    p_i = equilibria_list[i][1]  # (n_units, 2) scaled positions
                    p_j = equilibria_list[j][1]  # (n_units, 2) scaled positions
                    # Compare positions of TOP UNITS ONLY
                    dist = np.linalg.norm(p_i[top_units_filtered] - p_j[top_units_filtered])
                    distances.append(dist)
            return distances
        
        within_class_0 = compute_pairwise_distances(all_equilibria_0_scaled)
        within_class_1 = compute_pairwise_distances(all_equilibria_1_scaled)
        
        # Compute between-class distances (using scaled positions)
        between_class = []
        for idx_0, p_0 in all_equilibria_0_scaled:
            for idx_1, p_1 in all_equilibria_1_scaled:
                dist = np.linalg.norm(p_0[top_units_filtered] - p_1[top_units_filtered])
                between_class.append(dist)
        
        print(f"\n--- EQUILIBRIA DISTANCES ---")
        print(f"Within-Class 0 distances: {np.mean(within_class_0):.4f} ± {np.std(within_class_0):.4f} (n={len(within_class_0)} pairs)")
        print(f"Within-Class 1 distances: {np.mean(within_class_1):.4f} ± {np.std(within_class_1):.4f} (n={len(within_class_1)} pairs)")
        print(f"Between-Class distances:  {np.mean(between_class):.4f} ± {np.std(between_class):.4f} (n={len(between_class)} pairs)")
        print(f"Random baseline:          {random_baseline:.4f} ± {random_std:.4f}")
        print(f"  (All distances in SCALED space, using top {len(top_units_filtered)} classifier units)")
        
        # Compare to baseline
        print(f"\n--- COMPARISON TO RANDOM BASELINE ---")
        print(f"  Within-0 / Random: {np.mean(within_class_0) / random_baseline:.2f}x")
        print(f"  Within-1 / Random: {np.mean(within_class_1) / random_baseline:.2f}x")
        print(f"  Between / Random:  {np.mean(between_class) / random_baseline:.2f}x")
        
        # Check if within-class < between-class (suggests class-specific equilibria)
        avg_within = (np.mean(within_class_0) + np.mean(within_class_1)) / 2
        avg_between = np.mean(between_class)
        
        # More detailed comparison
        print(f"\n--- DETAILED COMPARISON ---")
        print(f"  Between vs Within-0: {avg_between:.4f} vs {np.mean(within_class_0):.4f} → ", end="")
        if avg_between > np.mean(within_class_0):
            print("Between > Within-0 ✓")
        else:
            print("Between ≤ Within-0 ✗")
        
        print(f"  Between vs Within-1: {avg_between:.4f} vs {np.mean(within_class_1):.4f} → ", end="")
        if avg_between > np.mean(within_class_1):
            print("Between > Within-1 ✓")
        else:
            print("Between ≤ Within-1 ✗")
        
        # Check for true separation vs asymmetric spread
        true_separation = (avg_between > np.mean(within_class_0)) and (avg_between > np.mean(within_class_1))
        
        if true_separation:
            print(f"\n✓ TRUE SEPARATION: Between-class > BOTH within-class distances")
            print(f"  → Classes occupy distinct regions of equilibrium space")
        elif avg_within < avg_between:
            print(f"\n⚠ ASYMMETRIC SPREAD: Between > avg(within), but not both individually")
            print(f"  → One class is tighter than the other, but they may overlap")
            print(f"  → Class 0 spread: {np.mean(within_class_0):.4f}, Class 1 spread: {np.mean(within_class_1):.4f}")
        else:
            print(f"\n✗ NO SEPARATION: Within-class distances ≥ between-class")
        
        print(f"  Ratio (between/avg_within): {avg_between/avg_within:.2f}")
        
        # Statistical test (permutation test)
        # H0: Class labels don't matter for equilibria clustering
        # H1: Within-class equilibria are more similar than between-class
        # Test statistic: (between - within), larger = more clustering by class
        observed_diff = avg_between - avg_within
        n_permutations = 1000
        perm_diffs = []
        
        # Use SCALED equilibria for permutation test
        all_equilibria_combined = all_equilibria_0_scaled + all_equilibria_1_scaled
        n_class_0 = len(all_equilibria_0_scaled)
        
        for _ in range(n_permutations):
            # Randomly shuffle class assignments
            perm_indices = np.random.permutation(len(all_equilibria_combined))
            perm_class_0 = [all_equilibria_combined[i] for i in perm_indices[:n_class_0]]
            perm_class_1 = [all_equilibria_combined[i] for i in perm_indices[n_class_0:]]
            
            perm_within_0 = compute_pairwise_distances(perm_class_0)
            perm_within_1 = compute_pairwise_distances(perm_class_1)
            
            perm_between = []
            for _, p_0 in perm_class_0:
                for _, p_1 in perm_class_1:
                    # Use top units only (same as above)
                    perm_between.append(np.linalg.norm(p_0[top_units_filtered] - p_1[top_units_filtered]))
            
            if len(perm_within_0) > 0 and len(perm_within_1) > 0:
                perm_avg_within = (np.mean(perm_within_0) + np.mean(perm_within_1)) / 2
                perm_avg_between = np.mean(perm_between)
                perm_diffs.append(perm_avg_between - perm_avg_within)
        
        # p-value: proportion of permutations with difference >= observed
        # If p=0, it means the observed clustering is stronger than ALL permutations
        perm_diffs = np.array(perm_diffs)
        p_value = np.mean(perm_diffs >= observed_diff)
        
        print(f"\nPermutation test:")
        print(f"  Observed (between - within): {observed_diff:.4f}")
        print(f"  Permutation mean: {np.mean(perm_diffs):.4f} ± {np.std(perm_diffs):.4f}")
        print(f"  Permutation range: [{np.min(perm_diffs):.4f}, {np.max(perm_diffs):.4f}]")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value == 0:
            print(f"  (p=0 means observed effect stronger than all {n_permutations} permutations)")
            print(f"  → Highly significant: Classes have distinct equilibria!")
        elif p_value < 0.05:
            print("  → Significant difference (p < 0.05): Classes have distinct equilibria!")
        else:
            print("  → Not significant: Equilibria distribution similar across classes")
        
        # Visualization: Distance distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        ax = axes[0]
        ax.hist(within_class_0, bins=20, alpha=0.5, label='Within Class 0', color='orange')
        ax.hist(within_class_1, bins=20, alpha=0.5, label='Within Class 1', color='blue')
        ax.hist(between_class, bins=20, alpha=0.5, label='Between Classes', color='green')
        ax.axvline(np.mean(within_class_0), color='orange', linestyle='--', linewidth=2)
        ax.axvline(np.mean(within_class_1), color='blue', linestyle='--', linewidth=2)
        ax.axvline(np.mean(between_class), color='green', linestyle='--', linewidth=2)
        ax.set_xlabel('Distance between equilibria (in (x,y) position space)')
        ax.set_ylabel('Count')
        ax.set_title('Equilibria Distance Distributions')
        ax.legend()
        
        ax = axes[1]
        categories = ['Within\nClass 0', 'Within\nClass 1', 'Between\nClasses']
        means = [np.mean(within_class_0), np.mean(within_class_1), np.mean(between_class)]
        stds = [np.std(within_class_0), np.std(within_class_1), np.std(between_class)]
        colors = ['orange', 'blue', 'green']
        
        bars = ax.bar(categories, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax.set_ylabel('Mean distance')
        ax.set_title(f'Equilibria Similarity Comparison\n(p-value: {p_value:.4f})')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{parent_dir}/plots/forda_equilibrium_cross_class_comparison_{data_source_name.lower()}.png', dpi=150)
        plt.show()
        
    else:
        print("\nNot enough equilibria found for cross-class comparison")

    #%%
    # ============================================================
    # VISUALIZATIONS
    # ============================================================
    
    # 1. Histogram of number of equilibria found
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ax = axes[0]
    ax.hist(n_eq_0, bins=range(max(n_eq_0+n_eq_1)+2), alpha=0.6, label='Class 0', color='orange')
    ax.hist(n_eq_1, bins=range(max(n_eq_0+n_eq_1)+2), alpha=0.6, label='Class 1', color='blue')
    ax.set_xlabel('Number of equilibria found')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of equilibria count')
    ax.legend()
    
    # 2. Histogram of closest distances
    ax = axes[1]
    if len(dist_0) > 0:
        ax.hist(dist_0, bins=20, alpha=0.6, label='Class 0', color='orange')
    if len(dist_1) > 0:
        ax.hist(dist_1, bins=20, alpha=0.6, label='Class 1', color='blue')
    ax.set_xlabel('Distance to closest equilibrium')
    ax.set_ylabel('Count')
    ax.set_title('Distance to nearest equilibrium')
    ax.legend()
    
    # 3. Scatter: n_equilibria vs closest_distance
    ax = axes[2]
    if len(dist_0) > 0:
        ax.scatter(n_eq_0[:len(dist_0)], dist_0, alpha=0.6, label='Class 0', color='orange', s=50)
    if len(dist_1) > 0:
        ax.scatter(n_eq_1[:len(dist_1)], dist_1, alpha=0.6, label='Class 1', color='blue', s=50)
    ax.set_xlabel('Number of equilibria found')
    ax.set_ylabel('Distance to closest equilibrium')
    ax.set_title('Equilibria count vs distance')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{parent_dir}/plots/forda_equilibrium_analysis_{data_source_name.lower()}.png', dpi=150)
    plt.show()
    
    #%%
    # 4. Visualize one sample from each class with its equilibria
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for class_idx, (info_list, class_name, color) in enumerate([
        (equilibria_info_0, 'Class 0', 'orange'),
        (equilibria_info_1, 'Class 1', 'blue')
    ]):
        ax = axes[class_idx]
        
        # Find a sample with equilibria
        sample_info = None
        for info in info_list:
            if info['n_equilibria'] > 0:
                sample_info = info
                break
        
        if sample_info is None:
            ax.text(0.5, 0.5, f'No equilibria found for {class_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        idx = sample_info['idx']
        p_state = np.column_stack([all_x[idx], all_y[idx]])
        theta_state = all_theta[idx]
        e_state = np.stack([np.cos(theta_state), np.sin(theta_state)], axis=1)
        
        # Plot current state positions
        ax.scatter(p_state[:, 0], p_state[:, 1], c=color, s=100, 
                  marker='o', label=f'Final state', zorder=5)
        
        # Plot heading directions
        for i in range(n_units):
            ax.arrow(p_state[i, 0], p_state[i, 1], 
                    0.3*e_state[i, 0], 0.3*e_state[i, 1],
                    head_width=0.1, head_length=0.05, fc=color, ec=color, alpha=0.5)
        
        # Plot equilibrium positions
        _, F, positions_func = find_equilibria_from_state(
            K_mat, A_mat, theta_state, p_state, n_units, n_trials=0
        )
        
        for eq_idx, s_eq in enumerate(sample_info['equilibria']):
            p_eq = positions_func(s_eq)
            ax.scatter(p_eq[:, 0], p_eq[:, 1], c='green', s=50, 
                      marker='^', alpha=0.7, zorder=4)
            # Connect to show correspondence
            for i in range(n_units):
                ax.plot([p_state[i, 0], p_eq[i, 0]], [p_state[i, 1], p_eq[i, 1]],
                       'g--', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{class_name} sample (idx={idx})\n'
                    f'{sample_info["n_equilibria"]} equilibria, '
                    f'closest dist={sample_info["closest_distance"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{parent_dir}/plots/forda_equilibrium_visualization_{data_source_name.lower()}.png', dpi=150)
    plt.show()
    
    print("\nAnalysis complete!")
    return equilibria_info_0, equilibria_info_1, all_x, all_y, all_theta, all_labels

#%%
if __name__ == "__main__":
    results = main()

# %%
