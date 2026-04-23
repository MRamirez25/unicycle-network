#%%
"""
Analyze final reservoir states in relation to equilibria for MNIST dataset.

This script:
1. Loads a trained MNIST model configuration from Optuna
2. Runs the reservoir on MNIST data
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

from utils import get_mnist_data
from unicycle_network_class import UnicycleNetwork, UnicycleReservoir

#%%
# ============================================================
# CONFIGURATION
# ============================================================

# Optuna study configuration
DATABASE_NAME = "unicycle_mnist_all_digits_logreg"
STUDY_NAME = "not_aligned_w_input_w_connections_actual_100_units_last_readout_no_tanh"

# Seed for reproducibility
SEED = 42

# Equilibrium finding parameters
N_EQUILIBRIUM_TRIALS = 50  # Random restarts (reduced since equilibria found easily)
EQUILIBRIUM_TOL = 1e-4  # Tolerance for equilibrium
UNIQUE_TOL = 1e-3  # Tolerance for considering equilibria as same

# Number of classes to analyze (MNIST has 10 classes)
CLASSES_TO_ANALYZE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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


def find_equilibria_from_state(K_mat, A_mat, theta, p_init, N, n_trials=100, tol=1e-6, unique_tol=1e-3, verbose=False):
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
        verbose: print debugging info
    
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
    n_success = 0
    n_fail = 0
    best_residual = np.inf
    
    # Initial guess: s = 0 (positions at p0)
    s0 = np.zeros(N - 1)
    
    # Try from the exact initial state
    try:
        sol = root(F, s0, method='hybr')
        residual = np.linalg.norm(sol.fun)
        best_residual = min(best_residual, residual)
        if sol.success and residual < tol:
            solutions.append(sol.x)
            n_success += 1
        else:
            n_fail += 1
    except Exception as ex:
        if verbose:
            print(f"    Exception at s0: {ex}")
        n_fail += 1
    
    # Try random perturbations
    for i in range(n_trials):
        s_init = np.random.randn(N - 1) * 0.5  # Small perturbations
        try:
            sol = root(F, s_init, method='hybr')
            residual = np.linalg.norm(sol.fun)
            best_residual = min(best_residual, residual)
            if sol.success and residual < tol:
                solutions.append(sol.x)
                n_success += 1
            else:
                n_fail += 1
        except Exception as ex:
            if verbose and i == 0:
                print(f"    Exception in small perturbation: {ex}")
            n_fail += 1
            continue
    
    # Also try larger range random restarts
    for i in range(n_trials // 2):
        s_init = np.random.randn(N - 1) * 5  # Larger perturbations
        try:
            sol = root(F, s_init, method='hybr')
            residual = np.linalg.norm(sol.fun)
            best_residual = min(best_residual, residual)
            if sol.success and residual < tol:
                solutions.append(sol.x)
                n_success += 1
            else:
                n_fail += 1
        except Exception as ex:
            if verbose and i == 0:
                print(f"    Exception in large perturbation: {ex}")
            n_fail += 1
            continue
    
    if verbose or len(solutions) == 0:
        print(f"    Trials: {n_success} success, {n_fail} fail, best_residual={best_residual:.2e}, tol={tol:.2e}")
    
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
    print("EQUILIBRIUM ANALYSIS FOR MNIST RESERVOIR STATES")
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
    n_units = 100  # MNIST uses 100 units
    aligned_orientations = False
    ang_input = True
    ang_connections = True
    
    bs_train = 500
    dt = params['dt']
    washup = params['washup_steps']
    
    # Connection parameters
    n_connections_fraction = params['n_connections_fraction']
    n_connections = int(n_units * n_connections_fraction)
    anchor_con_fraction = params['anchor_con_fraction']
    n_connections_anchor = int(n_units * anchor_con_fraction)
    
    n_connections_ang_fraction = params['n_connections_ang_fraction']
    anchor_con_fraction_ang = params['anchor_con_fraction_ang']
    n_connections_anchor_ang = int(anchor_con_fraction_ang * n_units)
    n_connections_ang = int(n_connections_ang_fraction * n_units)
    
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
    num_non_zero_ang = params['non_zero_elements_ang']
    non_zero_indices_ang = torch.randperm(n_units)[:num_non_zero_ang]
    magnitude_min_ang = params['magnitude_min_ang']
    magnitude_max_ang = params['magnitude_max_ang']
    ang_input_map[0, non_zero_indices_ang] = torch.rand(num_non_zero_ang) * (magnitude_max_ang - magnitude_min_ang) + magnitude_min_ang
    
    #%%
    # Create model
    model = UnicycleReservoir(
        n_inp=1, n_units=n_units, dt=dt, n_out=10,
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
    print("\nLoading MNIST data...")
    data_root = parent_dir + '/data/'
    train_loader, valid_loader, test_loader = get_mnist_data(
        bs_train=bs_train, bs_test=bs_train, 
        classes=CLASSES_TO_ANALYZE,
        new_fraction=0.5, test_fraction=0.5, path=data_root
    )
    
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
    # Run on training data and collect final states
    print("\nRunning reservoir on training data...")
    all_states_list = []
    all_labels = []
    
    for batch_images, batch_labels in tqdm(train_loader, desc="Processing batches"):
        # Reshape MNIST images: (batch, 28, 28) -> (batch, 784, 1)
        batch_images = batch_images.reshape(batch_images.shape[0], 1, 784)
        batch_images = batch_images.permute(0, 2, 1)
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        
        states_list, output, mid_states = model(batch_images, batch_images)
        
        # Get final state: states_list[-1] has shape (batch, 5*n_units)
        # [x, y, theta, s, omega] for all units
        final_states = states_list[-1].detach().cpu().numpy()
        all_states_list.append(final_states)
        all_labels.append(batch_labels.cpu().numpy())
    
    all_final_states = np.vstack(all_states_list)
    all_labels = np.concatenate(all_labels).flatten()
    
    print(f"Final states shape: {all_final_states.shape}")
    print(f"Labels shape: {all_labels.shape}")
    for c in CLASSES_TO_ANALYZE:
        print(f"Class {c}: {np.sum(all_labels == c)}")
    
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
    print("TRAINING CLASSIFIER ON FINAL (x, y) STATES")
    print("="*70)
    
    # Features: x and y for each unit -> 2*n_units features
    X_train = np.column_stack([all_x, all_y])  # (n_samples, 2*n_units)
    y_train = all_labels
    
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train_scaled, y_train)
    
    # Get weights: clf.coef_ has shape (n_classes, 2*n_units) for multiclass
    # For multiclass, we take the mean absolute weight across classes
    weights = np.mean(np.abs(clf.coef_), axis=0)
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
    print(f"Top {n_top} units by weight magnitude: {top_units[:10]}...")  # Show first 10
    print(f"\nWeight magnitudes - Top: {unit_weight_magnitude[top_units].mean():.4f} ± {unit_weight_magnitude[top_units].std():.4f}")
    print(f"Weight magnitudes - Bottom: {unit_weight_magnitude[bottom_units].mean():.4f} ± {unit_weight_magnitude[bottom_units].std():.4f}")

    # Get classifier predictions for sample selection
    y_pred = clf.predict(X_train_scaled)
    print(f"\nClassifier predictions distribution:")
    for c in CLASSES_TO_ANALYZE:
        print(f"  Class {c}: {np.sum(y_pred == c)}")

    #%%
    # ============================================================
    # FIND EQUILIBRIA FOR SAMPLE STATES FROM SELECTED CLASSES
    # ============================================================
    print("\n" + "="*70)
    print("FINDING EQUILIBRIA NEAR FINAL STATES")
    print("="*70)
    
    # For MNIST with 10 classes, let's analyze a subset of samples per class
    n_samples_per_class = 20  # Fewer samples due to 100 units being slower
    
    # Store results for each class
    equilibria_info_by_class = {c: [] for c in CLASSES_TO_ANALYZE}
    
    # Debug: check if we have data
    print(f"\nDebug: Total samples = {len(y_pred)}, n_units = {n_units}")
    print(f"Debug: CLASSES_TO_ANALYZE = {CLASSES_TO_ANALYZE}")
    
    for class_label in CLASSES_TO_ANALYZE:
        idxs_pred = np.where(y_pred == class_label)[0]
        sample_idxs = idxs_pred[:n_samples_per_class]
        
        print(f"\nClass {class_label}: found {len(idxs_pred)} predictions, using {len(sample_idxs)} samples")
        
        if len(sample_idxs) == 0:
            print(f"  WARNING: No samples for class {class_label}, skipping...")
            continue
        
        first_sample = True
        for idx in tqdm(sample_idxs, desc=f"Class {class_label}"):
            p_state = np.column_stack([all_x[idx], all_y[idx]])
            theta_state = all_theta[idx]
            
            # Verbose for first sample of first class only
            verbose = first_sample and class_label == CLASSES_TO_ANALYZE[0]
            
            equilibria, F, positions_func = find_equilibria_from_state(
                K_mat, A_mat, theta_state, p_state, n_units,
                n_trials=N_EQUILIBRIUM_TRIALS,
                tol=EQUILIBRIUM_TOL,
                unique_tol=UNIQUE_TOL,
                verbose=verbose
            )
            first_sample = False
            
            if len(equilibria) > 0:
                distances = distance_to_equilibria(p_state, equilibria, positions_func)
                closest_eq_idx = np.argmin(distances)
                closest_dist = distances[closest_eq_idx]
            else:
                closest_dist = np.inf
                closest_eq_idx = -1
            
            equilibria_info_by_class[class_label].append({
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
    
    for class_label in CLASSES_TO_ANALYZE:
        info_list = equilibria_info_by_class[class_label]
        n_eq = [info['n_equilibria'] for info in info_list]
        dist = [info['closest_distance'] for info in info_list if info['closest_distance'] < np.inf]
        
        print(f"\nPredicted Class {class_label}:")
        print(f"  Average equilibria found: {np.mean(n_eq):.2f} ± {np.std(n_eq):.2f}")
        if len(dist) > 0:
            print(f"  Average closest distance: {np.mean(dist):.4f} ± {np.std(dist):.4f}")
        else:
            print(f"  No equilibria found for any sample")

    #%%
    # ============================================================
    # CROSS-CLASS EQUILIBRIUM COMPARISON (SCALED SPACE)
    # ============================================================
    print("\n" + "="*70)
    print("CROSS-CLASS EQUILIBRIUM COMPARISON (SCALED SPACE)")
    print("="*70)
    
    # Collect equilibrium positions in SCALED space for each class
    # Scale using the same scaler as the classifier
    equilibria_by_class_scaled = {c: [] for c in CLASSES_TO_ANALYZE}
    
    for class_label in CLASSES_TO_ANALYZE:
        for info in equilibria_info_by_class[class_label]:
            if len(info['equilibria']) == 0:
                continue
            
            idx = info['idx']
            p_state = np.column_stack([all_x[idx], all_y[idx]])
            theta_state = all_theta[idx]
            e = np.stack([np.cos(theta_state), np.sin(theta_state)], axis=1)
            
            # Get closest equilibrium
            min_dist = np.inf
            closest_s_eq = None
            for s_eq in info['equilibria']:
                s_full = np.zeros(n_units)
                s_full[1:] = s_eq
                p_eq_candidate = p_state + s_full[:, None] * e
                dist = np.linalg.norm(p_state[1:] - p_eq_candidate[1:])
                if dist < min_dist:
                    min_dist = dist
                    closest_s_eq = s_eq
            
            # Get equilibrium position
            s_full = np.zeros(n_units)
            s_full[1:] = closest_s_eq
            p_eq = p_state + s_full[:, None] * e
            
            # Flatten and scale
            eq_flat = np.concatenate([p_eq[:, 0], p_eq[:, 1]])
            eq_scaled = scaler.transform(eq_flat.reshape(1, -1))[0]
            
            # Reshape to (n_units, 2) for distance computation
            eq_reshaped = np.stack([eq_scaled[:n_units], eq_scaled[n_units:]], axis=1)
            
            equilibria_by_class_scaled[class_label].append((idx, eq_reshaped))
    
    # Check how many equilibria we have per class
    n_eq_per_class = {c: len(equilibria_by_class_scaled[c]) for c in CLASSES_TO_ANALYZE}
    print(f"\nEquilibria found per class: {n_eq_per_class}")
    
    # Only proceed if we have enough equilibria
    min_samples = 3
    classes_with_enough = [c for c in CLASSES_TO_ANALYZE if n_eq_per_class[c] >= min_samples]
    
    if len(classes_with_enough) >= 2:
        # Use only top units by classifier weight (filtered to exclude unit 0 which is fixed)
        top_units_filtered = [u for u in top_units if u > 0]
        print(f"\nUsing top {len(top_units_filtered)} units (by classifier weight) for equilibrium comparison")
        
        # ============================================================
        # BASELINE: Expected distance for random points in scaled space
        # ============================================================
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
        
        print(f"\n--- BASELINE: Random point distances in scaled space ---")
        print(f"  Empirical (from training data): {random_baseline:.4f} ± {random_std:.4f}")
        
        # ============================================================
        # Compute within-class and between-class distances
        # ============================================================
        def compute_pairwise_distances(equilibria_list, units_to_use):
            """Compute pairwise distances between scaled equilibria positions."""
            n = len(equilibria_list)
            if n < 2:
                return []
            distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    p_i = equilibria_list[i][1]  # (n_units, 2) scaled positions
                    p_j = equilibria_list[j][1]  # (n_units, 2) scaled positions
                    dist = np.linalg.norm(p_i[units_to_use] - p_j[units_to_use])
                    distances.append(dist)
            return distances
        
        # Within-class distances
        within_class_distances = {}
        for c in classes_with_enough:
            within_class_distances[c] = compute_pairwise_distances(
                equilibria_by_class_scaled[c], top_units_filtered
            )
        
        # Between-class distances (all pairs of classes)
        between_class_distances = []
        between_class_by_pair = {}
        for i, c1 in enumerate(classes_with_enough):
            for c2 in classes_with_enough[i+1:]:
                pair_distances = []
                for _, p_1 in equilibria_by_class_scaled[c1]:
                    for _, p_2 in equilibria_by_class_scaled[c2]:
                        dist = np.linalg.norm(p_1[top_units_filtered] - p_2[top_units_filtered])
                        pair_distances.append(dist)
                        between_class_distances.append(dist)
                between_class_by_pair[(c1, c2)] = pair_distances
        
        # Compute summary statistics
        avg_within_per_class = {c: np.mean(within_class_distances[c]) if len(within_class_distances[c]) > 0 else np.nan 
                                for c in classes_with_enough}
        overall_avg_within = np.nanmean(list(avg_within_per_class.values()))
        avg_between = np.mean(between_class_distances) if len(between_class_distances) > 0 else np.nan
        
        print(f"\n--- EQUILIBRIA DISTANCES (scaled space, top units) ---")
        print(f"\nWithin-class distances:")
        for c in classes_with_enough:
            if len(within_class_distances[c]) > 0:
                print(f"  Class {c}: {np.mean(within_class_distances[c]):.4f} ± {np.std(within_class_distances[c]):.4f} (n={len(within_class_distances[c])} pairs)")
            else:
                print(f"  Class {c}: insufficient samples")
        
        print(f"\nOverall avg within-class: {overall_avg_within:.4f}")
        print(f"Between-class distances:  {avg_between:.4f} ± {np.std(between_class_distances):.4f} (n={len(between_class_distances)} pairs)")
        print(f"Random baseline:          {random_baseline:.4f} ± {random_std:.4f}")
        
        # Compare to baseline
        print(f"\n--- COMPARISON TO RANDOM BASELINE ---")
        print(f"  Avg Within / Random: {overall_avg_within / random_baseline:.2f}x")
        print(f"  Between / Random:    {avg_between / random_baseline:.2f}x")
        
        # Check if within-class < between-class (suggests class-specific equilibria)
        print(f"\n--- CLASS SEPARATION ANALYSIS ---")
        print(f"  Between-class avg: {avg_between:.4f}")
        print(f"  Avg within-class:  {overall_avg_within:.4f}")
        print(f"  Ratio (between/within): {avg_between/overall_avg_within:.2f}")
        
        n_classes_separated = sum(1 for c in classes_with_enough 
                                   if len(within_class_distances[c]) > 0 and 
                                   avg_between > np.mean(within_class_distances[c]))
        
        print(f"\n  Classes where between > within: {n_classes_separated}/{len(classes_with_enough)}")
        
        if avg_between > overall_avg_within:
            print(f"\n✓ OVERALL SEPARATION: Between-class > avg within-class")
            print(f"  → Classes tend to occupy distinct regions of equilibrium space")
        else:
            print(f"\n✗ NO OVERALL SEPARATION: Within-class distances ≥ between-class")
        
        # Permutation test
        print(f"\n--- PERMUTATION TEST ---")
        observed_diff = avg_between - overall_avg_within
        n_permutations = 1000
        perm_diffs = []
        
        # Combine all equilibria
        all_equilibria_combined = []
        class_sizes = []
        for c in classes_with_enough:
            all_equilibria_combined.extend(equilibria_by_class_scaled[c])
            class_sizes.append(len(equilibria_by_class_scaled[c]))
        
        for _ in range(n_permutations):
            # Randomly shuffle class assignments
            perm_indices = np.random.permutation(len(all_equilibria_combined))
            
            # Split into permuted classes
            perm_classes = []
            start_idx = 0
            for size in class_sizes:
                perm_classes.append([all_equilibria_combined[perm_indices[i]] for i in range(start_idx, start_idx + size)])
                start_idx += size
            
            # Compute within-class distances for permuted classes
            perm_within = []
            for pc in perm_classes:
                pw = compute_pairwise_distances(pc, top_units_filtered)
                if len(pw) > 0:
                    perm_within.append(np.mean(pw))
            
            # Compute between-class distances for permuted classes
            perm_between = []
            for i in range(len(perm_classes)):
                for j in range(i + 1, len(perm_classes)):
                    for _, p_1 in perm_classes[i]:
                        for _, p_2 in perm_classes[j]:
                            perm_between.append(np.linalg.norm(p_1[top_units_filtered] - p_2[top_units_filtered]))
            
            if len(perm_within) > 0 and len(perm_between) > 0:
                perm_avg_within = np.mean(perm_within)
                perm_avg_between = np.mean(perm_between)
                perm_diffs.append(perm_avg_between - perm_avg_within)
        
        p_value = np.mean([d >= observed_diff for d in perm_diffs])
        print(f"  Observed (between - within): {observed_diff:.4f}")
        print(f"  Permutation p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  → Significant class separation (p < 0.05)")
        else:
            print(f"  → No significant class separation (p >= 0.05)")
    
    else:
        print(f"\nInsufficient equilibria for cross-class comparison (need at least {min_samples} per class for at least 2 classes)")

    #%%
    # ============================================================
    # EQUILIBRIUM IMPORTANCE TEST
    # ============================================================
    print("\n" + "="*70)
    print("EQUILIBRIUM IMPORTANCE TEST")
    print("="*70)
    
    # Collect equilibrium positions for samples that have equilibria
    eq_positions_list = []
    eq_labels_list = []
    final_positions_with_eq = []
    
    for class_label in CLASSES_TO_ANALYZE:
        for info in equilibria_info_by_class[class_label]:
            if len(info['equilibria']) == 0:
                continue
            
            idx = info['idx']
            p_state = np.column_stack([all_x[idx], all_y[idx]])
            theta_state = all_theta[idx]
            e = np.stack([np.cos(theta_state), np.sin(theta_state)], axis=1)
            
            # Compute equilibrium positions and find closest one
            min_dist = np.inf
            closest_s_eq = None
            for s_eq in info['equilibria']:
                s_full = np.zeros(n_units)
                s_full[1:] = s_eq
                p_eq_candidate = p_state + s_full[:, None] * e
                dist = np.linalg.norm(p_state[1:] - p_eq_candidate[1:])
                if dist < min_dist:
                    min_dist = dist
                    closest_s_eq = s_eq
            
            # Get equilibrium position for closest
            s_full = np.zeros(n_units)
            s_full[1:] = closest_s_eq
            p_eq = p_state + s_full[:, None] * e
            
            # Flatten for classifier
            eq_flat = np.concatenate([p_eq[:, 0], p_eq[:, 1]])
            final_flat = np.concatenate([p_state[:, 0], p_state[:, 1]])
            
            eq_positions_list.append(eq_flat)
            final_positions_with_eq.append(final_flat)
            eq_labels_list.append(y_pred[idx])
    
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
        print(f"\nTest 2: Perturbation away from equilibrium")
        
        perturbation_magnitudes = [0.1, 0.5, 1.0, 2.0, 5.0]
        print(f"  Perturbing states in direction AWAY from equilibrium...")
        
        all_equilibria_info_flat = []
        for class_label in CLASSES_TO_ANALYZE:
            all_equilibria_info_flat.extend([inf for inf in equilibria_info_by_class[class_label] if len(inf['equilibria']) > 0])
        
        for pert_mag in perturbation_magnitudes:
            X_perturbed = []
            for i, info in enumerate(all_equilibria_info_flat):
                idx = info['idx']
                p_state = np.column_stack([all_x[idx], all_y[idx]])
                theta_state = all_theta[idx]
                e = np.stack([np.cos(theta_state), np.sin(theta_state)], axis=1)
                
                # Find closest equilibrium directly
                min_dist = np.inf
                closest_s_eq = None
                for s_eq in info['equilibria']:
                    s_full = np.zeros(n_units)
                    s_full[1:] = s_eq
                    p_eq_candidate = p_state + s_full[:, None] * e
                    dist = np.linalg.norm(p_state[1:] - p_eq_candidate[1:])
                    if dist < min_dist:
                        min_dist = dist
                        closest_s_eq = s_eq
                
                s_full = np.zeros(n_units)
                s_full[1:] = closest_s_eq
                p_eq = p_state + s_full[:, None] * e
                
                # Direction from equilibrium to final state (away from eq)
                direction = p_state - p_eq
                direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
                direction_norm[direction_norm == 0] = 1
                direction_unit = direction / direction_norm
                
                # Perturb away from equilibrium
                p_perturbed = p_state + pert_mag * direction_unit
                
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
    # VISUALIZATIONS
    # ============================================================
    
    # 1. Histogram of number of equilibria found per class
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    for class_label in CLASSES_TO_ANALYZE[:5]:  # First 5 classes
        n_eq = [info['n_equilibria'] for info in equilibria_info_by_class[class_label]]
        ax.hist(n_eq, bins=range(max(n_eq)+2) if n_eq else range(2), alpha=0.5, label=f'Class {class_label}')
    ax.set_xlabel('Number of equilibria found')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of equilibria count (Classes 0-4)')
    ax.legend()
    
    ax = axes[0, 1]
    for class_label in CLASSES_TO_ANALYZE[5:]:  # Last 5 classes
        n_eq = [info['n_equilibria'] for info in equilibria_info_by_class[class_label]]
        ax.hist(n_eq, bins=range(max(n_eq)+2) if n_eq else range(2), alpha=0.5, label=f'Class {class_label}')
    ax.set_xlabel('Number of equilibria found')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of equilibria count (Classes 5-9)')
    ax.legend()
    
    # 2. Average closest distance per class
    ax = axes[1, 0]
    means = []
    stds = []
    for class_label in CLASSES_TO_ANALYZE:
        dist = [info['closest_distance'] for info in equilibria_info_by_class[class_label] if info['closest_distance'] < np.inf]
        if len(dist) > 0:
            means.append(np.mean(dist))
            stds.append(np.std(dist))
        else:
            means.append(0)
            stds.append(0)
    
    ax.bar(CLASSES_TO_ANALYZE, means, yerr=stds, capsize=3, alpha=0.7)
    ax.set_xlabel('Class')
    ax.set_ylabel('Mean closest distance to equilibrium')
    ax.set_title('Distance to nearest equilibrium by class')
    ax.set_xticks(CLASSES_TO_ANALYZE)
    
    # 3. Unit importance (classifier weights)
    ax = axes[1, 1]
    colors = ['green' if i in top_units else 'gray' for i in range(n_units)]
    ax.bar(range(n_units), unit_weight_magnitude, color=colors, alpha=0.7)
    ax.set_xlabel('Unit index')
    ax.set_ylabel('Classifier weight magnitude')
    ax.set_title(f'Unit importance (green = top {n_top})')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{parent_dir}/plots/mnist_equilibrium_analysis.png', dpi=150)
    plt.show()
    
    print("\nAnalysis complete!")
    return equilibria_info_by_class, all_x, all_y, all_theta, all_labels, clf, scaler

#%%
if __name__ == "__main__":
    results = main()

# %%
