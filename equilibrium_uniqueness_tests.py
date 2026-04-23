"""
Test uniqueness conditions for equilibria in unicycle networks.

This script tests two key conditions for equilibrium uniqueness:

1. DIAGONAL DOMINANCE: |J_ii(s)| > Σ_{j≠i} |J_ij(s)| for all s
   - If this holds, the Jacobian is nonsingular everywhere
   - Can be checked locally (at equilibria) or globally (on a grid)

2. HADAMARD-LEVY INJECTIVITY:
   - J(s) nonsingular everywhere
   - F(s) → ∞ as ||s|| → ∞ (coercivity)
   - If both hold, F is globally injective → at most one equilibrium

The script:
1. Loads parameters from Optuna or uses random parameters
2. Finds equilibria numerically
3. Checks diagonal dominance at equilibria and on a grid
4. Checks Jacobian nonsingularity on a grid
5. Estimates coercivity by checking F behavior at large ||s||
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.linalg import eigvals
import optuna
import torch
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

# Source of parameters
USE_OPTUNA = True  # If True, load from Optuna; if False, use random
DATABASE_NAME = "unicycle_nets_forda_logreg"
STUDY_NAME = "only_last_state_as_readout_20_units_not_aligned_less_ang_magnitude_no_tanh"
TRIAL_RANK = 1

# Network parameters (used if USE_OPTUNA = False)
N = 6  # number of unicycles
SEED = 41

# Multi-seed analysis
N_SEEDS = 5  # Number of different p0/theta randomizations to try
SEED_LIST = list(range(N_SEEDS))  # Seeds to use: [0, 1, 2, ..., N_SEEDS-1]

# Analysis parameters
N_EQUILIBRIA_TRIALS = 5000  # Trials to find equilibria
GRID_POINTS = 20  # Points per dimension for grid analysis
GRID_RANGE = 20.0  # Range for s values: [-GRID_RANGE, GRID_RANGE]
COERCIVITY_TEST_RADIUS = [1, 2, 5, 10, 20, 50]  # Radii to test coercivity

# ============================================================
# SETUP NETWORK PARAMETERS
# ============================================================

def setup_random_network(N, seed=42):
    """Setup a random unicycle network."""
    np.random.seed(seed)
    
    # Fixed headings (radians)
    theta = np.random.uniform(0, 2*np.pi, N)
    e = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    
    # Reference points
    p0 = np.random.uniform(-1, 1, (N, 2))
    
    # Stiffness and rest-length matrices
    num_connections = N*(N-1)//2
    stiffnesses_array = np.random.rand(num_connections) * 2 + 0.5  # K in [0.5, 2.5]
    restlengths_array = np.random.rand(num_connections) * 1 + 0.5  # A in [0.5, 1.5]
    
    K_mat = np.zeros((N, N))
    A_mat = np.zeros((N, N))
    
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            K_mat[i, j] = stiffnesses_array[idx]
            K_mat[j, i] = stiffnesses_array[idx]
            A_mat[i, j] = restlengths_array[idx]
            A_mat[j, i] = restlengths_array[idx]
            idx += 1
    
    return {
        'N': N,
        'theta': theta,
        'e': e,
        'p0': p0,
        'K_mat': K_mat,
        'A_mat': A_mat,
    }

def setup_network_from_optuna(database_name, study_name, rank=1, seed=42, verbose=True):
    """Load network parameters from Optuna trial.
    
    Args:
        database_name: Name of the Optuna database
        study_name: Name of the study
        rank: Rank of the trial (1 = best)
        seed: Random seed for p0 and theta initialization
        verbose: Whether to print loading info
    """
    from unicycle_network_class import UnicycleNetwork
    
    storage_name = f"sqlite:///optuna_databases/{database_name}.db"
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    
    trials = study.trials
    trials_sorted = sorted([t for t in trials if t.state == optuna.trial.TrialState.COMPLETE],
                          key=lambda t: t.value, reverse=(study.direction == optuna.study.StudyDirection.MAXIMIZE))
    trial = trials_sorted[rank - 1]
    params = trial.params
    
    # Network structure
    n_units = params.get('n_units', 20)
    aligned_orientations = params.get('aligned_orientations', False)
    ang_input = params.get('ang_input', False) if not aligned_orientations else False
    ang_connections = params.get('ang_connections', True) if not aligned_orientations else False
    
    # Linear coupling parameters
    lin_stiff_min = params['lin_stiff_min']
    lin_stiff_max = params['lin_stiff_max']
    eq_dist_min = params['eq_dist_min']
    eq_dist_max = params['eq_dist_max']
    
    # Connection topology
    n_connections_fraction = params['n_connections_fraction']
    n_connections = int(n_units * n_connections_fraction)
    anchor_con_fraction = params['anchor_con_fraction']
    n_connections_anchor = int(n_units * anchor_con_fraction)
    
    # Angular coupling parameters (if enabled)
    if ang_connections:
        ang_stiff_min = params['ang_stiff_min']
        ang_stiff_max = params['ang_stiff_max']
        eq_dist_min_ang = params['eq_dist_min_ang']
        eq_dist_max_ang = params['eq_dist_max_ang']
        n_connections_ang_fraction = params['n_connections_ang_fraction']
        n_connections_ang = int(n_units * n_connections_ang_fraction)
        anchor_con_fraction_ang = params['anchor_con_fraction_ang']
        n_connections_anchor_ang = int(n_units * anchor_con_fraction_ang)
    else:
        ang_stiff_min = ang_stiff_max = 0.1
        eq_dist_min_ang = 0.0
        eq_dist_max_ang = np.pi
        n_connections_ang = 0
        n_connections_anchor_ang = 0
    
    # Create network to get matrices
    lin_input_map = torch.zeros(1, n_units)
    ang_input_map = torch.zeros(1, n_units)
    
    network = UnicycleNetwork(
        n_inp=1,
        n_units=n_units,
        dt=0.01,
        lin_stiff_min=lin_stiff_min,
        lin_stiff_max=lin_stiff_max,
        ang_stiff_min=ang_stiff_min,
        ang_stiff_max=ang_stiff_max,
        eq_dist_min=eq_dist_min,
        eq_dist_max=eq_dist_max,
        eq_dist_min_ang=eq_dist_min_ang,
        eq_dist_max_ang=eq_dist_max_ang,
        lin_input_map=lin_input_map,
        ang_input_map=ang_input_map,
        n_connections=n_connections,
        n_connections_anchor=n_connections_anchor,
        n_connections_ang=n_connections_ang,
        n_connections_anchor_ang=n_connections_anchor_ang,
    )
    
    K_mat = network.stiffness_coupling_matrix.detach().cpu().numpy()
    A_mat = network.eq_distances_matrix.detach().cpu().numpy().squeeze(-1)
    
    # Random orientations and positions (using provided seed)
    np.random.seed(seed)
    theta = np.random.uniform(0, 2*np.pi, n_units)
    theta[0] = 0
    e = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    p0 = np.random.uniform(-1, 1, (n_units, 2))
    
    if verbose:
        print(f"Loaded trial #{trial.number} (rank {rank}), objective: {trial.value:.6f}, seed: {seed}")
    
    return {
        'N': n_units,
        'theta': theta,
        'e': e,
        'p0': p0,
        'K_mat': K_mat,
        'A_mat': A_mat,
        'trial_number': trial.number,
        'trial_value': trial.value,
    }

# ============================================================
# EQUILIBRIUM EQUATIONS AND JACOBIAN
# ============================================================

def make_equilibrium_functions(params):
    """Create equilibrium function F and its Jacobian for given parameters."""
    N = params['N']
    e = params['e']
    p0 = params['p0']
    K_mat = params['K_mat']
    A_mat = params['A_mat']
    
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
    
    def jacobian_numerical(s, eps=1e-7):
        """Compute Jacobian numerically using central differences."""
        n = len(s)
        J = np.zeros((n, n))
        for i in range(n):
            ds = np.zeros(n)
            ds[i] = eps
            J[:, i] = (F(s + ds) - F(s - ds)) / (2 * eps)
        return J
    
    return F, jacobian_numerical, positions

# ============================================================
# FIND EQUILIBRIA
# ============================================================

def find_equilibria(F, N, n_trials=5000, tol=1e-8, unique_tol=1e-4):
    """Find equilibria by trying many random initial conditions."""
    solutions = []
    
    for _ in tqdm(range(n_trials), desc="Finding equilibria"):
        s0 = np.random.randn(N - 1) * 10
        
        try:
            sol = root(F, s0, method='hybr')
            if sol.success and np.linalg.norm(sol.fun) < tol:
                solutions.append(sol.x)
        except:
            continue
    
    if len(solutions) == 0:
        return []
    
    # Remove duplicates
    unique = []
    for s in solutions:
        if not any(np.linalg.norm(s - u) < unique_tol for u in unique):
            unique.append(s)
    
    return unique

# ============================================================
# DIAGONAL DOMINANCE CHECK
# ============================================================

def check_diagonal_dominance(J):
    """
    Check if matrix J is diagonally dominant.
    Returns (is_dominant, margin) where margin > 0 means dominant.
    """
    n = J.shape[0]
    margins = np.zeros(n)
    
    for i in range(n):
        diag = np.abs(J[i, i])
        off_diag_sum = np.sum(np.abs(J[i, :])) - diag
        margins[i] = diag - off_diag_sum
    
    is_dominant = np.all(margins > 0)
    min_margin = np.min(margins)
    
    return is_dominant, min_margin, margins

def check_diagonal_dominance_at_points(jacobian_fn, points, desc="Checking"):
    """Check diagonal dominance at multiple points."""
    results = []
    
    for s in tqdm(points, desc=desc, leave=False):
        J = jacobian_fn(s)
        is_dom, min_margin, margins = check_diagonal_dominance(J)
        results.append({
            's': s,
            'is_dominant': is_dom,
            'min_margin': min_margin,
            'margins': margins,
        })
    
    return results

# ============================================================
# JACOBIAN NONSINGULARITY CHECK
# ============================================================

def check_jacobian_nonsingular(J, tol=1e-6):
    """
    Check if Jacobian is nonsingular.
    Returns (is_nonsingular, min_singular_value, condition_number)
    
    Note: Using tol=1e-6 as threshold for practical nonsingularity,
    since numerical errors can make truly singular matrices appear
    to have tiny singular values.
    """
    try:
        U, s, Vh = np.linalg.svd(J)
        min_sv = np.min(s)
        max_sv = np.max(s)
        cond = max_sv / min_sv if min_sv > 0 else np.inf
        
        # Use stricter tolerance - if min singular value is very small,
        # the matrix is effectively singular
        is_nonsingular = min_sv > tol
        
        return is_nonsingular, min_sv, cond
    except:
        return False, 0, np.inf

def check_jacobian_eigenvalues(J):
    """
    Analyze Jacobian eigenvalues.
    Returns eigenvalues and stability info.
    """
    eigs = eigvals(J)
    real_parts = np.real(eigs)
    
    return {
        'eigenvalues': eigs,
        'real_parts': real_parts,
        'all_negative': np.all(real_parts < 0),
        'all_positive': np.all(real_parts > 0),
        'has_zero': np.any(np.abs(eigs) < 1e-10),
        'max_real': np.max(real_parts),
        'min_real': np.min(real_parts),
    }

# ============================================================
# COERCIVITY CHECK (F → ∞ as ||s|| → ∞)
# ============================================================

def check_coercivity(F, N, radii=[1, 2, 5, 10, 20], n_samples=100):
    """
    Check coercivity: ||F(s)|| should grow as ||s|| → ∞.
    Tests at multiple radii with random directions.
    """
    results = []
    
    for r in radii:
        norms = []
        for _ in range(n_samples):
            # Random point on sphere of radius r
            direction = np.random.randn(N - 1)
            direction = direction / np.linalg.norm(direction)
            s = r * direction
            
            F_val = F(s)
            norms.append(np.linalg.norm(F_val))
        
        results.append({
            'radius': r,
            'mean_F_norm': np.mean(norms),
            'min_F_norm': np.min(norms),
            'max_F_norm': np.max(norms),
        })
    
    # Check if F norm is increasing with radius
    mean_norms = [r['mean_F_norm'] for r in results]
    is_coercive = all(mean_norms[i] <= mean_norms[i+1] for i in range(len(mean_norms)-1))
    
    return results, is_coercive

# ============================================================
# GRID ANALYSIS
# ============================================================

def analyze_on_grid(jacobian_fn, N, grid_points=40, grid_range=20):
    """
    Analyze Jacobian properties on a grid (for small N only).
    For N > 3, uses random sampling instead.
    """
    dim = N - 1
    
    if dim <= 2:
        # Full grid for 2D
        s_vals = np.linspace(-grid_range, grid_range, grid_points)
        if dim == 1:
            points = [[s] for s in s_vals]
        else:
            points = [[s1, s2] for s1 in s_vals for s2 in s_vals]
    else:
        # Random sampling for higher dimensions
        n_samples = grid_points ** min(dim, 3)
        points = [np.random.uniform(-grid_range, grid_range, dim) for _ in range(n_samples)]
    
    # Check properties at each point
    diag_dom_count = 0
    nonsingular_count = 0
    min_sv_global = np.inf
    max_cond_global = 0
    
    for s in tqdm(points, desc="Grid analysis"):
        s = np.array(s)
        J = jacobian_fn(s)
        
        is_dom, _, _ = check_diagonal_dominance(J)
        is_nonsing, min_sv, cond = check_jacobian_nonsingular(J)
        
        if is_dom:
            diag_dom_count += 1
        if is_nonsing:
            nonsingular_count += 1
        
        min_sv_global = min(min_sv_global, min_sv)
        if cond < np.inf:
            max_cond_global = max(max_cond_global, cond)
    
    return {
        'n_points': len(points),
        'diag_dom_fraction': diag_dom_count / len(points),
        'nonsingular_fraction': nonsingular_count / len(points),
        'min_singular_value': min_sv_global,
        'max_condition_number': max_cond_global,
    }

# ============================================================
# SEARCH BETWEEN EQUILIBRIA
# ============================================================

def search_singularities_between_equilibria(jacobian_fn, equilibria, n_points=100):
    """
    If multiple equilibria exist, search along paths between them
    for singular Jacobians (which must exist by Hadamard-Levy).
    
    The idea: if we have two equilibria s1 and s2, and the Jacobian
    were nonsingular everywhere along the path, then by the implicit
    function theorem, the map F would be locally invertible everywhere,
    contradicting the existence of two distinct roots.
    """
    if len(equilibria) < 2:
        return None
    
    results = []
    
    # Check all pairs of equilibria
    for i in range(len(equilibria)):
        for j in range(i + 1, len(equilibria)):
            s1 = equilibria[i]
            s2 = equilibria[j]
            
            # Sample along the line segment between s1 and s2
            min_sv_path = np.inf
            worst_t = None
            worst_point = None
            worst_cond = 0
            
            for k in range(n_points + 1):
                t = k / n_points
                s = (1 - t) * s1 + t * s2
                
                J = jacobian_fn(s)
                is_nonsing, min_sv, cond = check_jacobian_nonsingular(J)
                
                if min_sv < min_sv_path:
                    min_sv_path = min_sv
                    worst_t = t
                    worst_point = s.copy()
                    worst_cond = cond
            
            results.append({
                'eq_pair': (i, j),
                's1': s1,
                's2': s2,
                'distance': np.linalg.norm(s2 - s1),
                'min_singular_value': min_sv_path,
                'worst_t': worst_t,  # parameter value where sv is smallest
                'worst_point': worst_point,
                'worst_condition_number': worst_cond,
            })
    
    return results

# ============================================================
# SINGLE SEED ANALYSIS (core analysis for one p0/theta config)
# ============================================================

def analyze_single_seed(params, n_equilibria_trials=5000, grid_points=20, grid_range=5.0,
                        coercivity_radii=[1, 2, 5, 10, 20, 50], verbose=True):
    """
    Run the full uniqueness analysis for a single p0/theta configuration.
    
    Returns a dictionary with all results.
    """
    N_units = params['N']
    
    # Create functions
    F, jacobian_fn, positions_fn = make_equilibrium_functions(params)
    
    # 1. Find equilibria
    equilibria = find_equilibria(F, N_units, n_trials=n_equilibria_trials)
    n_equilibria = len(equilibria)
    
    if n_equilibria == 0:
        return {
            'n_equilibria': 0,
            'equilibria': [],
            'grid_results': None,
            'coercivity_results': None,
            'is_coercive': None,
            'between_eq_results': None,
            'stable_equilibria': 0,
            'saddle_points': 0,
            'unstable_equilibria': 0,
        }
    
    # 2. Analyze stability at equilibria
    stable_count = 0
    saddle_count = 0
    unstable_count = 0
    
    for s_eq in equilibria:
        J = jacobian_fn(s_eq)
        eig_info = check_jacobian_eigenvalues(J)
        
        if eig_info['all_negative']:
            stable_count += 1
        elif eig_info['all_positive']:
            unstable_count += 1
        else:
            saddle_count += 1
    
    # 3. Grid analysis
    grid_results = analyze_on_grid(jacobian_fn, N_units, grid_points, grid_range)
    
    # 4. Coercivity check
    coercivity_results, is_coercive = check_coercivity(F, N_units, coercivity_radii)
    
    # 5. Search between equilibria (if multiple)
    between_eq_results = None
    if n_equilibria > 1:
        between_eq_results = search_singularities_between_equilibria(
            jacobian_fn, equilibria, n_points=200
        )
    
    return {
        'n_equilibria': n_equilibria,
        'equilibria': equilibria,
        'grid_results': grid_results,
        'coercivity_results': coercivity_results,
        'is_coercive': is_coercive,
        'between_eq_results': between_eq_results,
        'stable_equilibria': stable_count,
        'saddle_points': saddle_count,
        'unstable_equilibria': unstable_count,
    }

# ============================================================
# MULTI-SEED ANALYSIS
# ============================================================

def run_multi_seed_analysis(database_name=None, study_name=None, trial_rank=1,
                            n_units=None, seeds=None, use_optuna=True,
                            n_equilibria_trials=5000, grid_points=20, grid_range=5.0,
                            coercivity_radii=[1, 2, 5, 10, 20, 50]):
    """
    Run uniqueness analysis across multiple random seeds for p0/theta.
    
    This helps understand how the equilibrium structure depends on the
    initial configuration (positions and orientations).
    
    Args:
        database_name: Optuna database name (if use_optuna=True)
        study_name: Optuna study name (if use_optuna=True)  
        trial_rank: Which trial to use (1=best)
        n_units: Number of unicycles (if use_optuna=False)
        seeds: List of seeds to use for p0/theta randomization
        use_optuna: Whether to load from Optuna or use random network
        n_equilibria_trials: Trials per seed for finding equilibria
        grid_points: Grid resolution for Jacobian analysis
        grid_range: Range for grid sampling
        coercivity_radii: Radii to test coercivity
    
    Returns:
        Dictionary with aggregated results across all seeds
    """
    if seeds is None:
        seeds = list(range(5))
    
    print("="*70)
    print("MULTI-SEED EQUILIBRIUM UNIQUENESS ANALYSIS")
    print("="*70)
    print(f"\nRunning analysis with {len(seeds)} different p0/theta seeds")
    print(f"Seeds: {seeds}")
    
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed} ({i+1}/{len(seeds)})")
        print("="*70)
        
        # Setup network with this seed
        if use_optuna:
            params = setup_network_from_optuna(
                database_name, study_name, trial_rank, seed=seed, verbose=True
            )
        else:
            params = setup_random_network(n_units, seed=seed)
        
        # Run analysis
        results = analyze_single_seed(
            params,
            n_equilibria_trials=n_equilibria_trials,
            grid_points=grid_points,
            grid_range=grid_range,
            coercivity_radii=coercivity_radii,
            verbose=True
        )
        results['seed'] = seed
        all_results.append(results)
        
        # Print summary for this seed
        print(f"\n  Equilibria found: {results['n_equilibria']}")
        print(f"  Stable: {results['stable_equilibria']}, "
              f"Saddle: {results['saddle_points']}, "
              f"Unstable: {results['unstable_equilibria']}")
        if results['grid_results']:
            print(f"  Nonsingular fraction: {results['grid_results']['nonsingular_fraction']*100:.1f}%")
            print(f"  Min singular value: {results['grid_results']['min_singular_value']:.2e}")
        print(f"  Coercive: {results['is_coercive']}")
    
    # Aggregate statistics
    n_eq_list = [r['n_equilibria'] for r in all_results]
    stable_list = [r['stable_equilibria'] for r in all_results]
    
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS ACROSS ALL SEEDS")
    print("="*70)
    
    print(f"\nEquilibria count statistics:")
    print(f"  Mean: {np.mean(n_eq_list):.2f}")
    print(f"  Std:  {np.std(n_eq_list):.2f}")
    print(f"  Min:  {np.min(n_eq_list)}")
    print(f"  Max:  {np.max(n_eq_list)}")
    print(f"  All:  {n_eq_list}")
    
    print(f"\nStable equilibria statistics:")
    print(f"  Mean: {np.mean(stable_list):.2f}")
    print(f"  All:  {stable_list}")
    
    # Check if uniqueness ever holds
    unique_seeds = [r['seed'] for r in all_results if r['n_equilibria'] == 1]
    multiple_seeds = [r['seed'] for r in all_results if r['n_equilibria'] > 1]
    
    print(f"\nUniqueness assessment:")
    print(f"  Seeds with unique equilibrium: {len(unique_seeds)} ({unique_seeds})")
    print(f"  Seeds with multiple equilibria: {len(multiple_seeds)} ({multiple_seeds})")
    
    if len(multiple_seeds) > 0:
        print(f"\n  → Multiple equilibria found in {len(multiple_seeds)}/{len(seeds)} seeds")
        print(f"  → Hadamard-Levy conditions are NOT satisfied for this network")
    elif len(unique_seeds) == len(seeds):
        print(f"\n  → Only one equilibrium found across all {len(seeds)} seeds")
        print(f"  → Evidence suggests uniqueness may hold (but not proven)")
    
    return {
        'all_results': all_results,
        'n_equilibria_list': n_eq_list,
        'stable_list': stable_list,
        'mean_equilibria': np.mean(n_eq_list),
        'std_equilibria': np.std(n_eq_list),
        'unique_seeds': unique_seeds,
        'multiple_seeds': multiple_seeds,
    }

# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    print("="*70)
    print("EQUILIBRIUM UNIQUENESS ANALYSIS")
    print("="*70)
    
    # Setup network
    if USE_OPTUNA:
        params = setup_network_from_optuna(DATABASE_NAME, STUDY_NAME, TRIAL_RANK)
    else:
        params = setup_random_network(N, SEED)
    
    N_units = params['N']
    print(f"\nNetwork: {N_units} unicycles ({N_units - 1} free parameters)")
    print(f"K_mat range: [{params['K_mat'].min():.3f}, {params['K_mat'].max():.3f}]")
    print(f"A_mat range: [{params['A_mat'].min():.3f}, {params['A_mat'].max():.3f}]")
    
    # Create functions
    F, jacobian_fn, positions_fn = make_equilibrium_functions(params)
    
    # ============================================================
    # 1. FIND EQUILIBRIA
    # ============================================================
    print(f"\n{'='*70}")
    print("1. FINDING EQUILIBRIA")
    print("="*70)
    
    equilibria = find_equilibria(F, N_units, n_trials=N_EQUILIBRIA_TRIALS)
    print(f"\nFound {len(equilibria)} unique equilibria")
    
    if len(equilibria) == 0:
        print("No equilibria found! Cannot continue analysis.")
        return
    
    if len(equilibria) == 1:
        print("✓ Only ONE equilibrium found - system may have unique equilibrium")
    else:
        print(f"✗ Multiple equilibria ({len(equilibria)}) - uniqueness conditions likely fail")
    
    # ============================================================
    # 2. ANALYZE JACOBIAN AT EQUILIBRIA
    # ============================================================
    print(f"\n{'='*70}")
    print("2. JACOBIAN ANALYSIS AT EQUILIBRIA")
    print("="*70)
    
    for i, s_eq in enumerate(equilibria):
        print(f"\n--- Equilibrium {i+1} ---")
        print(f"s* = {s_eq}")
        print(f"||F(s*)|| = {np.linalg.norm(F(s_eq)):.2e}")
        
        J = jacobian_fn(s_eq)
        
        # Diagonal dominance
        is_dom, min_margin, margins = check_diagonal_dominance(J)
        print(f"\nDiagonal dominance:")
        print(f"  Is dominant: {is_dom}")
        print(f"  Min margin: {min_margin:.4f}")
        if not is_dom:
            print(f"  Rows failing: {np.where(margins <= 0)[0]}")
        
        # Nonsingularity
        is_nonsing, min_sv, cond = check_jacobian_nonsingular(J)
        print(f"\nNonsingularity:")
        print(f"  Is nonsingular: {is_nonsing}")
        print(f"  Min singular value: {min_sv:.2e}")
        print(f"  Condition number: {cond:.2e}")
        
        # Eigenvalues (stability)
        eig_info = check_jacobian_eigenvalues(J)
        print(f"\nStability (eigenvalues):")
        print(f"  Real parts: [{eig_info['min_real']:.4f}, {eig_info['max_real']:.4f}]")
        if eig_info['all_negative']:
            print(f"  ✓ All eigenvalues have negative real part → STABLE")
        elif eig_info['all_positive']:
            print(f"  All eigenvalues have positive real part → UNSTABLE")
        else:
            print(f"  Mixed eigenvalues → SADDLE POINT")
    
    # ============================================================
    # 3. GLOBAL ANALYSIS (Grid/Random sampling)
    # ============================================================
    print(f"\n{'='*70}")
    print("3. GLOBAL JACOBIAN ANALYSIS")
    print("="*70)
    
    grid_results = analyze_on_grid(jacobian_fn, N_units, GRID_POINTS, GRID_RANGE)
    
    print(f"\nAnalyzed {grid_results['n_points']} points in [{-GRID_RANGE}, {GRID_RANGE}]^{N_units-1}")
    print(f"  Diagonally dominant: {grid_results['diag_dom_fraction']*100:.1f}%")
    print(f"  Nonsingular: {grid_results['nonsingular_fraction']*100:.1f}%")
    print(f"  Min singular value (global): {grid_results['min_singular_value']:.2e}")
    print(f"  Max condition number: {grid_results['max_condition_number']:.2e}")
    
    if grid_results['nonsingular_fraction'] == 1.0:
        print("\n✓ Jacobian appears nonsingular everywhere in sampled region")
    else:
        print(f"\n✗ Jacobian singular at {(1-grid_results['nonsingular_fraction'])*100:.1f}% of sampled points")
    
    # ============================================================
    # 4. COERCIVITY CHECK
    # ============================================================
    print(f"\n{'='*70}")
    print("4. COERCIVITY CHECK (||F(s)|| → ∞ as ||s|| → ∞)")
    print("="*70)
    
    coercivity_results, is_coercive = check_coercivity(F, N_units, COERCIVITY_TEST_RADIUS)
    
    print(f"\n{'Radius':<10} {'Mean ||F||':<15} {'Min ||F||':<15} {'Max ||F||':<15}")
    print("-" * 55)
    for r in coercivity_results:
        print(f"{r['radius']:<10} {r['mean_F_norm']:<15.4f} {r['min_F_norm']:<15.4f} {r['max_F_norm']:<15.4f}")
    
    if is_coercive:
        print("\n✓ F appears coercive (||F|| increases with ||s||)")
    else:
        print("\n✗ F may not be coercive (||F|| not monotonically increasing)")
    
    # ============================================================
    # 4.5. SEARCH BETWEEN EQUILIBRIA (if multiple exist)
    # ============================================================
    between_eq_results = None
    if len(equilibria) > 1:
        print(f"\n{'='*70}")
        print("4.5. SEARCHING FOR SINGULARITIES BETWEEN EQUILIBRIA")
        print("="*70)
        print("\n(If multiple equilibria exist, there MUST be singularities between them)")
        
        between_eq_results = search_singularities_between_equilibria(
            jacobian_fn, equilibria, n_points=200
        )
        
        print(f"\n{'Pair':<10} {'Distance':<12} {'Min SV':<15} {'Worst t':<10} {'Condition':<15}")
        print("-" * 62)
        for r in between_eq_results:
            i, j = r['eq_pair']
            print(f"({i+1},{j+1}){'':<5} {r['distance']:<12.4f} {r['min_singular_value']:<15.2e} "
                  f"{r['worst_t']:<10.3f} {r['worst_condition_number']:<15.2e}")
        
        # Find the overall minimum singular value
        overall_min_sv = min(r['min_singular_value'] for r in between_eq_results)
        print(f"\nSmallest singular value found between equilibria: {overall_min_sv:.2e}")
        
        if overall_min_sv < 1e-6:
            print("→ Near-singularity detected (min SV < 1e-6)")
        elif overall_min_sv < 1e-3:
            print("→ Warning: Jacobian nearly singular (min SV < 1e-3)")
        else:
            print("→ Note: Singularities may exist off the straight-line paths")
    
    # ============================================================
    # 5. HADAMARD-LEVY CONCLUSION
    # ============================================================
    print(f"\n{'='*70}")
    print("5. HADAMARD-LEVY UNIQUENESS TEST")
    print("="*70)
    
    jacobian_sampled_ok = grid_results['nonsingular_fraction'] == 1.0
    coercivity_ok = is_coercive
    multiple_equilibria = len(equilibria) > 1
    
    print(f"\nConditions for at most one equilibrium (Hadamard-Levy):")
    print(f"  1. F is C¹: ✓ (by construction)")
    print(f"  2. J(s) nonsingular everywhere: {'?' if jacobian_sampled_ok else '✗'} (sampled only)")
    print(f"  3. F coercive (F → ∞ as ||s|| → ∞): {'✓' if coercivity_ok else '✗'}")
    print(f"\nEmpirical check:")
    print(f"  Number of equilibria found: {len(equilibria)}")
    
    print(f"\n{'='*70}")
    
    if multiple_equilibria:
        # If we found multiple equilibria, Hadamard-Levy conditions CANNOT be satisfied
        print("✗ MULTIPLE EQUILIBRIA FOUND")
        print(f"\n  → {len(equilibria)} equilibria were found numerically")
        print(f"  → By Hadamard-Levy theorem, this PROVES that the Jacobian")
        print(f"     must be singular somewhere (even if not sampled)")
        print(f"  → Min singular value in grid sample: {grid_results['min_singular_value']:.2e}")
        if between_eq_results:
            overall_min_sv = min(r['min_singular_value'] for r in between_eq_results)
            print(f"  → Min singular value between equilibria: {overall_min_sv:.2e}")
        print(f"\n  CONCLUSION: Uniqueness conditions are NOT satisfied")
    elif jacobian_sampled_ok and coercivity_ok:
        print("✓ HADAMARD-LEVY CONDITIONS APPEAR SATISFIED")
        print(f"\n  → Only ONE equilibrium found")
        print(f"  → Jacobian nonsingular at all {grid_results['n_points']} sampled points")
        print(f"  → F appears coercive")
        print(f"\n  NOTE: Sampling cannot prove global nonsingularity,")
        print(f"        but finding only one equilibrium is consistent")
        print(f"        with the theorem holding.")
    else:
        print("✗ HADAMARD-LEVY CONDITIONS NOT VERIFIED")
        if not jacobian_sampled_ok:
            print(f"\n  → Singular Jacobians found in sample")
        if not coercivity_ok:
            print(f"\n  → F does not appear coercive")
    
    print(f"{'='*70}")
    
    return {
        'equilibria': equilibria,
        'grid_results': grid_results,
        'coercivity_results': coercivity_results,
        'is_coercive': is_coercive,
        'between_equilibria_results': between_eq_results,
    }

if __name__ == "__main__":
    # Choose whether to run single seed or multi-seed analysis
    RUN_MULTI_SEED = True  # Set to False for single-seed detailed analysis
    
    if RUN_MULTI_SEED:
        # Multi-seed analysis
        results = run_multi_seed_analysis(
            database_name=DATABASE_NAME,
            study_name=STUDY_NAME,
            trial_rank=TRIAL_RANK,
            n_units=N,
            seeds=SEED_LIST,
            use_optuna=USE_OPTUNA,
            n_equilibria_trials=N_EQUILIBRIA_TRIALS,
            grid_points=GRID_POINTS,
            grid_range=GRID_RANGE,
            coercivity_radii=COERCIVITY_TEST_RADIUS,
        )
    else:
        # Single seed detailed analysis (original behavior)
        results = main()

