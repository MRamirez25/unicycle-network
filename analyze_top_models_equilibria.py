"""
Analyze equilibria for top performing models from Optuna study.

This script:
1. Loads top N trials from an Optuna study (e.g., FordA classification)
2. Extracts the unicycle network parameters for each trial
3. Computes the number of equilibria for each configuration
4. Saves results showing relationship between performance and equilibria count
"""

import numpy as np
import torch
import optuna
from scipy.optimize import root
from unicycle_network_class import UnicycleReservoir
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_NAME = "unicycle_nets_forda_logreg"
STUDY_NAME = "only_last_state_as_readout_20_units_not_aligned_less_ang_magnitude_no_tanh"
N_TOP_TRIALS = 5
N_TRIALS_EQUILIBRIA = 5000  # Number of random initial conditions to try per randomization
N_RANDOMIZATIONS = 5  # Number of different p0/theta initializations to try
EQUILIBRIA_TOL = 1e-6
UNIQUE_TOL = 1e-4

# ============================================================
# LOAD OPTUNA STUDY
# ============================================================

def load_top_trials(database_name, study_name, n_top=10):
    """Load top N trials from Optuna study."""
    storage_name = f"sqlite:///optuna_databases/{database_name}.db"
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    
    # Get trials sorted by objective value (best first)
    # For minimize: lower values are better, sorted puts them first
    # For maximize: higher values are better, need to reverse
    trials = study.trials
    trials_sorted = sorted([t for t in trials if t.state == optuna.trial.TrialState.COMPLETE],
                          key=lambda t: t.value,
                          reverse=(study.direction == optuna.study.StudyDirection.MAXIMIZE))
    
    return trials_sorted[:n_top], study

# ============================================================
# EXTRACT UNICYCLE PARAMETERS
# ============================================================

def extract_unicycle_params(trial):
    """Extract unicycle network parameters from an Optuna trial."""
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
    
    return {
        'n_units': n_units,
        'lin_stiff_min': lin_stiff_min,
        'lin_stiff_max': lin_stiff_max,
        'ang_stiff_min': ang_stiff_min,
        'ang_stiff_max': ang_stiff_max,
        'eq_dist_min': eq_dist_min,
        'eq_dist_max': eq_dist_max,
        'eq_dist_min_ang': eq_dist_min_ang,
        'eq_dist_max_ang': eq_dist_max_ang,
        'n_connections': n_connections,
        'n_connections_anchor': n_connections_anchor,
        'n_connections_ang': n_connections_ang,
        'n_connections_anchor_ang': n_connections_anchor_ang,
        'aligned_orientations': aligned_orientations,
        'ang_connections': ang_connections,
    }

# ============================================================
# CREATE UNICYCLE NETWORK INSTANCE
# ============================================================

def create_network_from_params(params):
    """Create a UnicycleNetwork instance from parameters (not full reservoir)."""
    from unicycle_network_class import UnicycleNetwork
    
    # Create dummy input maps (not needed for equilibria analysis)
    lin_input_map = torch.zeros(1, params['n_units'])
    ang_input_map = torch.zeros(1, params['n_units'])
    
    network = UnicycleNetwork(
        n_inp=1,
        n_units=params['n_units'],
        dt=0.01,  # dt doesn't matter for equilibria
        lin_stiff_min=params['lin_stiff_min'],
        lin_stiff_max=params['lin_stiff_max'],
        ang_stiff_min=params['ang_stiff_min'],
        ang_stiff_max=params['ang_stiff_max'],
        lin_damping_min=0.1,  # damping doesn't matter for equilibria
        lin_damping_max=0.2,
        ang_damping_min=0.1,
        ang_damping_max=0.2,
        eq_dist_min=params['eq_dist_min'],
        eq_dist_max=params['eq_dist_max'],
        eq_dist_min_ang=params['eq_dist_min_ang'],
        eq_dist_max_ang=params['eq_dist_max_ang'],
        lin_input_map=lin_input_map,
        ang_input_map=ang_input_map,
        n_connections=params['n_connections'],
        n_connections_anchor=params['n_connections_anchor'],
        n_connections_ang=params['n_connections_ang'],
        n_connections_anchor_ang=params['n_connections_anchor_ang'],
    )
    
    return network

# ============================================================
# EQUILIBRIA COMPUTATION
# ============================================================

def compute_equilibria(network, p0, theta, n_trials=5000, equilibria_tol=1e-6, unique_tol=1e-4):
    """
    Compute equilibria for a unicycle network with given p0 and theta.
    
    Args:
        network: UnicycleNetwork instance
        p0: Initial positions (n_units, 2)
        theta: Orientation angles (n_units,)
        n_trials: Number of random initial conditions to try
        equilibria_tol: Tolerance for accepting equilibria
        unique_tol: Tolerance for duplicate detection
    
    Returns:
        n_equilibria: Number of unique equilibria found
        equilibria: Array of equilibrium solutions
    """
    n_units = network.n_units
    
    # Extract network matrices (convert from torch to numpy)
    K_mat = network.stiffness_coupling_matrix.detach().cpu().numpy()  # (n_units, n_units)
    A_mat = network.eq_distances_matrix.detach().cpu().numpy().squeeze(-1)  # (n_units, n_units)
    
    # Use provided p0 and theta
    e = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # Unit direction vectors
    
    def positions(s):
        """Compute positions from line parameters s."""
        s_full = np.zeros(n_units)
        s_full[1:] = s  # First unicycle fixed at s=0
        return p0 + s_full[:, None] * e
    
    def equilibrium_equations(s):
        """Compute equilibrium equations (forces projected onto movement directions)."""
        p = positions(s)  # (N, 2)
        
        # Pairwise distance vectors: d[i,j] = p[i] - p[j]
        d = p[:, None, :] - p[None, :, :]  # (N, N, 2)
        
        # Distance magnitudes
        r = np.linalg.norm(d, axis=2) + np.eye(n_units)  # Add identity to avoid div by zero
        
        # Force contributions: K * (A - r) * (d · e_i) / r
        # einsum computes dot product between d[i,j] and e[i]
        F_contrib = K_mat * (A_mat - r) * np.einsum('ijk,ik->ij', d, e) / r
        
        # Sum over all neighbors
        F = np.sum(F_contrib, axis=1)
        
        # Return forces for unicycles 1..N-1 (unicycle 0 is fixed)
        return F[1:]
    
    # Find equilibria by trying many random initial conditions
    solutions = []
    
    for _ in tqdm(range(n_trials), desc="Finding equilibria", leave=False):
        s0 = np.random.randn(n_units - 1) * 10  # Random initial guess
        
        try:
            sol = root(equilibrium_equations, s0, method='hybr')
            
            if sol.success and np.linalg.norm(sol.fun) < equilibria_tol:
                solutions.append(sol.x)
        except:
            continue
    
    if len(solutions) == 0:
        return 0, []
    
    solutions = np.array(solutions)
    
    # Remove duplicate solutions
    unique = []
    for s in solutions:
        if not any(np.linalg.norm(s - u) < unique_tol for u in unique):
            unique.append(s)
    
    unique_eq = np.array(unique) if unique else np.array([])
    
    return len(unique_eq), unique_eq

def compute_equilibria_multiple_randomizations(network, n_randomizations=5, n_trials=5000, 
                                               equilibria_tol=1e-6, unique_tol=1e-4):
    """
    Compute equilibria for multiple random initializations of p0 and theta.
    
    Args:
        network: UnicycleNetwork instance
        n_randomizations: Number of different p0/theta initializations
        n_trials: Number of random initial conditions per randomization
        equilibria_tol: Tolerance for accepting equilibria
        unique_tol: Tolerance for duplicate detection
    
    Returns:
        mean_n_equilibria: Average number of equilibria across randomizations
        std_n_equilibria: Standard deviation of equilibria count
        all_counts: List of equilibria counts for each randomization
    """
    n_units = network.n_units
    equilibria_counts = []
    
    for i in tqdm(range(n_randomizations), desc="Randomizations", leave=False):
        # Generate random initial positions and orientations
        p0 = np.random.uniform(-1, 1, (n_units, 2))
        theta = np.random.uniform(-2*np.pi, 2*np.pi, n_units)
        theta[0] = 0  # Fix first unicycle orientation
        
        # Compute equilibria for this randomization
        n_eq, _ = compute_equilibria(
            network, p0, theta,
            n_trials=n_trials,
            equilibria_tol=equilibria_tol,
            unique_tol=unique_tol
        )
        equilibria_counts.append(n_eq)
    
    equilibria_counts = np.array(equilibria_counts)
    
    return equilibria_counts.mean(), equilibria_counts.std(), equilibria_counts.tolist()

# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    print(f"Loading top {N_TOP_TRIALS} trials from study: {STUDY_NAME}")
    
    top_trials, study = load_top_trials(DATABASE_NAME, STUDY_NAME, N_TOP_TRIALS)
    
    print(f"Found {len(top_trials)} trials")
    print(f"Best trial value: {top_trials[0].value:.6f}")
    print(f"Worst (of top {N_TOP_TRIALS}) trial value: {top_trials[-1].value:.6f}")
    
    results = []
    
    for i, trial in enumerate(top_trials):
        print(f"\n{'='*60}")
        print(f"Trial {i+1}/{len(top_trials)}: Trial #{trial.number}")
        print(f"Objective value: {trial.value:.6f}")
        
        # Extract parameters
        params = extract_unicycle_params(trial)
        print(f"n_units: {params['n_units']}")
        print(f"n_connections: {params['n_connections']}/{params['n_units']} "
              f"({params['n_connections']/params['n_units']*100:.0f}%)")
        print(f"Linear stiffness: [{params['lin_stiff_min']:.3f}, {params['lin_stiff_max']:.3f}]")
        print(f"Equilibrium distances: [{params['eq_dist_min']:.3f}, {params['eq_dist_max']:.3f}]")
        
        if params['ang_connections']:
            print(f"Angular connections: {params['n_connections_ang']}/{params['n_units']}")
            print(f"Angular stiffness: [{params['ang_stiff_min']:.3f}, {params['ang_stiff_max']:.3f}]")
        
        # Create network
        network = create_network_from_params(params)
        
        # Compute equilibria with multiple randomizations
        print(f"\nSearching for equilibria:")
        print(f"  {N_RANDOMIZATIONS} randomizations of p0/theta")
        print(f"  {N_TRIALS_EQUILIBRIA} random initial conditions per randomization")
        
        mean_n_eq, std_n_eq, all_counts = compute_equilibria_multiple_randomizations(
            network,
            n_randomizations=N_RANDOMIZATIONS,
            n_trials=N_TRIALS_EQUILIBRIA,
            equilibria_tol=EQUILIBRIA_TOL,
            unique_tol=UNIQUE_TOL
        )
        
        print(f"\nEquilibria found across {N_RANDOMIZATIONS} randomizations:")
        print(f"  Mean: {mean_n_eq:.1f}")
        print(f"  Std:  {std_n_eq:.1f}")
        print(f"  All counts: {all_counts}")
        
        # Store results
        results.append({
            'trial_number': trial.number,
            'rank': i + 1,
            'objective_value': trial.value,
            'n_equilibria_mean': mean_n_eq,
            'n_equilibria_std': std_n_eq,
            'n_equilibria_min': min(all_counts),
            'n_equilibria_max': max(all_counts),
            'n_equilibria_all': str(all_counts),  # Store as string for CSV
            'n_units': params['n_units'],
            'n_connections': params['n_connections'],
            'connectivity': params['n_connections'] / params['n_units'],
            'lin_stiff_min': params['lin_stiff_min'],
            'lin_stiff_max': params['lin_stiff_max'],
            'eq_dist_min': params['eq_dist_min'],
            'eq_dist_max': params['eq_dist_max'],
            'ang_connections': params['ang_connections'],
        })
    
    # Create results dataframe
    df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    
    # Save results
    output_file = f"results/{STUDY_NAME}_top{N_TOP_TRIALS}_equilibria_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Simple statistics
    print(f"\nEquilibria statistics (mean across randomizations):")
    print(f"  Mean: {df['n_equilibria_mean'].mean():.1f}")
    print(f"  Std: {df['n_equilibria_mean'].std():.1f}")
    print(f"  Min: {df['n_equilibria_mean'].min():.1f}")
    print(f"  Max: {df['n_equilibria_mean'].max():.1f}")
    
    # Correlation with performance
    corr = df['objective_value'].corr(df['n_equilibria_mean'])
    print(f"\nCorrelation (objective vs n_equilibria_mean): {corr:.3f}")
    
    return df

if __name__ == "__main__":
    main()
