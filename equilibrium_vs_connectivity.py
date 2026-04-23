"""
Analyze how equilibria count varies with connectivity.

This script:
1. Either loads a top configuration from Optuna OR generates random parameters
2. Varies the connectivity (n_connections) systematically
3. For each connectivity level, computes equilibria across multiple randomizations
4. Saves results showing relationship between connectivity and equilibria count
"""

import numpy as np
import torch
import optuna
from scipy.optimize import root
from unicycle_network_class import UnicycleNetwork
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

# Source of base parameters
USE_OPTUNA = False  # If True, load from Optuna; if False, use random parameters
DATABASE_NAME = "unicycle_nets_forda_logreg"
STUDY_NAME = "only_last_state_as_readout_20_units_not_aligned_less_ang_magnitude_no_tanh"
TRIAL_RANK = 1  # Which trial to use (1 = best, 2 = second best, etc.)

# Network size
N_UNITS = 20

# Connectivity sweep
CONNECTIVITY_FRACTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Fractions of max connections

# Equilibria search parameters
N_RANDOMIZATIONS = 5  # Number of different p0/theta initializations
N_TRIALS_EQUILIBRIA = 5000  # Random initial conditions per randomization
EQUILIBRIA_TOL = 1e-6
UNIQUE_TOL = 1e-4

# Random parameter ranges (used if USE_OPTUNA = False)
RANDOM_PARAMS = {
    'lin_stiff_min': (0.1, 1.0),
    'lin_stiff_max': (1.0, 10.0),
    'ang_stiff_min': (0.1, 1.0),
    'ang_stiff_max': (1.0, 2.0),
    'eq_dist_min': (0.2, 1.0),
    'eq_dist_max': (1.0, 2.0),
    'eq_dist_min_ang': (-2*np.pi, 0.0),
    'eq_dist_max_ang': (0.0, 2*np.pi),
}

# ============================================================
# LOAD OR GENERATE PARAMETERS
# ============================================================

def load_params_from_optuna(database_name, study_name, rank=1):
    """Load parameters from a specific trial in Optuna study."""
    storage_name = f"sqlite:///optuna_databases/{database_name}.db"
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    
    # Get trials sorted by objective value (best first)
    trials = study.trials
    trials_sorted = sorted([t for t in trials if t.state == optuna.trial.TrialState.COMPLETE],
                          key=lambda t: t.value,
                          reverse=(study.direction == optuna.study.StudyDirection.MAXIMIZE))
    
    trial = trials_sorted[rank - 1]
    params = trial.params
    
    print(f"Loaded trial #{trial.number} (rank {rank})")
    print(f"Objective value: {trial.value:.6f}")
    
    # Extract relevant parameters
    base_params = {
        'n_units': params.get('n_units', N_UNITS),
        'lin_stiff_min': params['lin_stiff_min'],
        'lin_stiff_max': params['lin_stiff_max'],
        'ang_stiff_min': params.get('ang_stiff_min', 0.1),
        'ang_stiff_max': params.get('ang_stiff_max', 1.0),
        'eq_dist_min': params['eq_dist_min'],
        'eq_dist_max': params['eq_dist_max'],
        'eq_dist_min_ang': params.get('eq_dist_min_ang', 0.0),
        'eq_dist_max_ang': params.get('eq_dist_max_ang', np.pi),
        'anchor_con_fraction': params.get('anchor_con_fraction', 0.5),
        'trial_number': trial.number,
        'objective_value': trial.value,
    }
    
    return base_params

def generate_random_params():
    """Generate random parameters within specified ranges."""
    
    lin_stiff_min = np.random.uniform(*RANDOM_PARAMS['lin_stiff_min'])
    lin_stiff_max = np.random.uniform(lin_stiff_min, RANDOM_PARAMS['lin_stiff_max'][1])
    
    ang_stiff_min = np.random.uniform(*RANDOM_PARAMS['ang_stiff_min'])
    ang_stiff_max = np.random.uniform(ang_stiff_min, RANDOM_PARAMS['ang_stiff_max'][1])
    
    eq_dist_min = np.random.uniform(*RANDOM_PARAMS['eq_dist_min'])
    eq_dist_max = np.random.uniform(eq_dist_min, RANDOM_PARAMS['eq_dist_max'][1])
    
    eq_dist_min_ang = np.random.uniform(*RANDOM_PARAMS['eq_dist_min_ang'])
    eq_dist_max_ang = np.random.uniform(eq_dist_min_ang, RANDOM_PARAMS['eq_dist_max_ang'][1])
    
    base_params = {
        'n_units': N_UNITS,
        'lin_stiff_min': lin_stiff_min,
        'lin_stiff_max': lin_stiff_max,
        'ang_stiff_min': ang_stiff_min,
        'ang_stiff_max': ang_stiff_max,
        'eq_dist_min': eq_dist_min,
        'eq_dist_max': eq_dist_max,
        'eq_dist_min_ang': eq_dist_min_ang,
        'eq_dist_max_ang': eq_dist_max_ang,
        'anchor_con_fraction': 0.5,
        'trial_number': None,
        'objective_value': None,
    }
    
    print("Generated random parameters:")
    print(f"  Linear stiffness: [{lin_stiff_min:.3f}, {lin_stiff_max:.3f}]")
    print(f"  Angular stiffness: [{ang_stiff_min:.3f}, {ang_stiff_max:.3f}]")
    print(f"  Eq distances: [{eq_dist_min:.3f}, {eq_dist_max:.3f}]")
    
    return base_params

# ============================================================
# CREATE NETWORK WITH SPECIFIC CONNECTIVITY
# ============================================================

def create_network_with_connectivity(base_params, n_connections, n_connections_anchor):
    """Create a UnicycleNetwork with specific connectivity."""
    
    # Dummy input maps (not needed for equilibria)
    lin_input_map = torch.zeros(1, base_params['n_units'])
    ang_input_map = torch.zeros(1, base_params['n_units'])
    
    network = UnicycleNetwork(
        n_inp=1,
        n_units=base_params['n_units'],
        dt=0.01,  # dt doesn't matter for equilibria
        lin_stiff_min=base_params['lin_stiff_min'],
        lin_stiff_max=base_params['lin_stiff_max'],
        ang_stiff_min=base_params['ang_stiff_min'],
        ang_stiff_max=base_params['ang_stiff_max'],
        lin_damping_min=0.1,  # damping doesn't matter for equilibria
        lin_damping_max=0.2,
        ang_damping_min=0.1,
        ang_damping_max=0.2,
        eq_dist_min=base_params['eq_dist_min'],
        eq_dist_max=base_params['eq_dist_max'],
        eq_dist_min_ang=base_params['eq_dist_min_ang'],
        eq_dist_max_ang=base_params['eq_dist_max_ang'],
        lin_input_map=lin_input_map,
        ang_input_map=ang_input_map,
        n_connections=n_connections,
        n_connections_anchor=n_connections_anchor,
        n_connections_ang=0,  # No angular connections for simplicity
        n_connections_anchor_ang=0,
    )
    
    return network

# ============================================================
# EQUILIBRIA COMPUTATION
# ============================================================

def compute_equilibria(network, p0, theta, n_trials=5000, equilibria_tol=1e-6, unique_tol=1e-4):
    """Compute equilibria for a unicycle network with given p0 and theta."""
    n_units = network.n_units
    
    # Extract network matrices
    K_mat = network.stiffness_coupling_matrix.detach().cpu().numpy()
    A_mat = network.eq_distances_matrix.detach().cpu().numpy().squeeze(-1)
    
    e = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    
    def positions(s):
        s_full = np.zeros(n_units)
        s_full[1:] = s
        return p0 + s_full[:, None] * e
    
    def equilibrium_equations(s):
        p = positions(s)
        d = p[:, None, :] - p[None, :, :]
        r = np.linalg.norm(d, axis=2) + np.eye(n_units)
        F_contrib = K_mat * (A_mat - r) * np.einsum('ijk,ik->ij', d, e) / r
        F = np.sum(F_contrib, axis=1)
        return F[1:]
    
    solutions = []
    
    for _ in range(n_trials):
        s0 = np.random.randn(n_units - 1) * 2
        
        try:
            sol = root(equilibrium_equations, s0, method='hybr')
            if sol.success and np.linalg.norm(sol.fun) < equilibria_tol:
                solutions.append(sol.x)
        except:
            continue
    
    if len(solutions) == 0:
        return 0, []
    
    solutions = np.array(solutions)
    
    # Remove duplicates
    unique = []
    for s in solutions:
        if not any(np.linalg.norm(s - u) < unique_tol for u in unique):
            unique.append(s)
    
    unique_eq = np.array(unique) if unique else np.array([])
    
    return len(unique_eq), unique_eq

def compute_equilibria_multiple_randomizations(network, n_randomizations=5, n_trials=5000,
                                               equilibria_tol=1e-6, unique_tol=1e-4):
    """Compute equilibria for multiple random initializations of p0 and theta."""
    n_units = network.n_units
    equilibria_counts = []
    
    for i in range(n_randomizations):
        # Generate random initial positions and orientations
        p0 = np.random.uniform(-1, 1, (n_units, 2))
        theta = np.random.uniform(-2*np.pi, 2*np.pi, n_units)
        theta[0] = 0  # Fix first unicycle orientation
        
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
    print("="*60)
    print("EQUILIBRIA vs CONNECTIVITY ANALYSIS")
    print("="*60)
    
    # Load or generate base parameters
    if USE_OPTUNA:
        print(f"\nLoading parameters from Optuna study: {STUDY_NAME}")
        base_params = load_params_from_optuna(DATABASE_NAME, STUDY_NAME, TRIAL_RANK)
    else:
        print("\nGenerating random parameters")
        base_params = generate_random_params()
    
    print(f"\nBase configuration:")
    print(f"  n_units: {base_params['n_units']}")
    print(f"  Linear stiffness: [{base_params['lin_stiff_min']:.3f}, {base_params['lin_stiff_max']:.3f}]")
    print(f"  Equilibrium distances: [{base_params['eq_dist_min']:.3f}, {base_params['eq_dist_max']:.3f}]")
    
    # Sweep connectivity
    results = []
    
    print(f"\n{'='*60}")
    print(f"Sweeping connectivity: {len(CONNECTIVITY_FRACTIONS)} levels")
    print(f"  {N_RANDOMIZATIONS} randomizations per level")
    print(f"  {N_TRIALS_EQUILIBRIA} trials per randomization")
    print(f"{'='*60}\n")
    
    for conn_frac in tqdm(CONNECTIVITY_FRACTIONS, desc="Connectivity sweep"):
        n_connections = int(base_params['n_units'] * conn_frac)
        n_connections_anchor = int(base_params['n_units'] * base_params['anchor_con_fraction'])
        
        print(f"\nConnectivity: {conn_frac:.1%} ({n_connections}/{base_params['n_units']-1} connections)")
        
        # Create network with this connectivity
        network = create_network_with_connectivity(base_params, n_connections, n_connections_anchor)
        
        # Compute equilibria across multiple randomizations
        mean_n_eq, std_n_eq, all_counts = compute_equilibria_multiple_randomizations(
            network,
            n_randomizations=N_RANDOMIZATIONS,
            n_trials=N_TRIALS_EQUILIBRIA,
            equilibria_tol=EQUILIBRIA_TOL,
            unique_tol=UNIQUE_TOL
        )
        
        print(f"  Mean equilibria: {mean_n_eq:.1f} ± {std_n_eq:.1f}")
        print(f"  Counts: {all_counts}")
        
        results.append({
            'connectivity_fraction': conn_frac,
            'n_connections': n_connections,
            'n_equilibria_mean': mean_n_eq,
            'n_equilibria_std': std_n_eq,
            'n_equilibria_min': min(all_counts),
            'n_equilibria_max': max(all_counts),
            'n_equilibria_all': str(all_counts),
        })
    
    # Create results dataframe
    df = pd.DataFrame(results)
    
    # Add base parameters to dataframe
    for key, value in base_params.items():
        df[key] = value
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(df[['connectivity_fraction', 'n_connections', 'n_equilibria_mean', 'n_equilibria_std']].to_string(index=False))
    
    # Save results
    if USE_OPTUNA:
        output_file = f"results/connectivity_sweep_trial{base_params['trial_number']}.csv"
    else:
        output_file = f"results/connectivity_sweep_random.csv"
    
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Plot results
    plot_results(df, output_file.replace('.csv', '.png'))
    
    return df

def plot_results(df, output_file):
    """Plot equilibria count vs connectivity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean with error bars
    ax = axes[0]
    ax.errorbar(df['connectivity_fraction'], df['n_equilibria_mean'], 
                yerr=df['n_equilibria_std'], marker='o', markersize=8,
                capsize=5, capthick=2, linewidth=2, label='Mean ± Std')
    ax.fill_between(df['connectivity_fraction'], 
                     df['n_equilibria_min'], 
                     df['n_equilibria_max'],
                     alpha=0.2, label='Min-Max range')
    ax.set_xlabel('Connectivity Fraction', fontsize=12)
    ax.set_ylabel('Number of Equilibria', fontsize=12)
    ax.set_title('Equilibria vs Connectivity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: All individual counts
    ax = axes[1]
    for i, row in df.iterrows():
        counts = eval(row['n_equilibria_all'])  # Convert string back to list
        x_vals = [row['connectivity_fraction']] * len(counts)
        ax.scatter(x_vals, counts, alpha=0.5, s=50)
    ax.plot(df['connectivity_fraction'], df['n_equilibria_mean'], 
            'r-', linewidth=2, marker='o', markersize=8, label='Mean')
    ax.set_xlabel('Connectivity Fraction', fontsize=12)
    ax.set_ylabel('Number of Equilibria', fontsize=12)
    ax.set_title('Individual Randomization Results', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()

if __name__ == "__main__":
    main()
