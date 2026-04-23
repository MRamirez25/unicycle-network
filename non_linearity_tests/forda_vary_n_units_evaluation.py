#%%
"""
Evaluate FordA with varying n_units using parameters from a single Optuna study.

This script:
1. Loads the best parameters from a single Optuna study
2. Varies only n_units while keeping all other parameters fixed
3. Evaluates performance across multiple seeds for each n_units value
4. Generates a line plot showing test accuracy vs n_units
"""

import os
import sys
import time
import numpy as np
import torch
import random
from tqdm import tqdm
from datetime import datetime
import pandas as pd

# Fix matplotlib backend issues - use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import optuna

#%%
# Add the parent directory to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import get_FordA_data, n_params, count_classifier_params
from unicycle_network_class import UnicycleReservoir

#%%
# ============================================================
# CONFIGURATION
# ============================================================

# Source study to get base parameters from
DATABASE_NAME = "forda_increasing_n_units"#"unicycle_nets_forda_logreg" #"forda_increasing_n_units"
SOURCE_STUDY_NAME = "20_units" #"only_last_state_as_readout_20_units_not_aligned_less_ang_magnitude_no_tanh"  # Study to load parameters from

# n_units values to evaluate
N_UNITS_LIST = [5, 10, 15, 20, 25, 30]

# Evaluation settings
RANDOM_SEEDS = [33, 42, 123, 777, 1024]

#%%
def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#%%
def load_study_params(database_name, study_name):
    """Load best parameters from an Optuna study."""
    storage_name = f"sqlite:///{parent_dir}/optuna_databases/{database_name}.db"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        
        if study.best_trial is None:
            print(f"  Warning: Study '{study_name}' has no completed trials")
            return None, None
        
        params = study.best_params
        
        study_info = {
            'study_name': study_name,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        }
        
        return params, study_info
    except Exception as e:
        print(f"  Error loading study '{study_name}': {e}")
        return None, None

#%%
def create_input_maps(params, n_units, seed, ang_input=True):
    """Create input maps with given seed for reproducibility"""
    set_seed(seed)
    
    # Linear input map
    lin_input_map = torch.zeros(1, n_units)
    # Scale non_zero_elements proportionally to n_units
    base_n_units = int(SOURCE_STUDY_NAME.split('_')[0])
    num_non_zero_base = params.get('non_zero_elements', base_n_units // 2)
    num_non_zero = max(1, int(num_non_zero_base * n_units / base_n_units))
    num_non_zero = min(num_non_zero, n_units)  # Can't exceed n_units
    
    non_zero_indices = torch.randperm(n_units)[:num_non_zero]
    magnitude_min = params.get('magnitude_min', 0.1)
    magnitude_max = params.get('magnitude_max', 1.0)
    lin_input_map[0, non_zero_indices] = torch.rand(num_non_zero) * (magnitude_max - magnitude_min) + magnitude_min
    
    # Angular input map
    ang_input_map = torch.zeros(1, n_units)
    if ang_input:
        num_non_zero_ang_base = params.get('non_zero_elements_ang', base_n_units // 2)
        num_non_zero_ang = max(1, int(num_non_zero_ang_base * n_units / base_n_units))
        num_non_zero_ang = min(num_non_zero_ang, n_units)
        
        non_zero_indices = torch.randperm(n_units)[:num_non_zero_ang]
        magnitude_min_ang = params.get('magnitude_min_ang', 0.1)
        magnitude_max_ang = params.get('magnitude_max_ang', 1.0)
        ang_input_map[0, non_zero_indices] = torch.rand(num_non_zero_ang) * (magnitude_max_ang - magnitude_min_ang) + magnitude_min_ang
    
    return lin_input_map, ang_input_map

#%%
def initialize_model(params, n_units, lin_input_map, ang_input_map, device, ang_connections=True):
    """Initialize and configure the model with specified n_units"""
    
    # Extract parameters - scale connection counts proportionally to n_units
    n_connections_fraction = params.get('n_connections_fraction', 1.0)
    n_connections = int(n_units * n_connections_fraction)
    anchor_con_fraction = params.get('anchor_con_fraction', 0.5)
    n_connections_anchor = int(n_units * anchor_con_fraction)
    
    # Angular connections
    if ang_connections:
        n_connections_ang_fraction = params.get('n_connections_ang_fraction', 0.5)
        anchor_con_fraction_ang = params.get('anchor_con_fraction_ang', 0.5)
        n_connections_anchor_ang = int(anchor_con_fraction_ang * n_units)
        n_connections_ang = int(n_connections_ang_fraction * n_units)
    else:
        n_connections_anchor_ang = 0
        n_connections_ang = 0
    
    # Create model with safe defaults
    model = UnicycleReservoir(
        n_inp=1, n_units=n_units, dt=params.get('dt', 0.01), n_out=2,
        lin_input_map=lin_input_map,
        lin_stiff_min=params.get('lin_stiff_min', 0.1),
        lin_stiff_max=params.get('lin_stiff_max', 1.0),
        ang_stiff_min=params.get('ang_stiff_min', 0.1),
        ang_stiff_max=params.get('ang_stiff_max', 1.0),
        lin_damping_min=params.get('lin_damping_min', 0.1),
        lin_damping_max=params.get('lin_damping_max', 0.5),
        ang_damping_min=params.get('ang_damping_min', 0.1),
        ang_damping_max=params.get('ang_damping_max', 0.5),
        eq_dist_min=params.get('eq_dist_min', 0.1),
        eq_dist_max=params.get('eq_dist_max', 1.0),
        eq_dist_min_ang=params.get('eq_dist_min_ang', 0.0),
        eq_dist_max_ang=params.get('eq_dist_max_ang', np.pi),
        n_connections=n_connections, n_connections_anchor=n_connections_anchor,
        n_past_steps_readout=0, n_connections_ang=n_connections_ang,
        n_connections_anchor_ang=n_connections_anchor_ang,
        inp_bias=params.get('inp_bias', 0.0), ang_input_map=ang_input_map
    ).to(device)
    
    return model

#%%
def setup_initial_states(model, batch_size, device, aligned_orientations=False):
    """Setup initial states for the model"""
    model.set_init_states_random(batch_size)
    
    # Move to device
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
    
    # Set specific initial conditions
    model.s_init[:,0] = 0
    model.omega_init[:,:] = 0
    
    if not aligned_orientations:
        model.theta_init[:,:] = torch.rand(model.theta_init.size()) * (4*torch.pi) - 2*torch.pi
        model.theta_init[:,0] = 0
    else:
        model.theta_init[:,:] = torch.rand(1) * (4*torch.pi) - 2*torch.pi
    
    return model

#%%
def run_washup(model, params, device):
    """Run washup phase and return final states"""
    washup = params.get('washup_steps', 100)
    
    x = model.x_init[0:1,:]
    z = model.z_init[0:1,:]
    theta = model.theta_init[0:1,:]
    s = model.s_init[0:1,:]
    omega = model.omega_init[0:1,:]
    
    u_lin = torch.zeros((1, washup, 1), device=device)
    u_ang = torch.zeros_like(u_lin, device=device)
    
    for t in range(u_lin.size()[1]):
        linear_input = (u_lin[:, t]) @ model.lin_input_map
        angular_input = u_ang[:, t] @ model.ang_input_map
        x, z, theta, s, omega = model.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)
    
    return x, z, theta, s, omega

#%%
@torch.no_grad()
def test_esn(data_loader, model, classifier, scaler, device):
    """Test ESN performance"""
    activations, ys = [], []
    for x, labels in data_loader:
        x = x.to(device)
        labels = labels.to(device)
        states_list, output, mid_states = model(x, x)
        activations.append(mid_states.cpu())
        ys.append(labels.cpu())
    
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)

#%%
def single_evaluation(params, n_units, seed, device, train_loader, valid_loader, test_loader,
                      ang_input=True, ang_connections=True, aligned_orientations=False):
    """Run a single evaluation with given seed and n_units"""
    batch_size = params.get('batch_size', 32)
    
    # Set seed and create input maps
    lin_input_map, ang_input_map = create_input_maps(params, n_units, seed, ang_input)
    
    # Initialize model
    model = initialize_model(params, n_units, lin_input_map, ang_input_map, device, ang_connections)
    model = setup_initial_states(model, batch_size, device, aligned_orientations)
    
    # Run washup
    x, z, theta, s, omega = run_washup(model, params, device)
    
    # Set initial states after washup
    model.set_init_states(batch_size, x, z, theta, s, omega)
    
    # Train logistic regression classifier
    activations, ys = [], []
    for x, labels in train_loader:
        x = x.to(device)
        labels = labels.to(device)
        states_list, output, mid_states = model(x, x)
        activations.append(mid_states.detach().cpu())
        ys.append(labels.cpu())
    
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    
    # Check for NaN values
    if np.isnan(activations).any():
        return None, None, None
    
    # Standardize and train classifier
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000).fit(activations, ys)
    
    # Count classifier parameters
    n_classifier_params = count_classifier_params(classifier)
    
    # Evaluate
    valid_score = test_esn(valid_loader, model, classifier, scaler, device)
    test_score = test_esn(test_loader, model, classifier, scaler, device)
    
    return valid_score, test_score, n_classifier_params

#%%
def evaluate_n_units(params, n_units, device, train_loader, valid_loader, test_loader, seeds,
                     ang_input=True, ang_connections=True, aligned_orientations=False):
    """Evaluate a single n_units value across multiple seeds."""
    print(f"\n  Evaluating n_units={n_units}")
    
    valid_scores = []
    test_scores = []
    classifier_params = None
    
    for seed in seeds:
        try:
            valid_score, test_score, n_classifier_params = single_evaluation(
                params, n_units, seed, device, train_loader, valid_loader, test_loader,
                ang_input, ang_connections, aligned_orientations
            )
            
            if valid_score is not None and test_score is not None:
                valid_scores.append(valid_score)
                test_scores.append(test_score)
                if classifier_params is None:
                    classifier_params = n_classifier_params
                print(f"    Seed {seed}: Valid={valid_score:.4f}, Test={test_score:.4f}")
            else:
                print(f"    Seed {seed}: Skipped (NaN values)")
        except Exception as e:
            print(f"    Seed {seed}: Error - {e}")
    
    if len(valid_scores) == 0:
        print(f"  No successful evaluations for n_units={n_units}")
        return None
    
    # Calculate statistics
    results = {
        'n_units': n_units,
        'valid_mean': np.mean(valid_scores),
        'valid_std': np.std(valid_scores),
        'test_mean': np.mean(test_scores),
        'test_std': np.std(test_scores),
        'valid_min': np.min(valid_scores),
        'valid_max': np.max(valid_scores),
        'test_min': np.min(test_scores),
        'test_max': np.max(test_scores),
        'n_successful_seeds': len(valid_scores),
        'classifier_params': classifier_params,
    }
    
    print(f"  Results: Valid={results['valid_mean']:.4f}±{results['valid_std']:.4f}, "
          f"Test={results['test_mean']:.4f}±{results['test_std']:.4f}")
    
    return results

#%%
def main():
    """Main function to evaluate varying n_units with fixed parameters."""
    print("="*70)
    print("VARYING N_UNITS EVALUATION FOR FordA")
    print("="*70)
    print(f"Source study: {DATABASE_NAME}/{SOURCE_STUDY_NAME}")
    print(f"n_units values: {N_UNITS_LIST}")
    print(f"Seeds: {RANDOM_SEEDS}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load base parameters from source study
    print(f"\nLoading parameters from study '{SOURCE_STUDY_NAME}'...")
    params, study_info = load_study_params(DATABASE_NAME, SOURCE_STUDY_NAME)
    
    if params is None:
        print("Failed to load study parameters!")
        return None
    
    print(f"  Original best value: {study_info['best_value']:.4f}")
    print(f"  Completed trials: {study_info['n_complete']}/{study_info['n_trials']}")
    
    # Extract config flags from params
    ang_input = params.get('ang_input', True)
    ang_connections = params.get('ang_connections', True)
    aligned_orientations = params.get('aligned_orientations', False)
    print(f"  Config: ang_input={ang_input}, ang_connections={ang_connections}, aligned={aligned_orientations}")
    
    # Load data with batch size from params
    batch_size = params.get('batch_size', 32)
    print(f"\nLoading FordA data (batch_size={batch_size})...")
    train_loader, valid_loader, test_loader = get_FordA_data(batch_size, batch_size)
    print(f"Data loaded. Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}, Test batches: {len(test_loader)}")
    
    # Evaluate each n_units value
    print("\n" + "="*70)
    print("EVALUATING DIFFERENT N_UNITS VALUES")
    print("="*70)
    
    all_results = []
    
    for n_units in N_UNITS_LIST:
        result = evaluate_n_units(
            params, n_units, device, train_loader, valid_loader, test_loader,
            RANDOM_SEEDS, ang_input, ang_connections, aligned_orientations
        )
        if result is not None:
            all_results.append(result)
    
    if len(all_results) == 0:
        print("\nNo successful evaluations!")
        return None
    
    # Create results dataframe
    df = pd.DataFrame(all_results)
    
    # Sort by n_units
    df = df.sort_values('n_units')
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Print table
    print("\n" + df[['n_units', 'valid_mean', 'valid_std', 
                    'test_mean', 'test_std', 'classifier_params']].to_string(index=False))
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{parent_dir}/results/forda_vary_n_units_{SOURCE_STUDY_NAME}_{timestamp}.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")
    
    # Create line plot of test accuracy vs n_units
    plt.figure(figsize=(10, 6))
    plt.errorbar(df['n_units'], df['test_mean'], 
                 yerr=df['test_std'], 
                 marker='o', markersize=8, capsize=5, capthick=2, 
                 linewidth=2, elinewidth=1.5, linestyle='--', color='b', label='Test Accuracy')
    
    plt.xlabel('Number of Units', fontsize=12)
    plt.ylabel('mean Test Accuracy', fontsize=12)
    # plt.title(f'Test Accuracy vs Network Size\n(Parameters from {SOURCE_STUDY_NAME})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to actual n_units values
    plt.xticks(df['n_units'])
    
    # Add horizontal line for original study's best value
    # plt.axhline(y=study_info['best_value'], color='r', linestyle='--', 
                # alpha=0.7, label=f"Original Optuna value ({study_info['best_value']:.4f})")
    # plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{parent_dir}/plots/forda_vary_n_units_{SOURCE_STUDY_NAME}_{timestamp}.png"
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    plt.close()
    
    return df

#%%
if __name__ == "__main__":
    results_df = main()
    
    if results_df is not None:
        print("\n" + "="*70)
        print("FINAL RESULTS (sorted by n_units)")
        print("="*70)
        for _, row in results_df.iterrows():
            print(f"n_units={int(row['n_units']):3d}: "
                  f"Test={row['test_mean']:.4f}±{row['test_std']:.4f}, "
                  f"Valid={row['valid_mean']:.4f}±{row['valid_std']:.4f}, "
                  f"Classifier params={row['classifier_params']}")
