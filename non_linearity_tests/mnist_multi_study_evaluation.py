#%%
"""
Evaluate multiple Optuna studies from a database and aggregate results for MNIST.

This script:
1. Lists all studies in a given database
2. Runs evaluation for each study (or a specified subset)
3. Aggregates results into a comparison table
4. Saves results to CSV and generates comparison plots
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

from utils import get_mnist_data, n_params, count_classifier_params
from unicycle_network import UnicycleReservoir

#%%
# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_NAME = "unicycle_mnist_increasing_n_smaller"

# List of studies to evaluate (None = all studies in database)
# You can specify a list like: ["study1", "study2"] or None for all
STUDY_NAMES = [f"{n_units}_units" for n_units in range(4, 21, 4)]  # Set to None to evaluate all studies

# Evaluation settings
RANDOM_SEEDS = [33, 42, 123, 777, 1024] # [777, 1024]# Seeds for multi-study evaluation
N_UNITS = 100  # Default number of units for MNIST if not specified in study

# MNIST-specific settings
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # All 10 digits
N_OUTPUTS = 10  # Number of output classes

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
def get_all_studies(database_name):
    """Get list of all study names in a database."""
    storage_name = f"sqlite:///{parent_dir}/optuna_databases/{database_name}.db"
    study_summaries = optuna.study.get_all_study_summaries(storage=storage_name)
    return [s.study_name for s in study_summaries]

#%%
def load_study_params(database_name, study_name, n_units_override=None):
    """Load best parameters and study info from an Optuna study.
    
    Args:
        database_name: Name of the database
        study_name: Name of the study
        n_units_override: If provided, use this value for n_units instead of 
                         extracting from params (useful when n_units is encoded
                         in the study name rather than stored in params)
    """
    storage_name = f"sqlite:///{parent_dir}/optuna_databases/{database_name}.db"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        
        if study.best_trial is None:
            print(f"  Warning: Study '{study_name}' has no completed trials")
            return None, None, None
        
        # Get study configuration from the best trial's params
        params = study.best_params
        
        # Determine n_units: override > params > default
        if n_units_override is not None:
            n_units = n_units_override
        else:
            n_units = params.get('n_units', N_UNITS)
        
        # Extract configuration flags from params (with defaults for MNIST)
        config = {
            'aligned_orientations': params.get('aligned_orientations', False),
            'ang_input': params.get('ang_input', True),
            'ang_connections': params.get('ang_connections', True),
            'n_units': n_units,
        }
        
        study_info = {
            'study_name': study_name,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'direction': str(study.direction),
        }
        
        return params, config, study_info
    except Exception as e:
        print(f"  Error loading study '{study_name}': {e}")
        return None, None, None

#%%
def create_input_maps(params, n_units, seed, ang_input=True):
    """Create input maps with given seed for reproducibility"""
    set_seed(seed)
    
    # Linear input map
    lin_input_map = torch.zeros(1, n_units)
    num_non_zero = params.get('non_zero_elements', n_units // 2)
    non_zero_indices = torch.randperm(n_units)[:num_non_zero]
    magnitude_min = params.get('magnitude_min', 0.1)
    magnitude_max = params.get('magnitude_max', 1.0)
    lin_input_map[0, non_zero_indices] = torch.rand(num_non_zero) * (magnitude_max - magnitude_min) + magnitude_min
    
    # Angular input map
    ang_input_map = torch.zeros(1, n_units)
    if ang_input:
        num_non_zero_ang = params.get('non_zero_elements_ang', n_units // 2)
        non_zero_indices = torch.randperm(n_units)[:num_non_zero_ang]
        magnitude_min_ang = params.get('magnitude_min_ang', 0.1)
        magnitude_max_ang = params.get('magnitude_max_ang', 1.0)
        ang_input_map[0, non_zero_indices] = torch.rand(num_non_zero_ang) * (magnitude_max_ang - magnitude_min_ang) + magnitude_min_ang
    
    return lin_input_map, ang_input_map

#%%
def initialize_model(params, config, lin_input_map, ang_input_map, device):
    """Initialize and configure the model for MNIST"""
    n_units = config.get('n_units', N_UNITS)
    ang_connections = config.get('ang_connections', True)
    
    # Extract parameters
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
    
    # Create model with safe defaults (n_out=10 for MNIST)
    model = UnicycleReservoir(
        n_inp=1, n_units=n_units, dt=params.get('dt', 0.01), n_out=N_OUTPUTS,
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
def setup_initial_states(model, params, config, device):
    """Setup initial states for the model"""
    bs_train = params.get('batch_size', 500)  # MNIST typically uses larger batches
    aligned_orientations = config.get('aligned_orientations', False)
    
    model.set_init_states_random(bs_train)
    
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
def test_esn(data_loader, model, classifier, scaler, device, n_units):
    """Test ESN performance on MNIST"""
    activations, ys = [], []
    for images, labels in data_loader:
        # Reshape MNIST images: (batch, 28, 28) -> (batch, 784, 1)
        images = images.reshape(images.shape[0], 1, 784)
        images = images.permute(0, 2, 1)
        images = images.to(device)
        labels = labels.to(device)
        
        states_list, output, mid_states = model(images, images)
        # Use only x and y positions (first 2*n_units features)
        mid_states = mid_states[:, :]
        activations.append(mid_states.cpu())
        ys.append(labels.cpu())
    
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)

#%%
def single_evaluation(params, config, seed, device, train_loader, valid_loader, test_loader):
    """Run a single evaluation with given seed"""
    n_units = config.get('n_units', N_UNITS)
    ang_input = config.get('ang_input', True)
    
    # Set seed and create input maps
    lin_input_map, ang_input_map = create_input_maps(params, n_units, seed, ang_input)
    
    # Initialize model
    model = initialize_model(params, config, lin_input_map, ang_input_map, device)
    model = setup_initial_states(model, params, config, device)
    
    # Run washup
    x, z, theta, s, omega = run_washup(model, params, device)
    
    # Set initial states after washup
    model.set_init_states(params.get('batch_size', 500), x, z, theta, s, omega)
    
    # Train logistic regression classifier
    activations, ys = [], []
    for images, labels in train_loader:
        # Reshape MNIST images: (batch, 28, 28) -> (batch, 784, 1)
        images = images.reshape(images.shape[0], 1, 784)
        images = images.permute(0, 2, 1)
        images = images.to(device)
        labels = labels.to(device)
        
        states_list, output, mid_states = model(images, images)
        # Use only x and y positions (first 2*n_units features)
        mid_states = mid_states[:, :]
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
    valid_score = test_esn(valid_loader, model, classifier, scaler, device, n_units)
    test_score = test_esn(test_loader, model, classifier, scaler, device, n_units)
    
    return valid_score, test_score, n_classifier_params

#%%
def evaluate_study(database_name, study_name, device, seeds, n_units_override=None):
    """Evaluate a single study across multiple seeds.
    
    Args:
        database_name: Name of the Optuna database
        study_name: Name of the study
        device: PyTorch device
        seeds: List of random seeds to use
        n_units_override: If provided, use this value for n_units
    """
    print(f"\n{'='*60}")
    print(f"Evaluating study: {study_name}")
    print('='*60)
    
    # Load study parameters
    result = load_study_params(database_name, study_name, n_units_override)
    if result[0] is None:
        return None
    
    params, config, study_info = result
    
    # Load data with the batch size from this study's params
    batch_size = params.get('batch_size', 500)
    data_root = parent_dir + '/data/'
    train_loader, valid_loader, test_loader = get_mnist_data(
        bs_train=batch_size, bs_test=batch_size, 
        classes=CLASSES,
        new_fraction=1.0, test_fraction=1.0, path=data_root
    )
    
    print(f"  Best optuna value: {study_info['best_value']:.4f}")
    print(f"  Completed trials: {study_info['n_complete']}/{study_info['n_trials']}")
    print(f"  Config: n_units={config['n_units']}, aligned={config['aligned_orientations']}, ang_input={config['ang_input']}, ang_conn={config['ang_connections']}")
    print(f"  Batch size: {batch_size}")
    
    # Run evaluations
    valid_scores = []
    test_scores = []
    classifier_params = None
    
    for seed in seeds:
        try:
            valid_score, test_score, n_classifier_params = single_evaluation(
                params, config, seed, device, train_loader, valid_loader, test_loader
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
            import traceback
            traceback.print_exc()
    
    if len(valid_scores) == 0:
        print(f"  No successful evaluations for study {study_name}")
        return None
    
    # Calculate statistics
    results = {
        'study_name': study_name,
        'optuna_best_value': study_info['best_value'],
        'n_trials': study_info['n_trials'],
        'n_complete': study_info['n_complete'],
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
        'aligned_orientations': config['aligned_orientations'],
        'ang_input': config['ang_input'],
        'ang_connections': config['ang_connections'],
        'n_units': config['n_units'],
    }
    
    print(f"  Results: Valid={results['valid_mean']:.4f}±{results['valid_std']:.4f}, "
          f"Test={results['test_mean']:.4f}±{results['test_std']:.4f}")
    
    return results

#%%
def main():
    """Main function to evaluate multiple studies."""
    print("="*70)
    print("MULTI-STUDY EVALUATION FOR MNIST")
    print("="*70)
    print(f"Database: {DATABASE_NAME}")
    print(f"Seeds: {RANDOM_SEEDS}")
    print(f"Classes: {CLASSES}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get list of studies
    if STUDY_NAMES is None:
        study_names = get_all_studies(DATABASE_NAME)
        print(f"\nFound {len(study_names)} studies in database")
    else:
        study_names = STUDY_NAMES
        print(f"\nEvaluating {len(study_names)} specified studies")
    print(f"Studies: {study_names}")
    
    # Evaluate each study
    all_results = []
    
    for study_name in study_names:
        # Try to extract n_units from study name (e.g., "100_units_study" → 100)
        # If the study name contains a number followed by "units", use that
        n_units_override = None
        try:
            parts = study_name.lower().split('_')
            for i, part in enumerate(parts):
                if part == 'units' and i > 0:
                    n_units_override = int(parts[i-1])
                    break
                elif part.endswith('units'):
                    n_units_override = int(part.replace('units', ''))
                    break
        except ValueError:
            pass
        
        # Also try format like "100_units"
        if n_units_override is None:
            try:
                if '_units' in study_name.lower():
                    n_units_override = int(study_name.lower().split('_units')[0].split('_')[-1])
            except ValueError:
                pass
        
        result = evaluate_study(
            DATABASE_NAME, study_name, device,
            RANDOM_SEEDS, n_units_override=n_units_override
        )
        if result is not None:
            all_results.append(result)
    
    if len(all_results) == 0:
        print("\nNo successful evaluations!")
        return None
    
    # Create results dataframe
    df = pd.DataFrame(all_results)
    
    # Sort by test accuracy
    df = df.sort_values('test_mean', ascending=False)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY OF ALL STUDIES")
    print("="*70)
    
    # Print table
    print("\n" + df[['study_name', 'optuna_best_value', 'valid_mean', 'valid_std', 
                    'test_mean', 'test_std', 'n_units', 'classifier_params']].to_string(index=False))
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{parent_dir}/results/mnist_multi_study_comparison_{timestamp}.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")
    
    # Create comparison plot if we have multiple studies with different n_units
    if len(df) > 1 and df['n_units'].nunique() > 1:
        # Create line plot of test accuracy vs n_units
        df_sorted = df.sort_values('n_units')
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(df_sorted['n_units'], df_sorted['test_mean'], 
                     yerr=df_sorted['test_std'], 
                     marker='o', markersize=8, capsize=5, capthick=2, 
                     linewidth=2, elinewidth=1.5)
        
        plt.xlabel('Number of Units', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.title(f'MNIST Test Accuracy vs Network Size ({DATABASE_NAME})', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Set x-axis ticks to actual n_units values
        plt.xticks(df_sorted['n_units'])
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{parent_dir}/plots/mnist_multi_study_comparison_{timestamp}.png"
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_filename}")
        plt.close()
    else:
        # Bar plot for comparing different studies
        plt.figure(figsize=(12, 6))
        x_pos = range(len(df))
        plt.bar(x_pos, df['test_mean'], yerr=df['test_std'], capsize=5, alpha=0.7)
        plt.xticks(x_pos, [s[:30] + '...' if len(s) > 30 else s for s in df['study_name']], 
                   rotation=45, ha='right')
        plt.xlabel('Study Name', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.title(f'MNIST Test Accuracy Comparison ({DATABASE_NAME})', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{parent_dir}/plots/mnist_multi_study_comparison_{timestamp}.png"
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
        print("FINAL RANKING (by test accuracy)")
        print("="*70)
        for i, (_, row) in enumerate(results_df.iterrows()):
            print(f"{i+1}. {row['study_name'][:50]}")
            print(f"   Test: {row['test_mean']:.4f}±{row['test_std']:.4f}, "
                  f"Valid: {row['valid_mean']:.4f}±{row['valid_std']:.4f}, "
                  f"Units: {row['n_units']}, Params: {row['classifier_params']}")
