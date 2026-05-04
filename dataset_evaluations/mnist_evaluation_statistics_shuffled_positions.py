#%%
"""
MNIST Evaluation with Shuffled Position Features

This script tests whether the ORDER/STRUCTURE of position features matters,
or if it's just their statistics. If positions truly encode class information,
randomly shuffling them should destroy performance.

Key idea: Use positions but randomly permute them across units, destroying
the spatial structure while keeping marginal statistics intact.
"""

import os
import sys
import time
import numpy as np
import torch
import random
from tqdm import tqdm

# Fix matplotlib backend issues - use non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
# Configuration
RANDOM_SEEDS = [33, 42, 123, 456, 789]  # Multiple seeds for statistics
STUDY_NAME = "not_aligned_w_input_w_connections_actual_100_units_last_readout_no_tanh"
DATABASE_NAME = "unicycle_mnist_all_digits_logreg"

# Model configuration flags
ALIGNED_ORIENTATIONS = False
ANG_INPUT = True
ANG_CONNECTIONS = True

# MNIST specific configurations
N_UNITS = 100
N_CLASSES = 10

# Shuffling configuration
SHUFFLE_POSITIONS = True  # Set to True to shuffle x,y positions
SHUFFLE_SEED = 42  # Fixed seed for shuffling permutation (same across all evaluations)

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
def load_best_params():
    """Load best parameters from Optuna study"""
    storage_name = f"sqlite:///{parent_dir}/optuna_databases/{DATABASE_NAME}.db"
    study = optuna.create_study(storage=storage_name, study_name=STUDY_NAME, 
                               direction='maximize', load_if_exists="True")
    return study.best_params

#%%
def create_input_maps(params, n_units, seed):
    """Create input maps with given seed for reproducibility"""
    set_seed(seed)
    
    # Linear input map (1D for MNIST)
    lin_input_map = torch.zeros(1, n_units)
    num_non_zero = params['non_zero_elements']
    non_zero_indices = torch.randperm(n_units)[:num_non_zero]
    magnitude_min = params['magnitude_min']
    magnitude_max = params['magnitude_max']
    lin_input_map[0, non_zero_indices] = torch.rand(num_non_zero) * (magnitude_max - magnitude_min) + magnitude_min
    
    # Angular input map
    ang_input_map = torch.zeros(1, n_units)
    if ANG_INPUT:
        num_non_zero_ang = params['non_zero_elements_ang']
        non_zero_indices_ang = torch.randperm(n_units)[:num_non_zero_ang]
        magnitude_min_ang = params['magnitude_min_ang']
        magnitude_max_ang = params['magnitude_max_ang']
        ang_input_map[0, non_zero_indices_ang] = torch.rand(num_non_zero_ang) * (magnitude_max_ang - magnitude_min_ang) + magnitude_min_ang
    
    return lin_input_map, ang_input_map

#%%
def initialize_model(params, lin_input_map, ang_input_map, device):
    """Initialize and configure the model"""
    n_units = N_UNITS
    
    # Extract parameters
    n_connections_fraction = params['n_connections_fraction']
    n_connections = int(n_units * n_connections_fraction)
    anchor_con_fraction = params['anchor_con_fraction']
    n_connections_anchor = int(n_units * anchor_con_fraction)
    
    # Angular connections
    if ANG_CONNECTIONS:
        n_connections_ang_fraction = params['n_connections_ang_fraction']
        anchor_con_fraction_ang = params['anchor_con_fraction_ang']
        n_connections_anchor_ang = int(anchor_con_fraction_ang * n_units)
        n_connections_ang = int(n_connections_ang_fraction * n_units)
    else:
        n_connections_anchor_ang = 0
        n_connections_ang = 0
    
    # Create model
    model = UnicycleReservoir(
        n_inp=1, n_units=n_units, dt=params['dt'], n_out=N_CLASSES,
        lin_input_map=lin_input_map,
        lin_stiff_min=params['lin_stiff_min'], lin_stiff_max=params['lin_stiff_max'],
        ang_stiff_min=params['ang_stiff_min'], ang_stiff_max=params['ang_stiff_max'],
        lin_damping_min=params['lin_damping_min'], lin_damping_max=params['lin_damping_max'],
        ang_damping_min=params['ang_damping_min'], ang_damping_max=params['ang_damping_max'],
        eq_dist_min=params['eq_dist_min'], eq_dist_max=params['eq_dist_max'],
        eq_dist_min_ang=params['eq_dist_min_ang'], eq_dist_max_ang=params['eq_dist_max_ang'],
        n_connections=n_connections, n_connections_anchor=n_connections_anchor,
        n_past_steps_readout=0, n_connections_ang=n_connections_ang,
        n_connections_anchor_ang=n_connections_anchor_ang,
        inp_bias=params['inp_bias'], ang_input_map=ang_input_map
    ).to(device)
    
    return model

#%%
def setup_initial_states(model, params, device):
    """Setup initial states for the model"""
    bs_train = 500
    
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
    
    if not ALIGNED_ORIENTATIONS:
        model.theta_init[:,:] = torch.rand(model.theta_init.size()) * (4*torch.pi) - 2*torch.pi
        model.theta_init[:,0] = 0
    else:
        model.theta_init[:,:] = torch.rand(1) * (4*torch.pi) - 2*torch.pi
    
    return model

#%%
def run_washup(model, params, device):
    """Run washup phase and return final states"""
    washup = params['washup_steps']
    
    x = model.x_init[0:1,:]
    z = model.z_init[0:1,:]
    theta = model.theta_init[0:1,:]
    s = model.s_init[0:1,:]
    omega = model.omega_init[0:1,:]
    
    # MNIST uses 1D input
    u_lin = torch.zeros((1, washup, 1), device=device)
    u_ang = torch.zeros_like(u_lin, device=device)
    
    for t in range(u_lin.size()[1]):
        linear_input = (u_lin[:, t]) @ model.lin_input_map
        angular_input = u_ang[:, t] @ model.ang_input_map
        x, z, theta, s, omega = model.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)
    
    return x, z, theta, s, omega

#%%
def shuffle_positions(mid_states, shuffle_perm):
    """
    Shuffle the x and y position features to destroy spatial structure.
    
    Args:
        mid_states: (batch, 400) tensor with [x1,x2,...,x100,y1,y2,...,y100,theta1,...,s1,...]
        shuffle_perm: permutation indices for shuffling
    
    Returns:
        Shuffled mid_states where x and y positions are permuted
    """
    # Extract positions (first 200 features: x and y)
    positions = mid_states[:, :200].clone()
    other_features = mid_states[:, 200:].clone()
    
    # Apply same permutation to positions
    positions_shuffled = positions[:, shuffle_perm]
    
    # Concatenate back
    return torch.cat([positions_shuffled, other_features], dim=1)

#%%
@torch.no_grad()
def test_esn(data_loader, model, classifier, scaler, device, shuffle_perm):
    """Test ESN performance"""
    activations, ys = [], []
    for images, labels in data_loader:
        images = images.reshape(images.shape[0], 1, 784)
        images = images.permute(0, 2, 1)
        images = images.to(device)
        labels = labels.to(device)
        states_list, output, mid_states = model(images, images)
        
        # Shuffle positions if configured
        if SHUFFLE_POSITIONS:
            mid_states = shuffle_positions(mid_states, shuffle_perm)
        
        # Use only positions (first 200 features)
        mid_states = mid_states[:, :200]
        
        activations.append(mid_states.cpu())
        ys.append(labels.cpu())
    
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)

#%%
def single_evaluation(params, seed, device, train_loader, valid_loader, test_loader):
    """Run a single evaluation with given seed"""
    print(f"Running evaluation with seed {seed}")
    
    # Create fixed shuffling permutation (same for all seeds for consistency)
    np.random.seed(SHUFFLE_SEED)
    shuffle_perm = np.random.permutation(200)  # Permute all 200 position features
    shuffle_perm = torch.from_numpy(shuffle_perm).long()
    
    # Set seed and create input maps
    lin_input_map, ang_input_map = create_input_maps(params, N_UNITS, seed)
    
    # Initialize model
    model = initialize_model(params, lin_input_map, ang_input_map, device)
    model = setup_initial_states(model, params, device)
    
    # Run washup
    x, z, theta, s, omega = run_washup(model, params, device)
    
    # Set initial states after washup
    model.set_init_states(500, x, z, theta, s, omega)
    
    # Train logistic regression classifier
    activations, ys = [], []
    for images, labels in tqdm(train_loader, desc=f"Training (seed {seed})"):
        images = images.reshape(images.shape[0], 1, 784)
        images = images.permute(0, 2, 1)
        images = images.to(device)
        labels = labels.to(device)
        states_list, output, mid_states = model(images, images)
        
        # Shuffle positions if configured
        if SHUFFLE_POSITIONS:
            mid_states = shuffle_positions(mid_states, shuffle_perm)
        
        # Use only positions (first 200 features)
        mid_states = mid_states[:, :200]
        
        activations.append(mid_states.detach().cpu())
        ys.append(labels.cpu())
    
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    
    # Check for NaN values
    if np.isnan(activations).any():
        print(f"Warning: NaN values detected in activations for seed {seed}")
        return None, None, None
    
    # Standardize and train classifier
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000).fit(activations, ys)
    
    # Count classifier parameters
    n_classifier_params = count_classifier_params(classifier)
    
    # Evaluate
    valid_score = test_esn(valid_loader, model, classifier, scaler, device, shuffle_perm)
    test_score = test_esn(test_loader, model, classifier, scaler, device, shuffle_perm)
    
    print(f"Seed {seed}: Valid={valid_score:.4f}, Test={test_score:.4f}, Classifier params={n_classifier_params}")
    
    return valid_score, test_score, n_classifier_params

#%%
def main():
    """Main evaluation function"""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load best parameters
    params = load_best_params()
    print(f"Loaded parameters from study: {STUDY_NAME}")
    print(f"Configuration: aligned_orientations={ALIGNED_ORIENTATIONS}, ang_input={ANG_INPUT}, ang_connections={ANG_CONNECTIONS}")
    print(f"Network size: {N_UNITS} units, Classes: {N_CLASSES}")
    print(f"Shuffle positions: {SHUFFLE_POSITIONS} (seed: {SHUFFLE_SEED})")
    bs = 500
    # Load data
    root = parent_dir + '/data/'
    train_loader, valid_loader, test_loader = get_mnist_data(
        bs_train=bs, 
        bs_test=bs, 
        classes=[0,1,2,3,4,5,6,7,8,9], 
        new_fraction=1.0, 
        test_fraction=1.0, 
        path=root
    )
    
    # Run evaluations
    valid_scores = []
    test_scores = []
    classifier_params = None  # Will be same for all runs
    
    print(f"\nRunning {len(RANDOM_SEEDS)} evaluations...")
    for seed in RANDOM_SEEDS:
        valid_score, test_score, n_classifier_params = single_evaluation(params, seed, device, train_loader, valid_loader, test_loader)
        
        if valid_score is not None and test_score is not None:
            valid_scores.append(valid_score)
            test_scores.append(test_score)
            if classifier_params is None:
                classifier_params = n_classifier_params
        else:
            print(f"Skipping seed {seed} due to NaN values")
    
    # Calculate statistics
    if len(valid_scores) > 0:
        valid_mean = np.mean(valid_scores)
        valid_std = np.std(valid_scores)
        test_mean = np.mean(test_scores)
        test_std = np.std(test_scores)
        
        print("\n" + "="*50)
        print("MNIST EVALUATION STATISTICS - SHUFFLED POSITIONS")
        print("="*50)
        print(f"Number of successful runs: {len(valid_scores)}/{len(RANDOM_SEEDS)}")
        print(f"Classifier trainable parameters: {classifier_params}")
        print(f"Validation accuracy: {valid_mean:.4f} ± {valid_std:.4f}")
        print(f"Test accuracy: {test_mean:.4f} ± {test_std:.4f}")
        print(f"Valid scores: {[f'{s:.4f}' for s in valid_scores]}")
        print(f"Test scores: {[f'{s:.4f}' for s in test_scores]}")
        
        # Save results
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"{parent_dir}/results/mnist_stats_shuffled_positions_{timestamp}.txt"
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)
        with open(results_filename, 'w') as f:
            f.write(f"MNIST Evaluation Statistics - SHUFFLED POSITIONS\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  - Network units: {N_UNITS}\n")
            f.write(f"  - Classes: {N_CLASSES}\n")
            f.write(f"  - Shuffle positions: {SHUFFLE_POSITIONS}\n")
            f.write(f"  - Shuffle seed: {SHUFFLE_SEED}\n\n")
            
            f.write(f"Results Summary:\n")
            f.write(f"  - Validation accuracy: {valid_mean:.4f} ± {valid_std:.4f}\n")
            f.write(f"  - Test accuracy: {test_mean:.4f} ± {test_std:.4f}\n\n")
            
            f.write("Individual Results:\n")
            for i, seed in enumerate(RANDOM_SEEDS[:len(valid_scores)]):
                f.write(f"  Seed {seed:4d}: Valid={valid_scores[i]:.4f}, Test={test_scores[i]:.4f}\n")
        
        print(f"\nResults saved to: {results_filename}")
        
        return {
            'valid_mean': valid_mean,
            'test_mean': test_mean,
            'valid_std': valid_std,
            'test_std': test_std
        }
    else:
        print("No successful evaluations completed!")
        return None

#%%
if __name__ == "__main__":
    results = main()
