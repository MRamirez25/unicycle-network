#%%
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

from utils import get_cifar_data, n_params, count_classifier_params
from unicycle_network import UnicycleReservoir

#%%
# Configuration
RANDOM_SEEDS = [33, 42, 123, 456, 789]# 999, 1337, 2024, 5555, 8888]  # Multiple seeds for statistics
STUDY_NAME = "only_last_state_as_readout_200_units_ang_input_ang_coupled_no_tanh"
DATABASE_NAME = "unicycle_nets_cifar10_logreg"

# Model configuration flags
ALIGNED_ORIENTATIONS = False
ANG_INPUT = True
ANG_CONNECTIONS = True

# CIFAR-10 specific configurations
N_UNITS = 200
N_INP = 96
N_CLASSES = 10

# Feature selection for readout (slicing mid_states)
# Set to None to use all features, or specify (start, stop) indices
# For CIFAR-10 with 200 units: mid_states has shape (batch, 200*4) = (batch, 800)
# Common choices: (0, 400) for x,y positions only, (0, 800) or None for all features
FEATURE_SLICE_START = 0
FEATURE_SLICE_STOP = 600

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
def create_input_maps(params, n_units, n_inp, seed):
    """Create input maps with given seed for reproducibility"""
    set_seed(seed)
    
    # Linear input map (2D for CIFAR-10)
    lin_input_map = torch.zeros(n_inp, n_units)
    total_elements = n_inp * n_units
    non_zero_fraction = params['non_zero_fraction']
    num_non_zero = int(total_elements * non_zero_fraction)
    
    # Flatten indices and randomly select
    flat_indices = torch.randperm(total_elements)[:num_non_zero]
    row_indices = flat_indices // n_units
    col_indices = flat_indices % n_units
    
    magnitude_min = params['magnitude_min']
    magnitude_max = params['magnitude_max']
    random_values = torch.rand(num_non_zero) * (magnitude_max - magnitude_min) + magnitude_min
    lin_input_map[row_indices, col_indices] = random_values
    
    # Angular input map
    ang_input_map = torch.zeros(n_inp, n_units)
    if ANG_INPUT:
        non_zero_fraction_ang = params['non_zero_fraction_ang']
        num_non_zero_ang = int(total_elements * non_zero_fraction_ang)
        
        flat_indices_ang = torch.randperm(total_elements)[:num_non_zero_ang]
        row_indices_ang = flat_indices_ang // n_units
        col_indices_ang = flat_indices_ang % n_units
        
        magnitude_min_ang = params['magnitude_min_ang']
        magnitude_max_ang = params['magnitude_max_ang']
        random_values_ang = torch.rand(num_non_zero_ang) * (magnitude_max_ang - magnitude_min_ang) + magnitude_min_ang
        ang_input_map[row_indices_ang, col_indices_ang] = random_values_ang
    
    return lin_input_map, ang_input_map

#%%
def initialize_model(params, lin_input_map, ang_input_map, device):
    """Initialize and configure the model"""
    n_units = N_UNITS
    n_inp = N_INP
    
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
        n_inp=n_inp, n_units=n_units, dt=params['dt'], n_out=N_CLASSES,
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
    bs_train = params['batch_size']
    
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
    
    # CIFAR-10 uses higher dimensional washup
    u_lin = torch.zeros((1, washup, N_INP), device=device)
    u_ang = torch.zeros_like(u_lin, device=device)
    
    for t in range(u_lin.size()[1]):
        linear_input = (u_lin[:, t]) @ model.lin_input_map
        angular_input = u_ang[:, t] @ model.ang_input_map
        x, z, theta, s, omega = model.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)
    
    return x, z, theta, s, omega

# Removed create_random_padding and process_cifar_batch functions
# Using the same direct approach as cifar10_ang_input_coupled.py

#%%
@torch.no_grad()
def test_esn(data_loader, model, classifier, scaler, device, rand_test, bs_test):
    """Test ESN performance - memory efficient version"""
    activations, ys = [], []
    
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Process CIFAR-10 images using FIXED batch size like working version
        images_processed = torch.cat((images.permute(0,2,1,3).reshape(bs_test,32,N_INP), 
                                    rand_test), dim=1)
        
        states_list, output, mid_states = model(images_processed, images_processed)
        
        # Apply feature slicing if configured
        if FEATURE_SLICE_STOP is not None:
            mid_states = mid_states[:, FEATURE_SLICE_START:FEATURE_SLICE_STOP]
        
        # Immediately move to CPU and clean up GPU tensors
        activations.append(mid_states.detach().cpu())
        ys.append(labels.detach().cpu())
        
        # Clean up GPU memory immediately after each batch
        del images, labels, images_processed, output, mid_states
        # Clear states_list which might contain many intermediate tensors
        if isinstance(states_list, list):
            for state in states_list:
                del state
        del states_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Check if we have any data
    if len(activations) == 0:
        print("Warning: No data processed in test_esn")
        return 0.0
    
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)

#%%
def single_evaluation(params, seed, device, train_loader, valid_loader, test_loader):
    """Run a single evaluation with given seed"""
    print(f"Running evaluation with seed {seed}")
    
    # Clear CUDA memory before each evaluation to prevent accumulation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated() / 1e6
        print(f"Initial GPU memory: {initial_memory:.1f} MB")
    
    # Set seed and create input maps
    lin_input_map, ang_input_map = create_input_maps(params, N_UNITS, N_INP, seed)
    
    # Initialize model
    model = initialize_model(params, lin_input_map, ang_input_map, device)
    model = setup_initial_states(model, params, device)
    
    # Run washup
    x, z, theta, s, omega = run_washup(model, params, device)
    
    # Set initial states after washup
    model.set_init_states(params['batch_size'], x, z, theta, s, omega)
    
    # Create random padding once and reuse (like in working version)
    rands = torch.randn(1, 1000 - 32, N_INP).to(device)
    rand_train = rands.repeat(params['batch_size'], 1, 1)
    rand_test = rands.repeat(params['batch_size'], 1, 1)
    
    # Train logistic regression classifier
    activations, ys = [], []
    bs_train = params['batch_size']  # Use fixed batch size like working version
    for images, labels in tqdm(train_loader, desc=f"Training (seed {seed})"):
        images = images.to(device)
        labels = labels.to(device)
        # Process CIFAR-10 images using FIXED batch size like working version
        images_processed = torch.cat((images.permute(0,2,1,3).reshape(bs_train,32,N_INP), 
                                    rand_train), dim=1)
        
        with torch.no_grad():  # Ensure no gradient tracking
            states_list, output, mid_states = model(images_processed, images_processed)
        
        # Apply feature slicing if configured
        if FEATURE_SLICE_STOP is not None:
            mid_states = mid_states[:, FEATURE_SLICE_START:FEATURE_SLICE_STOP]
        
        # Immediately move to CPU and clean up
        activations.append(mid_states.detach().cpu())
        ys.append(labels.detach().cpu())
        
        # Clean up GPU tensors after each batch to prevent accumulation
        del images, labels, images_processed, output, mid_states
        # Clear states_list which might contain many intermediate tensors
        if isinstance(states_list, list):
            for state in states_list:
                del state
        del states_list
        
        # Force garbage collection every few batches
        if len(activations) % 10 == 0:
            torch.cuda.empty_cache()
    
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    
    # Check for NaN values
    if np.isnan(activations).any():
        print(f"Warning: NaN values detected in activations for seed {seed}")
        return None, None, None
    
    # Standardize and train classifier
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=500).fit(activations, ys)
    
    # Count classifier parameters
    n_classifier_params = count_classifier_params(classifier)
    
    # Aggressively free memory after training
    del activations, ys, rand_train  # Remove large training tensors
    
    # Also clear intermediate model states that are no longer needed
    # del model.x_init, model.z_init, model.theta_init, model.s_init, model.omega_init
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    
    # Evaluate
    bs_test = params['batch_size']  # Use same batch size for test as working version
    print(f"Starting validation evaluation...")
    valid_score = test_esn(valid_loader, model, classifier, scaler, device, rand_test, bs_test)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU memory after validation: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    
    print(f"Starting test evaluation...")
    test_score = test_esn(test_loader, model, classifier, scaler, device, rand_test, bs_test)
    
    # Final cleanup - release all remaining tensors
    del model, classifier, scaler, rand_test, rands
    del lin_input_map, ang_input_map
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU memory after evaluation cleanup: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    
    print(f"Seed {seed}: Valid={valid_score:.4f}, Test={test_score:.4f}, Classifier params={n_classifier_params}")
    
    return valid_score, test_score, n_classifier_params

#%%
def main():
    """Main evaluation function"""
    # Clear CUDA memory at the beginning
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"CUDA memory cleared. Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Memory allocated before start: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load best parameters
    params = load_best_params()
    print(f"Loaded parameters from study: {STUDY_NAME}")
    print(f"Configuration: aligned_orientations={ALIGNED_ORIENTATIONS}, ang_input={ANG_INPUT}, ang_connections={ANG_CONNECTIONS}")
    print(f"Network size: {N_UNITS} units, Input dim: {N_INP}, Classes: {N_CLASSES}")
    
    # Load data
    train_loader, valid_loader, test_loader = get_cifar_data(params['batch_size'], params['batch_size'], 
                                                           new_fraction=0.5, test_fraction=0.5)
    
    # Debug data loader sizes
    print(f"Batch size: {params['batch_size']}")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Valid loader batches: {len(valid_loader)}")
    print(f"Test loader batches: {len(test_loader)}")
    
    if len(valid_loader) == 0:
        print("ERROR: Validation loader is empty!")
        return None
    if len(test_loader) == 0:
        print("ERROR: Test loader is empty!")
        return None
    
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
        print("CIFAR-10 EVALUATION STATISTICS")
        print("="*50)
        print(f"Number of successful runs: {len(valid_scores)}/{len(RANDOM_SEEDS)}")
        print(f"Classifier trainable parameters: {classifier_params}")
        print(f"Validation accuracy: {valid_mean:.4f} ± {valid_std:.4f}")
        print(f"Test accuracy: {test_mean:.4f} ± {test_std:.4f}")
        print(f"Valid scores: {[f'{s:.4f}' for s in valid_scores]}")
        print(f"Test scores: {[f'{s:.4f}' for s in test_scores]}")
        
        # Advanced visualization
        plt.figure(figsize=(15, 10))
        
        # 1. Box plot comparison
        plt.subplot(2, 3, 1)
        plt.boxplot([valid_scores, test_scores], labels=['Validation', 'Test'])
        plt.ylabel('Accuracy')
        plt.title('Score Distribution')
        plt.grid(True, alpha=0.3)
        
        # 2. Scores by run
        plt.subplot(2, 3, 2)
        plt.plot(range(len(valid_scores)), valid_scores, 'o-', label='Validation', alpha=0.7)
        plt.plot(range(len(test_scores)), test_scores, 's-', label='Test', alpha=0.7)
        plt.xlabel('Run Index')
        plt.ylabel('Accuracy')
        plt.title('Scores by Run')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Score histograms
        plt.subplot(2, 3, 3)
        plt.hist(valid_scores, alpha=0.7, label='Validation', bins=10, edgecolor='black')
        plt.hist(test_scores, alpha=0.7, label='Test', bins=10, edgecolor='black')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.title('Score Histograms')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Validation vs Test correlation
        plt.subplot(2, 3, 4)
        plt.scatter(valid_scores, test_scores, alpha=0.7)
        plt.xlabel('Validation Accuracy')
        plt.ylabel('Test Accuracy')
        plt.title('Validation vs Test Correlation')
        
        # Add correlation coefficient
        correlation = np.corrcoef(valid_scores, test_scores)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top')
        
        # Add diagonal line
        min_score = min(min(valid_scores), min(test_scores))
        max_score = max(max(valid_scores), max(test_scores))
        plt.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # 5. Score ranges
        plt.subplot(2, 3, 5)
        score_ranges = [max(valid_scores) - min(valid_scores), max(test_scores) - min(test_scores)]
        plt.bar(['Validation', 'Test'], score_ranges)
        plt.ylabel('Score Range')
        plt.title('Score Variability')
        plt.grid(True, alpha=0.3)
        
        # 6. Statistical summary
        plt.subplot(2, 3, 6)
        plt.axis('off')
        summary_text = f"""
CIFAR-10 Statistics Summary:

Mean ± Std:
• Validation: {valid_mean:.4f} ± {valid_std:.4f}
• Test: {test_mean:.4f} ± {test_std:.4f}

Range:
• Validation: [{min(valid_scores):.4f}, {max(valid_scores):.4f}]
• Test: [{min(test_scores):.4f}, {max(test_scores):.4f}]

Correlation: {correlation:.3f}

Network Config:
• Units: {N_UNITS}
• Input dim: {N_INP}
• Classes: {N_CLASSES}
• Classifier params: {classifier_params}
        """
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{parent_dir}/plots/cifar10_evaluation_stats_{STUDY_NAME}.png"
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_filename}")
        plt.close()
        
        return {
            'valid_scores': valid_scores,
            'test_scores': test_scores,
            'valid_mean': valid_mean,
            'valid_std': valid_std,
            'test_mean': test_mean,
            'test_std': test_std,
            'correlation': correlation,
            'seeds_used': RANDOM_SEEDS[:len(valid_scores)],
            'classifier_params': classifier_params,
            'config': {
                'n_units': N_UNITS,
                'n_inp': N_INP,
                'n_classes': N_CLASSES,
                'aligned_orientations': ALIGNED_ORIENTATIONS,
                'ang_input': ANG_INPUT,
                'ang_connections': ANG_CONNECTIONS
            }
        }
    else:
        print("No successful evaluations completed!")
        return None

#%%
if __name__ == "__main__":
    results = main()

# %%
# Optional: Display individual seed results in more detail
if 'results' in locals() and results is not None:
    print("\nDetailed Results:")
    print("-" * 50)
    for i, seed in enumerate(results['seeds_used']):
        print(f"Seed {seed:4d}: Valid={results['valid_scores'][i]:.4f}, Test={results['test_scores'][i]:.4f}")
    
    # Additional analysis
    print(f"\nBest performing seed: {results['seeds_used'][np.argmax(results['test_scores'])]}")
    print(f"Worst performing seed: {results['seeds_used'][np.argmin(results['test_scores'])]}")
    print(f"Most stable (lowest std): Validation={results['valid_std']:.4f}, Test={results['test_std']:.4f}")

# %%
# Optional: Save results to file
if 'results' in locals() and results is not None:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"{parent_dir}/results/cifar10_stats_results_{timestamp}.txt"
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    with open(results_filename, 'w') as f:
        f.write(f"CIFAR-10 Evaluation Statistics - {STUDY_NAME}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Network units: {results['config']['n_units']}\n")
        f.write(f"  - Input dimension: {results['config']['n_inp']}\n") 
        f.write(f"  - Classes: {results['config']['n_classes']}\n")
        f.write(f"  - Feature slice: [{FEATURE_SLICE_START}:{FEATURE_SLICE_STOP}]\n")
        f.write(f"  - Aligned orientations: {results['config']['aligned_orientations']}\n")
        f.write(f"  - Angular input: {results['config']['ang_input']}\n")
        f.write(f"  - Angular connections: {results['config']['ang_connections']}\n\n")
        
        f.write(f"Results Summary:\n")
        f.write(f"  - Validation accuracy: {results['valid_mean']:.4f} ± {results['valid_std']:.4f}\n")
        f.write(f"  - Test accuracy: {results['test_mean']:.4f} ± {results['test_std']:.4f}\n")
        f.write(f"  - Correlation: {results['correlation']:.4f}\n")
        f.write(f"  - Successful runs: {len(results['valid_scores'])}/{len(RANDOM_SEEDS)}\n\n")
        
        f.write("Individual Results:\n")
        for i, seed in enumerate(results['seeds_used']):
            f.write(f"  Seed {seed:4d}: Valid={results['valid_scores'][i]:.4f}, Test={results['test_scores'][i]:.4f}\n")
    
    print(f"\nResults saved to: {results_filename}")
