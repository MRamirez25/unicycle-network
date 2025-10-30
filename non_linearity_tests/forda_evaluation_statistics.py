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

from utils import get_FordA_data, n_params
from unicycle_network_class import UnicycleReservoir

#%%
# Configuration
RANDOM_SEEDS = [33, 42, 123, 456, 789]  # Multiple seeds for statistics
STUDY_NAME = "only_last_state_as_readout_20_units_not_aligned_less_ang_magnitude_no_tanh"
DATABASE_NAME = "unicycle_nets_forda_logreg"

# Model configuration flags
ALIGNED_ORIENTATIONS = False
ANG_INPUT = True
ANG_CONNECTIONS = False

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
    
    # Linear input map
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
        non_zero_indices = torch.randperm(n_units)[:num_non_zero_ang]
        magnitude_min_ang = params['magnitude_min_ang']
        magnitude_max_ang = params['magnitude_max_ang']
        ang_input_map[0, non_zero_indices] = torch.rand(num_non_zero_ang) * (magnitude_max_ang - magnitude_min_ang) + magnitude_min_ang
    
    return lin_input_map, ang_input_map

#%%
def initialize_model(params, lin_input_map, ang_input_map, device):
    """Initialize and configure the model"""
    n_units = 20
    
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
        n_inp=1, n_units=n_units, dt=params['dt'], n_out=2,
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
        # mid_states = mid_states[:, :60]  # Take first 40 features, not first 40 samples
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
    
    # Set seed and create input maps
    lin_input_map, ang_input_map = create_input_maps(params, 20, seed)
    
    # Initialize model
    model = initialize_model(params, lin_input_map, ang_input_map, device)
    model = setup_initial_states(model, params, device)
    
    # Run washup
    x, z, theta, s, omega = run_washup(model, params, device)
    
    # Set initial states after washup
    model.set_init_states(params['batch_size'], x, z, theta, s, omega)
    
    # Train logistic regression classifier
    activations, ys = [], []
    for x, labels in tqdm(train_loader, desc=f"Training (seed {seed})"):
        x = x.to(device)
        print(f"Input batch shape: {x.shape}")  # Debug: print input shape
        labels = labels.to(device)
        states_list, output, mid_states = model(x, x)
        # mid_states = mid_states[:, :60]  # Take first 40 features, not first 40 samples

        activations.append(mid_states.detach().cpu())
        ys.append(labels.cpu())
    
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    
    # Check for NaN values
    if np.isnan(activations).any():
        print(f"Warning: NaN values detected in activations for seed {seed}")
        return None, None
    
    # Standardize and train classifier
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000).fit(activations, ys)
    
    # Evaluate
    valid_score = test_esn(valid_loader, model, classifier, scaler, device)
    test_score = test_esn(test_loader, model, classifier, scaler, device)
    
    print(f"Seed {seed}: Valid={valid_score:.4f}, Test={test_score:.4f}")
    
    return valid_score, test_score

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
    
    # Load data
    train_loader, valid_loader, test_loader = get_FordA_data(params['batch_size'], params['batch_size'])
    
    # Run evaluations
    valid_scores = []
    test_scores = []
    
    print(f"\nRunning {len(RANDOM_SEEDS)} evaluations...")
    for seed in RANDOM_SEEDS:
        valid_score, test_score = single_evaluation(params, seed, device, train_loader, valid_loader, test_loader)
        
        if valid_score is not None and test_score is not None:
            valid_scores.append(valid_score)
            test_scores.append(test_score)
        else:
            print(f"Skipping seed {seed} due to NaN values")
    
    # Calculate statistics
    if len(valid_scores) > 0:
        valid_mean = np.mean(valid_scores)
        valid_std = np.std(valid_scores)
        test_mean = np.mean(test_scores)
        test_std = np.std(test_scores)
        
        print("\n" + "="*50)
        print("EVALUATION STATISTICS")
        print("="*50)
        print(f"Number of successful runs: {len(valid_scores)}/{len(RANDOM_SEEDS)}")
        print(f"Validation accuracy: {valid_mean:.4f} ± {valid_std:.4f}")
        print(f"Test accuracy: {test_mean:.4f} ± {test_std:.4f}")
        print(f"Valid scores: {[f'{s:.4f}' for s in valid_scores]}")
        print(f"Test scores: {[f'{s:.4f}' for s in test_scores]}")
        
        # Calculate correlation between validation and test scores
        correlation = np.corrcoef(valid_scores, test_scores)[0, 1]
        
        # Enhanced visualization
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
        
        # 3. Scatter plot of validation vs test
        plt.subplot(2, 3, 3)
        plt.scatter(valid_scores, test_scores, alpha=0.7)
        plt.xlabel('Validation Accuracy')
        plt.ylabel('Test Accuracy')
        plt.title(f'Valid vs Test (r={correlation:.3f})')
        # Add diagonal line
        min_val = min(min(valid_scores), min(test_scores))
        max_val = max(max(valid_scores), max(test_scores))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # 4. Histogram of scores
        plt.subplot(2, 3, 4)
        plt.hist(valid_scores, alpha=0.5, label='Validation', bins=5)
        plt.hist(test_scores, alpha=0.5, label='Test', bins=5)
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.title('Score Histogram')
        plt.legend()
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
FordA Statistics Summary:

Mean ± Std:
• Validation: {valid_mean:.4f} ± {valid_std:.4f}
• Test: {test_mean:.4f} ± {test_std:.4f}

Range:
• Validation: [{min(valid_scores):.4f}, {max(valid_scores):.4f}]
• Test: [{min(test_scores):.4f}, {max(test_scores):.4f}]

Correlation: {correlation:.3f}

Network Config:
• Units: 20
• Classes: 2
        """
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot instead of showing it (since we're using Agg backend)
        plot_filename = f"{parent_dir}/plots/forda_evaluation_stats_{STUDY_NAME}.png"
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_filename}")
        plt.close()  # Close the figure to free memory
        
        return {
            'valid_scores': valid_scores,
            'test_scores': test_scores,
            'valid_mean': valid_mean,
            'valid_std': valid_std,
            'test_mean': test_mean,
            'test_std': test_std,
            'correlation': correlation,
            'seeds_used': RANDOM_SEEDS[:len(valid_scores)],
            'config': {
                'n_units': 20,
                'n_classes': 2,
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
    results_filename = f"{parent_dir}/results/forda_stats_results_{timestamp}.txt"
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    with open(results_filename, 'w') as f:
        f.write(f"FordA Evaluation Statistics - {STUDY_NAME}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Network units: {results['config']['n_units']}\n")
        f.write(f"  - Classes: {results['config']['n_classes']}\n")
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
    
    print(f"Results saved to: {results_filename}")
