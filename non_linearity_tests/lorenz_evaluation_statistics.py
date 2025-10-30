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

from sklearn.linear_model import Ridge
from sklearn import preprocessing
import optuna

#%%
# Add the parent directory to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import get_lorenz, n_params
from unicycle_network_class import UnicycleReservoir

#%%
# Configuration
RANDOM_SEEDS = [33, 42, 123, 456, 789]  # Multiple seeds for statistics
STUDY_NAME = "lorenz_prediction_esn_lag25"
DATABASE_NAME = "unicycle_nets_lorenz"

# Model configuration flags (will be loaded from params)
ALIGNED_ORIENTATIONS = None  # Will be set from params
ANG_INPUT = None  # Will be set from params
ANG_CONNECTIONS = None  # Will be set from params

# Lorenz specific configurations
N_LORENZ = 5  # Lorenz system dimension
LAG = 25
WASHOUT = 200

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
                               direction='minimize', load_if_exists="True")
    return study.best_params

#%%
def create_input_maps(params, n_units, n_inp, seed):
    """Create input maps with given seed for reproducibility"""
    set_seed(seed)
    
    # Linear input map (for Lorenz)
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
    if params.get('ang_input', False):
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
    n_units = params['n_units']
    n_inp = N_LORENZ
    
    # Extract parameters
    n_connections_fraction = params['n_connections_fraction']
    n_connections = int(n_units * n_connections_fraction)
    anchor_con_fraction = params['anchor_con_fraction']
    n_connections_anchor = int(n_units * anchor_con_fraction)
    
    # Angular connections
    if params.get('ang_connections', False):
        n_connections_ang_fraction = params['n_connections_ang_fraction']
        anchor_con_fraction_ang = params['anchor_con_fraction_ang']
        n_connections_anchor_ang = int(anchor_con_fraction_ang * n_units)
        n_connections_ang = int(n_connections_ang_fraction * n_units)
    else:
        n_connections_anchor_ang = 0
        n_connections_ang = 0
    
    # Create model
    model = UnicycleReservoir(
        n_inp=n_inp, n_units=n_units, dt=params['dt'], n_out=n_inp,  # n_out = n_inp for time series prediction
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
def setup_initial_states(model, params, device, batch_size):
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
    
    if not params.get('aligned_orientations', False):
        model.theta_init[:,:] = torch.rand(model.theta_init.size()) * (4*torch.pi) - 2*torch.pi
    else:
        model.theta_init[:,:] = torch.rand(1) * (4*torch.pi) - 2*torch.pi
    
    return model

#%%
def run_washup(model, params, device, batch_size):
    """Run washup phase and return final states"""
    washout = params['washout']
    
    x = model.x_init[0:1,:]
    z = model.z_init[0:1,:]
    theta = model.theta_init[0:1,:]
    s = model.s_init[0:1,:]
    omega = model.omega_init[0:1,:]
    
    # Lorenz uses multi-dimensional input
    u_lin = torch.zeros((1, washout, N_LORENZ), device=device)
    u_ang = torch.zeros_like(u_lin, device=device)
    
    for t in range(u_lin.size()[1]):
        linear_input = (u_lin[:, t]) @ model.lin_input_map
        angular_input = u_ang[:, t] @ model.ang_input_map
        x, z, theta, s, omega = model.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)
    
    return x, z, theta, s, omega

#%%
@torch.no_grad()
def test_esn(dataset, model, classifier, scaler, device, params):
    """Test ESN performance for Lorenz prediction"""
    activations, targets = [], []
    washout = params['washout']
    
    for batch_idx, batch_data in enumerate(dataset):
        # Process data like in unicycle_optuna_lorenz.py
        target = batch_data[:, (LAG+washout):].numpy()  # Shape: (batch, time_steps, 5)
        input_data = batch_data[:, :(2000+washout)].to(device)  # Shape: (batch, time_steps, 5)
        
        states_list, _, _ = model(input_data, input_data)
        
        # Process activations - concatenate all states from states_list
        batch_activations = torch.stack(states_list, dim=1)  # (batch, time, n_units*5)
        batch_activations = batch_activations.cpu().numpy()
        
        # Remove washout period from activations to match target
        batch_activations = batch_activations[:, washout:, :]  # Remove first washout time steps
        
        activations.append(batch_activations)
        targets.append(target)
    
    activations = np.concatenate(activations, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Reshape for prediction
    activations = activations.reshape(-1, activations.shape[-1])
    targets = targets.reshape(-1, targets.shape[-1])
    
    # Transform and predict
    activations = scaler.transform(activations)
    predictions = classifier.predict(activations)
    
    # Calculate NRMSE
    mse = np.mean(np.square(predictions - targets))
    rmse = np.sqrt(mse)
    norm = np.sqrt(np.square(targets).mean())
    nrmse = rmse / (norm + 1e-9)
    
    return nrmse

#%%
def single_evaluation(params, seed, device, train_dataset, valid_dataset, test_dataset):
    """Run a single evaluation with given seed"""
    print(f"Running evaluation with seed {seed}")
    
    # Set seed and create input maps
    lin_input_map, ang_input_map = create_input_maps(params, params['n_units'], N_LORENZ, seed)
    
    # Initialize model
    model = initialize_model(params, lin_input_map, ang_input_map, device)
    
    # Use batch size from dataset
    batch_size = train_dataset[0].shape[0]
    model = setup_initial_states(model, params, device, batch_size)
    
    # Run washup
    x, z, theta, s, omega = run_washup(model, params, device, batch_size)
    
    # Set initial states after washup
    model.set_init_states(batch_size, x, z, theta, s, omega)
    
    # Train Ridge regression classifier
    activations, targets = [], []
    washout = params['washout']
    
    for batch_idx, batch_data in enumerate(tqdm(train_dataset, desc=f"Training (seed {seed})")):
        # Process data like in unicycle_optuna_lorenz.py
        target = batch_data[:, (LAG+washout):].numpy()  # Shape: (batch, time_steps, 5)
        input_data = batch_data[:, :(2000+washout)].to(device)  # Shape: (batch, time_steps, 5)
        
        states_list, _, _ = model(input_data, input_data)
        
        # Process activations - concatenate all states from states_list
        batch_activations = torch.stack(states_list, dim=1)  # (batch, time, n_units*5)
        batch_activations = batch_activations.detach().cpu().numpy()
        
        # Remove washout period from activations to match target
        batch_activations = batch_activations[:, washout:, :]  # Remove first washout time steps
        
        activations.append(batch_activations)
        targets.append(target)
    
    activations = np.concatenate(activations, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Reshape for regression
    activations = activations.reshape(-1, activations.shape[-1])
    targets = targets.reshape(-1, targets.shape[-1])
    
    # Check for NaN values
    if np.isnan(activations).any():
        print(f"Warning: NaN values detected in activations for seed {seed}")
        return None, None
    
    # Standardize and train classifier
    scaler = preprocessing.StandardScaler().fit(activations)
    activations_scaled = scaler.transform(activations)
    classifier = Ridge(alpha=params['ridge_alpha']).fit(activations_scaled, targets)
    
    # Evaluate
    valid_nrmse = test_esn(valid_dataset, model, classifier, scaler, device, params)
    test_nrmse = test_esn(test_dataset, model, classifier, scaler, device, params)
    
    print(f"Seed {seed}: Valid NRMSE={valid_nrmse:.6f}, Test NRMSE={test_nrmse:.6f}")
    
    return valid_nrmse, test_nrmse

#%%
def main():
    """Main evaluation function"""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load best parameters
    params = load_best_params()
    print(f"Loaded parameters from study: {STUDY_NAME}")
    
    # Extract configuration from params
    aligned_orientations = params.get('aligned_orientations', False)
    ang_input = params.get('ang_input', False)
    ang_connections = params.get('ang_connections', False)
    
    print(f"Configuration: aligned_orientations={aligned_orientations}, ang_input={ang_input}, ang_connections={ang_connections}")
    print(f"Network size: {params['n_units']} units, Lorenz dim: {N_LORENZ}, Lag: {LAG}")
    
    # Load Lorenz data - create separate datasets like in unicycle_optuna_lorenz.py
    train_dataset = get_lorenz(N=N_LORENZ, F=8, lag=LAG, washout=params['washout'])
    valid_dataset = get_lorenz(N=N_LORENZ, F=8, lag=LAG, washout=params['washout'])
    test_dataset = get_lorenz(N=N_LORENZ, F=8, lag=LAG, washout=params['washout'])
    
    # Convert to list of tensors for batch processing like in the original
    train_dataset = [train_dataset]
    valid_dataset = [valid_dataset]
    test_dataset = [test_dataset]
    
    # Run evaluations
    valid_nrmses = []
    test_nrmses = []
    
    print(f"\nRunning {len(RANDOM_SEEDS)} evaluations...")
    for seed in RANDOM_SEEDS:
        valid_nrmse, test_nrmse = single_evaluation(params, seed, device, train_dataset, valid_dataset, test_dataset)
        
        if valid_nrmse is not None and test_nrmse is not None:
            valid_nrmses.append(valid_nrmse)
            test_nrmses.append(test_nrmse)
        else:
            print(f"Skipping seed {seed} due to NaN values")
    
    # Calculate statistics
    if len(valid_nrmses) > 0:
        valid_mean = np.mean(valid_nrmses)
        valid_std = np.std(valid_nrmses)
        test_mean = np.mean(test_nrmses)
        test_std = np.std(test_nrmses)
        
        print("\n" + "="*50)
        print("LORENZ EVALUATION STATISTICS")
        print("="*50)
        print(f"Number of successful runs: {len(valid_nrmses)}/{len(RANDOM_SEEDS)}")
        print(f"Validation NRMSE: {valid_mean:.6f} ± {valid_std:.6f}")
        print(f"Test NRMSE: {test_mean:.6f} ± {test_std:.6f}")
        print(f"Valid NRMSEs: {[f'{s:.6f}' for s in valid_nrmses]}")
        print(f"Test NRMSEs: {[f'{s:.6f}' for s in test_nrmses]}")
        
        # Advanced visualization
        plt.figure(figsize=(15, 10))
        
        # 1. Box plot comparison
        plt.subplot(2, 3, 1)
        plt.boxplot([valid_nrmses, test_nrmses], labels=['Validation', 'Test'])
        plt.ylabel('NRMSE')
        plt.title('Score Distribution')
        plt.grid(True, alpha=0.3)
        
        # 2. Scores by run
        plt.subplot(2, 3, 2)
        plt.plot(range(len(valid_nrmses)), valid_nrmses, 'o-', label='Validation', alpha=0.7)
        plt.plot(range(len(test_nrmses)), test_nrmses, 's-', label='Test', alpha=0.7)
        plt.xlabel('Run Index')
        plt.ylabel('NRMSE')
        plt.title('Scores by Run')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Score histograms
        plt.subplot(2, 3, 3)
        plt.hist(valid_nrmses, alpha=0.7, label='Validation', bins=10, edgecolor='black')
        plt.hist(test_nrmses, alpha=0.7, label='Test', bins=10, edgecolor='black')
        plt.xlabel('NRMSE')
        plt.ylabel('Frequency')
        plt.title('Score Histograms')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Validation vs Test correlation
        plt.subplot(2, 3, 4)
        plt.scatter(valid_nrmses, test_nrmses, alpha=0.7)
        plt.xlabel('Validation NRMSE')
        plt.ylabel('Test NRMSE')
        plt.title('Validation vs Test Correlation')
        
        # Add correlation coefficient
        correlation = np.corrcoef(valid_nrmses, test_nrmses)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top')
        
        # Add diagonal line
        min_score = min(min(valid_nrmses), min(test_nrmses))
        max_score = max(max(valid_nrmses), max(test_nrmses))
        plt.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # 5. Score ranges
        plt.subplot(2, 3, 5)
        score_ranges = [max(valid_nrmses) - min(valid_nrmses), max(test_nrmses) - min(test_nrmses)]
        plt.bar(['Validation', 'Test'], score_ranges)
        plt.ylabel('NRMSE Range')
        plt.title('Score Variability')
        plt.grid(True, alpha=0.3)
        
        # 6. Statistical summary
        plt.subplot(2, 3, 6)
        plt.axis('off')
        summary_text = f"""
Lorenz Statistics Summary:

Mean ± Std:
• Validation: {valid_mean:.6f} ± {valid_std:.6f}
• Test: {test_mean:.6f} ± {test_std:.6f}

Range:
• Validation: [{min(valid_nrmses):.6f}, {max(valid_nrmses):.6f}]
• Test: [{min(test_nrmses):.6f}, {max(test_nrmses):.6f}]

Correlation: {correlation:.3f}

Network Config:
• Units: {params['n_units']}
• Lorenz dim: {N_LORENZ}
• Lag: {LAG}
        """
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{parent_dir}/plots/lorenz_evaluation_stats_{STUDY_NAME}.png"
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_filename}")
        plt.close()
        
        return {
            'valid_nrmses': valid_nrmses,
            'test_nrmses': test_nrmses,
            'valid_mean': valid_mean,
            'valid_std': valid_std,
            'test_mean': test_mean,
            'test_std': test_std,
            'correlation': correlation,
            'seeds_used': RANDOM_SEEDS[:len(valid_nrmses)],
            'config': {
                'n_units': params['n_units'],
                'lorenz_dim': N_LORENZ,
                'lag': LAG,
                'aligned_orientations': aligned_orientations,
                'ang_input': ang_input,
                'ang_connections': ang_connections
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
        print(f"Seed {seed:4d}: Valid={results['valid_nrmses'][i]:.6f}, Test={results['test_nrmses'][i]:.6f}")
    
    # Additional analysis
    print(f"\nBest performing seed: {results['seeds_used'][np.argmin(results['test_nrmses'])]}")  # Lower NRMSE is better
    print(f"Worst performing seed: {results['seeds_used'][np.argmax(results['test_nrmses'])]}")
    print(f"Most stable (lowest std): Validation={results['valid_std']:.6f}, Test={results['test_std']:.6f}")

# %%
# Optional: Save results to file
if 'results' in locals() and results is not None:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"{parent_dir}/results/lorenz_stats_results_{timestamp}.txt"
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    with open(results_filename, 'w') as f:
        f.write(f"Lorenz Evaluation Statistics - {STUDY_NAME}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Network units: {results['config']['n_units']}\n")
        f.write(f"  - Lorenz dimension: {results['config']['lorenz_dim']}\n")
        f.write(f"  - Prediction lag: {results['config']['lag']}\n")
        f.write(f"  - Aligned orientations: {results['config']['aligned_orientations']}\n")
        f.write(f"  - Angular input: {results['config']['ang_input']}\n")
        f.write(f"  - Angular connections: {results['config']['ang_connections']}\n\n")
        
        f.write(f"Results Summary:\n")
        f.write(f"  - Validation NRMSE: {results['valid_mean']:.6f} ± {results['valid_std']:.6f}\n")
        f.write(f"  - Test NRMSE: {results['test_mean']:.6f} ± {results['test_std']:.6f}\n")
        f.write(f"  - Correlation: {results['correlation']:.6f}\n")
        f.write(f"  - Successful runs: {len(results['valid_nrmses'])}/{len(RANDOM_SEEDS)}\n\n")
        
        f.write("Individual Results:\n")
        for i, seed in enumerate(results['seeds_used']):
            f.write(f"  Seed {seed:4d}: Valid={results['valid_nrmses'][i]:.6f}, Test={results['test_nrmses'][i]:.6f}\n")
    
    print(f"\nResults saved to: {results_filename}")
