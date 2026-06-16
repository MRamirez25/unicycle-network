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

from utils import get_lorenz, n_params, count_classifier_params
from deep_unicycle_network import DeepUnicycleReservoir

#%%
# Configuration
RANDOM_SEEDS = [33, 42, 123, 456, 789]  # Multiple seeds for statistics
STUDY_NAME = "lorenz_prediction_deep_lag25_2layers_ang_connected_layers"
DATABASE_NAME = "unicycle_nets_lorenz_deep"

# Model configuration flags (will be loaded from params)
ALIGNED_ORIENTATIONS = None  # Will be set from params
ANG_INPUT = None  # Will be set from params
ANG_CONNECTIONS = None  # Will be set from params

# Deep network specific configuration
N_LAYERS = 2  # Number of stacked layers
N_UNITS_PER_LAYER = [50,50]  # Units for each layer, e.g., [20, 15] for 2-layer network
linear_transform_scale = 10.0  # Scale for position transforms (x,z -> next layer linear input)
angular_transform_scale = 1.0  # Scale for angular transforms (theta -> next layer angular input)

# Lorenz specific configurations
N_LORENZ = 5  # Lorenz system dimension
LAG = 25
WASHOUT = 200

# Feature selection for readout (slicing batch_activations in feature dimension)
# batch_activations has shape (batch, time, n_units*5) where states are concatenated as:
#   [x (n_units), z (n_units), theta (n_units), s (n_units), omega (n_units)]
#
# Option 1: Feature-based slicing (select by feature indices)
#   FEATURE_SLICE_START/STOP: Select features by index in flattened space
#   Example: (200, 300) selects features 200-300
#   Set to (None, None) for all features
FEATURE_SLICE_START = None
FEATURE_SLICE_STOP = None
#
# Option 2: Unit-based slicing (select by unit range, all states per unit)
#   UNIT_SLICE_START/STOP: Select units start:stop with all their states
#   Example: (0, 30) selects all 5 states (x,z,theta,s,omega) from units 0-29 of final layer
#   Set to (None, None) for all units
#   Note: Unit-based slicing takes precedence over feature-based slicing if both are specified
UNIT_SLICE_START = None
UNIT_SLICE_STOP = None

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
def get_feature_slice_from_units(n_units_total, unit_start=None, unit_stop=None):
    """
    Convert unit-based slicing to feature indices.
    
    States are organized as: [x, z, theta, s, omega] each with n_units elements
    This function extracts all 5 states for units [unit_start:unit_stop]
    
    Args:
        n_units_total: Total number of units in the network (final layer)
        unit_start: Starting unit index (default: 0)
        unit_stop: Stopping unit index, exclusive (default: n_units_total)
    
    Returns:
        Feature slice indices as a list, or None if invalid
    
    Example:
        n_units=60, unit_start=0, unit_stop=30
        Returns indices for x[0:30], z[0:30], theta[0:30], s[0:30], omega[0:30]
    """
    if unit_start is None and unit_stop is None:
        return None  # Use all features
    
    if unit_start is None:
        unit_start = 0
    if unit_stop is None:
        unit_stop = n_units_total
    
    if not (0 <= unit_start <= n_units_total and 0 <= unit_stop <= n_units_total and unit_start <= unit_stop):
        raise ValueError(f"Invalid unit slice: start={unit_start}, stop={unit_stop}, n_units={n_units_total}")
    
    # Collect indices for all 5 states in the specified unit range
    n_states = 5  # x, z, theta, s, omega
    indices = []
    for state_idx in range(n_states):
        state_offset = state_idx * n_units_total
        indices.extend(range(state_offset + unit_start, state_offset + unit_stop))
    
    return indices

#%%
def apply_feature_slice(batch_activations, feature_start=None, feature_stop=None, unit_slice_indices=None):
    """
    Apply feature slicing to batch activations.
    
    Args:
        batch_activations: Array of shape (batch, time, n_features)
        feature_start: Start index for feature-based slicing
        feature_stop: Stop index for feature-based slicing
        unit_slice_indices: List of indices for unit-based slicing (takes precedence)
    
    Returns:
        Sliced batch_activations
    """
    # Unit-based slicing takes precedence
    if unit_slice_indices is not None:
        return batch_activations[:, :, unit_slice_indices]
    
    # Feature-based slicing
    if feature_stop is not None:
        return batch_activations[:, :, feature_start:feature_stop]
    elif feature_start is not None and feature_start > 0:
        return batch_activations[:, :, feature_start:]
    
    # No slicing
    return batch_activations

#%%
def load_best_params():
    """Load best parameters from Optuna study"""
    storage_name = f"sqlite:///{parent_dir}/optuna_databases/{DATABASE_NAME}.db"
    study = optuna.create_study(storage=storage_name, study_name=STUDY_NAME, 
                               direction='minimize', load_if_exists="True")
    print(f"Best parameters from study '{STUDY_NAME}': {study.best_params}")
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
    """Initialize and configure the deep model"""
    
    # Expand N_UNITS_PER_LAYER if needed
    if isinstance(N_UNITS_PER_LAYER, int):
        n_units_per_layer = [N_UNITS_PER_LAYER] * N_LAYERS
    else:
        n_units_per_layer = N_UNITS_PER_LAYER
    
    n_inp = N_LORENZ
    
    # Helper function to extract per-layer parameters from params dict
    def get_per_layer_params(params, base_name, n_layers):
        """Extract per-layer parameters from params dict with layer-specific keys"""
        values = []
        for layer_idx in range(n_layers):
            key = f"{base_name}_layer{layer_idx}"
            if key in params:
                values.append(params[key])
            else:
                # Fallback: try without layer suffix for backward compatibility
                if base_name in params:
                    values.append(params[base_name])
                else:
                    raise KeyError(f"Parameter '{key}' or '{base_name}' not found in params")
        return values
    
    # Extract per-layer connectivity parameters
    n_connections_fraction_list = get_per_layer_params(params, 'n_connections_fraction', N_LAYERS)
    anchor_con_fraction_list = get_per_layer_params(params, 'anchor_con_fraction', N_LAYERS)
    
    n_connections_list = [int(n_units_per_layer[i] * n_connections_fraction_list[i]) for i in range(N_LAYERS)]
    n_connections_anchor_list = [int(anchor_con_fraction_list[i] * n_units_per_layer[i]) for i in range(N_LAYERS)]
    
    # Angular connections
    if params.get('ang_connections', False):
        n_connections_ang_fraction_list = get_per_layer_params(params, 'n_connections_ang_fraction', N_LAYERS)
        anchor_con_fraction_ang_list = get_per_layer_params(params, 'anchor_con_fraction_ang', N_LAYERS)
        n_connections_ang_list = [int(n_units_per_layer[i] * n_connections_ang_fraction_list[i]) for i in range(N_LAYERS)]
        n_connections_anchor_ang_list = [int(anchor_con_fraction_ang_list[i] * n_units_per_layer[i]) for i in range(N_LAYERS)]
    else:
        n_connections_ang_list = [0] * N_LAYERS
        n_connections_anchor_ang_list = [0] * N_LAYERS
    
    # Extract per-layer stiffness parameters
    lin_stiff_min_list = get_per_layer_params(params, 'lin_stiff_min', N_LAYERS)
    lin_stiff_max_list = get_per_layer_params(params, 'lin_stiff_max', N_LAYERS)
    ang_stiff_min_list = get_per_layer_params(params, 'ang_stiff_min', N_LAYERS)
    ang_stiff_max_list = get_per_layer_params(params, 'ang_stiff_max', N_LAYERS)
    
    # Extract per-layer damping parameters
    lin_damping_min_list = get_per_layer_params(params, 'lin_damping_min', N_LAYERS)
    lin_damping_max_list = get_per_layer_params(params, 'lin_damping_max', N_LAYERS)
    ang_damping_min_list = get_per_layer_params(params, 'ang_damping_min', N_LAYERS)
    ang_damping_max_list = get_per_layer_params(params, 'ang_damping_max', N_LAYERS)
    
    # Extract per-layer equilibrium distance parameters
    eq_dist_min_list = get_per_layer_params(params, 'eq_dist_min', N_LAYERS)
    eq_dist_max_list = get_per_layer_params(params, 'eq_dist_max', N_LAYERS)
    eq_dist_min_ang_list = get_per_layer_params(params, 'eq_dist_min_ang', N_LAYERS)
    eq_dist_max_ang_list = get_per_layer_params(params, 'eq_dist_max_ang', N_LAYERS)
    
    # Extract transform scales from params
    lin_transform_scale = params.get('linear_transform_scale', linear_transform_scale)
    ang_transform_scale = params.get('angular_transform_scale', angular_transform_scale)
    
    # Create deep model
    model = DeepUnicycleReservoir(
        n_inp=n_inp, 
        n_units_per_layer=n_units_per_layer,
        dt=params['dt'], 
        n_out=n_inp,  # n_out = n_inp for time series prediction
        n_layers=N_LAYERS,
        lin_input_map=lin_input_map,
        lin_stiff_min=lin_stiff_min_list, lin_stiff_max=lin_stiff_max_list,
        ang_stiff_min=ang_stiff_min_list, ang_stiff_max=ang_stiff_max_list,
        lin_damping_min=lin_damping_min_list, lin_damping_max=lin_damping_max_list,
        ang_damping_min=ang_damping_min_list, ang_damping_max=ang_damping_max_list,
        eq_dist_min=eq_dist_min_list, eq_dist_max=eq_dist_max_list,
        eq_dist_min_ang=eq_dist_min_ang_list, eq_dist_max_ang=eq_dist_max_ang_list,
        n_connections=n_connections_list, n_connections_anchor=n_connections_anchor_list,
        n_past_steps_readout=0, n_connections_ang=n_connections_ang_list,
        n_connections_anchor_ang=n_connections_anchor_ang_list,
        inp_bias=params['inp_bias'], 
        ang_input_map=ang_input_map,
        use_linear_transform=True,
        linear_transform_scale=lin_transform_scale,
        use_angular_transform=True,
        angular_transform_scale=ang_transform_scale,
        concatenate_all_layers=True#params.get('concatenate_all_layers', False)
    ).to(device)
    
    return model

#%%
def setup_initial_states(model, params, device, batch_size):
    """Setup initial states for the deep model"""
    model.set_init_states_random(batch_size)
    
    # Move all layer states to device
    for layer_idx in range(model.n_layers):
        model.x_init_per_layer[layer_idx] = model.x_init_per_layer[layer_idx].to(device)
        model.z_init_per_layer[layer_idx] = model.z_init_per_layer[layer_idx].to(device)
        model.theta_init_per_layer[layer_idx] = model.theta_init_per_layer[layer_idx].to(device)
        model.s_init_per_layer[layer_idx] = model.s_init_per_layer[layer_idx].to(device)
        model.omega_init_per_layer[layer_idx] = model.omega_init_per_layer[layer_idx].to(device)
    
    # Move input maps to device
    model.lin_input_map = model.lin_input_map.to(device)
    model.ang_input_map = model.ang_input_map.to(device)
    
    # Move network parameters to device
    for layer_idx in range(model.n_layers):
        unicycle_net = model.unicycle_layers[layer_idx]
        unicycle_net.lin_damping = unicycle_net.lin_damping.to(device)
        unicycle_net.ang_damping = unicycle_net.ang_damping.to(device)
        unicycle_net.mass_vector = unicycle_net.mass_vector.to(device)
        unicycle_net.j_vector = unicycle_net.j_vector.to(device)
    
    # Position transforms are already on device from model.to(device)
    
    # Set specific initial conditions for all layers
    for layer_idx in range(model.n_layers):
        model.s_init_per_layer[layer_idx][:, 0] = 0
        model.omega_init_per_layer[layer_idx][:, :] = 0
        
        if not params.get('aligned_orientations', False):
            model.theta_init_per_layer[layer_idx][:, :] = torch.rand(model.theta_init_per_layer[layer_idx].size()) * (4*torch.pi) - 2*torch.pi
        else:
            model.theta_init_per_layer[layer_idx][:, :] = torch.rand(1) * (4*torch.pi) - 2*torch.pi
    
    return model

#%%
def run_washup(model, params, device):
    """Run washup phase through all layers and return final states"""
    washout = params['washout']
    
    # Initialize states for all layers
    layer_states = []
    for layer_idx in range(model.n_layers):
        layer_state = {
            'x': model.x_init_per_layer[layer_idx][0:1, :],
            'z': model.z_init_per_layer[layer_idx][0:1, :],
            'theta': model.theta_init_per_layer[layer_idx][0:1, :],
            's': model.s_init_per_layer[layer_idx][0:1, :],
            'omega': model.omega_init_per_layer[layer_idx][0:1, :],
        }
        layer_states.append(layer_state)
    
    u_lin = torch.zeros((1, washout, N_LORENZ), device=device)
    u_ang = torch.zeros_like(u_lin, device=device)
    
    for t in range(u_lin.size()[1]):
        # Process through each layer
        linear_input = (u_lin[:, t]) @ model.lin_input_map
        angular_input = u_ang[:, t] @ model.ang_input_map
        
        # First layer
        state = layer_states[0]
        x, z, theta, s, omega = model.unicycle_layers[0](
            linear_input, angular_input,
            state['x'], state['z'], state['theta'], state['s'], state['omega']
        )
        layer_states[0]['x'] = x
        layer_states[0]['z'] = z
        layer_states[0]['theta'] = theta
        layer_states[0]['s'] = s
        layer_states[0]['omega'] = omega
        
        # Subsequent layers
        for layer_idx in range(1, model.n_layers):
            # During washup, use zero input to let system settle
            lin_input = torch.zeros((1, model.n_units_per_layer[layer_idx]), device=device)
            ang_input = torch.zeros_like(lin_input)
            
            state = layer_states[layer_idx]
            x, z, theta, s, omega = model.unicycle_layers[layer_idx](
                lin_input, ang_input,
                state['x'], state['z'], state['theta'], state['s'], state['omega']
            )
            layer_states[layer_idx]['x'] = x
            layer_states[layer_idx]['z'] = z
            layer_states[layer_idx]['theta'] = theta
            layer_states[layer_idx]['s'] = s
            layer_states[layer_idx]['omega'] = omega
    
    # Return final states from all layers
    return layer_states

#%%
@torch.no_grad()
def test_esn(dataset, model, classifier, scaler, device, params):
    """Test ESN performance for Lorenz prediction"""
    activations, targets = [], []
    washout = params['washout']
    n_units_final = N_UNITS_PER_LAYER[-1]  # Units in final layer
    
    # Determine feature slice indices
    unit_slice_indices = None
    if UNIT_SLICE_START is not None or UNIT_SLICE_STOP is not None:
        unit_slice_indices = get_feature_slice_from_units(n_units_final, UNIT_SLICE_START, UNIT_SLICE_STOP)
    
    for batch_idx, batch_data in enumerate(dataset):
        # Process data like in unicycle_optuna_lorenz.py
        target = batch_data[:, (LAG+washout):].numpy()  # Shape: (batch, time_steps, 5)
        input_data = batch_data[:, :(2000+washout)].to(device)  # Shape: (batch, time_steps, 5)
        
        states_list, _, _ = model(input_data, input_data)
        
        # Process activations - concatenate all states from states_list
        batch_activations = torch.stack(states_list, dim=1)  # (batch, time, n_units*5)
        batch_activations = batch_activations.cpu().numpy()
        
        # Apply feature slicing
        batch_activations = apply_feature_slice(
            batch_activations, 
            feature_start=FEATURE_SLICE_START, 
            feature_stop=FEATURE_SLICE_STOP,
            unit_slice_indices=unit_slice_indices
        )
        
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
    lin_input_map, ang_input_map = create_input_maps(params, N_UNITS_PER_LAYER[0] if isinstance(N_UNITS_PER_LAYER, list) else N_UNITS_PER_LAYER, N_LORENZ, seed)
    
    # Initialize model
    model = initialize_model(params, lin_input_map, ang_input_map, device)
    
    # Use batch size from dataset
    batch_size = train_dataset[0].shape[0]
    model = setup_initial_states(model, params, device, batch_size)
    
    # Run washup
    layer_states = run_washup(model, params, device)
    
    # Set initial states after washup for all layers
    x_per_layer = [layer_states[i]['x'] for i in range(model.n_layers)]
    z_per_layer = [layer_states[i]['z'] for i in range(model.n_layers)]
    theta_per_layer = [layer_states[i]['theta'] for i in range(model.n_layers)]
    s_per_layer = [layer_states[i]['s'] for i in range(model.n_layers)]
    omega_per_layer = [layer_states[i]['omega'] for i in range(model.n_layers)]
    
    model.set_init_states(batch_size, x_per_layer, z_per_layer, theta_per_layer, s_per_layer, omega_per_layer)
    
    # Determine feature slice indices
    n_units_final = N_UNITS_PER_LAYER[-1]  # Units in final layer
    unit_slice_indices = None
    if UNIT_SLICE_START is not None or UNIT_SLICE_STOP is not None:
        unit_slice_indices = get_feature_slice_from_units(n_units_final, UNIT_SLICE_START, UNIT_SLICE_STOP)
    
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
        
        # Apply feature slicing
        batch_activations = apply_feature_slice(
            batch_activations, 
            feature_start=FEATURE_SLICE_START, 
            feature_stop=FEATURE_SLICE_STOP,
            unit_slice_indices=unit_slice_indices
        )
        
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
        return None, None, None
    
    # Standardize and train classifier
    scaler = preprocessing.StandardScaler().fit(activations)
    activations_scaled = scaler.transform(activations)
    classifier = Ridge(alpha=params['ridge_alpha']).fit(activations_scaled, targets)
    n_classifier_params = count_classifier_params(classifier)
    
    # Evaluate
    valid_nrmse = test_esn(valid_dataset, model, classifier, scaler, device, params)
    test_nrmse = test_esn(test_dataset, model, classifier, scaler, device, params)
    
    print(f"Seed {seed}: Valid NRMSE={valid_nrmse:.6f}, Test NRMSE={test_nrmse:.6f}, Classifier params={n_classifier_params}")
    
    return valid_nrmse, test_nrmse, n_classifier_params

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
    print(f"Deep network: n_layers={N_LAYERS}, n_units_per_layer={N_UNITS_PER_LAYER}")
    print(f"Lorenz dim: {N_LORENZ}, Lag: {LAG}")
    
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
    classifier_params = None  # Will be same for all runs
    
    print(f"\nRunning {len(RANDOM_SEEDS)} evaluations...")
    for seed in RANDOM_SEEDS:
        valid_nrmse, test_nrmse, n_classifier_params = single_evaluation(params, seed, device, train_dataset, valid_dataset, test_dataset)
        
        if valid_nrmse is not None and test_nrmse is not None:
            valid_nrmses.append(valid_nrmse)
            test_nrmses.append(test_nrmse)
            if classifier_params is None:
                classifier_params = n_classifier_params
        else:
            print(f"Skipping seed {seed} due to NaN values")
    
    # Calculate statistics
    if len(valid_nrmses) > 0:
        valid_mean = np.mean(valid_nrmses)
        valid_std = np.std(valid_nrmses)
        test_mean = np.mean(test_nrmses)
        test_std = np.std(test_nrmses)
        
        print("\n" + "="*50)
        print("LORENZ EVALUATION STATISTICS (DEEP NETWORK)")
        print("="*50)
        print(f"Number of successful runs: {len(valid_nrmses)}/{len(RANDOM_SEEDS)}")
        print(f"Classifier trainable parameters: {classifier_params}")
        print(f"Validation NRMSE: {valid_mean:.6f} ± {valid_std:.6f}")
        print(f"Test NRMSE: {test_mean:.6f} ± {test_std:.6f}")
        print(f"Valid NRMSEs: {[f'{s:.6f}' for s in valid_nrmses]}")
        print(f"Test NRMSEs: {[f'{s:.6f}' for s in test_nrmses]}")
        
        # Calculate correlation between validation and test scores
        correlation = np.corrcoef(valid_nrmses, test_nrmses)[0, 1]
        
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
        units_str = str(N_UNITS_PER_LAYER) if isinstance(N_UNITS_PER_LAYER, list) else f"[{N_UNITS_PER_LAYER}]*{N_LAYERS}"
        summary_text = f"""
Lorenz Deep Statistics Summary:

Mean ± Std:
• Validation: {valid_mean:.6f} ± {valid_std:.6f}
• Test: {test_mean:.6f} ± {test_std:.6f}

Range:
• Validation: [{min(valid_nrmses):.6f}, {max(valid_nrmses):.6f}]
• Test: [{min(test_nrmses):.6f}, {max(test_nrmses):.6f}]

Correlation: {correlation:.3f}

Network Config:
• Layers: {N_LAYERS}
• Units/layer: {units_str}
• Lorenz dim: {N_LORENZ}
• Lag: {LAG}
• Classifier params: {classifier_params}
        """
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{parent_dir}/plots/lorenz_evaluation_stats_deep_{STUDY_NAME}.png"
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
            'classifier_params': classifier_params,
            'config': {
                'n_layers': N_LAYERS,
                'n_units_per_layer': N_UNITS_PER_LAYER if isinstance(N_UNITS_PER_LAYER, list) else [N_UNITS_PER_LAYER]*N_LAYERS,
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
        print(f"Seed {seed:4d}: Valid NRMSE={results['valid_nrmses'][i]:.6f}, Test NRMSE={results['test_nrmses'][i]:.6f}")
    
    # Additional analysis
    best_idx = np.argmin(results['test_nrmses'])
    worst_idx = np.argmax(results['test_nrmses'])
    print(f"\nBest performing seed: {results['seeds_used'][best_idx]} (Test NRMSE={results['test_nrmses'][best_idx]:.6f})")
    print(f"Worst performing seed: {results['seeds_used'][worst_idx]} (Test NRMSE={results['test_nrmses'][worst_idx]:.6f})")
    print(f"Most stable (lowest std): Validation={results['valid_std']:.6f}, Test={results['test_std']:.6f}")

# %%
# Optional: Save results to file
if 'results' in locals() and results is not None:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"{parent_dir}/results/lorenz_stats_results_deep_{timestamp}.txt"
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    with open(results_filename, 'w') as f:
        f.write(f"Lorenz Deep Evaluation Statistics - {STUDY_NAME}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Network layers: {results['config']['n_layers']}\n")
        f.write(f"  - Units per layer: {results['config']['n_units_per_layer']}\n")
        f.write(f"  - Lorenz dim: {results['config']['lorenz_dim']}\n")
        f.write(f"  - Lag: {results['config']['lag']}\n")
        f.write(f"  - Feature slice: [{FEATURE_SLICE_START}:{FEATURE_SLICE_STOP}]\n")
        f.write(f"  - Classifier trainable params: {results['classifier_params']}\n")
