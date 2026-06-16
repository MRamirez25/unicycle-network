from functools import partial
import torch
from torch import nn, optim
from tqdm import tqdm
import optuna
from deep_unicycle_network import DeepUnicycleReservoir
from utils import get_lorenz
import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn import preprocessing
import traceback

# Configuration for deep network
N_LAYERS = 5  # Number of layers to explore
N_UNITS_PER_LAYER = [20, 20, 20, 20, 20]  # Units for each layer

# Thresholds for detecting explosive dynamics
MAX_STD_THRESHOLD = 50.0  # Maximum allowed std for any state component
MAX_VALUE_THRESHOLD = 1000.0  # Maximum allowed absolute value
DIVERGENCE_RATIO = 5.0  # If later half std is > 5x earlier half std, likely diverging

def check_dynamics_stability(activations, states_list=None):
    """
    Check if the network dynamics are exploding.
    
    Args:
        activations: numpy array of shape (batch, time, features)
        states_list: optional list of per-layer activations for more detailed analysis
        
    Returns:
        (is_stable, reason): tuple where is_stable is True if dynamics are okay,
                            reason is a string explaining any issues
    """
    # Check for NaN/Inf
    if np.isnan(activations).any() or np.isinf(activations).any():
        return False, "NaN or Inf values detected"
    
    # Check for extremely large values
    if np.max(np.abs(activations)) > MAX_VALUE_THRESHOLD:
        return False, f"Values exceed threshold (max={np.max(np.abs(activations)):.2e})"
    
    # Check standard deviation per time window
    # Split into early and late periods to detect divergence
    time_steps = activations.shape[1]
    if time_steps > 100:
        early_half = activations[:, :time_steps//2, :]
        late_half = activations[:, time_steps//2:, :]
        
        early_std = np.std(early_half, axis=(0, 1))
        late_std = np.std(late_half, axis=(0, 1))
        
        # Check for divergence (sudden growth)
        divergence = late_std / (early_std + 1e-9)
        if np.any(divergence > DIVERGENCE_RATIO):
            return False, f"Dynamics diverging (divergence ratio={np.max(divergence):.2f})"
        
        # Check absolute std values
        if np.any(late_std > MAX_STD_THRESHOLD):
            return False, f"Late-period std too large (max={np.max(late_std):.2f})"
    
    # Overall statistics check
    overall_std = np.std(activations, axis=(0, 1))
    if np.any(overall_std > MAX_STD_THRESHOLD):
        return False, f"Overall std too large (max={np.max(overall_std):.2f})"
    
    return True, "Stable"

# Objective function for Optuna
def objective(trial, aligned_orientations=None, ang_input=None, ang_connections=None):
    # Suggest hyperparameters for Optuna to search
    aligned_orientations = trial.suggest_categorical("aligned_orientations", [True, False]) if aligned_orientations is None else aligned_orientations
    if not aligned_orientations:
        ang_input = trial.suggest_categorical("ang_input", [True, False]) if ang_input is None else ang_input
        ang_connections = trial.suggest_categorical("ang_connections", [True, False]) if ang_connections is None else ang_connections
    else:
        ang_input = False
        ang_connections = False

    # Global parameters (same across all layers)
    bs = 500
    dt = trial.suggest_float("dt", 0.0001, 0.01, log=True)
    inp_bias = trial.suggest_float("inp_bias", -1, 1)
    
    # Ridge regression hyperparameter
    ridge_alpha = trial.suggest_float("ridge_alpha", 1e-6, 1e2, log=True)
    
    # Lorenz dataset parameters
    lag = 25
    washout = trial.suggest_int("washout", 0, 4000, step=500)
    
    # Input map parameters (for first layer)
    n_inp = 5  # Lorenz system has 5 dimensions
    non_zero_fraction = trial.suggest_float("non_zero_fraction", 0.1, 1.0, step=0.1)
    total_elements = n_inp * N_UNITS_PER_LAYER[0]
    num_non_zero = int(total_elements * non_zero_fraction)
    
    # Create sparse input maps for first layer
    lin_input_map = torch.zeros(n_inp, N_UNITS_PER_LAYER[0])
    flat_indices = torch.randperm(total_elements)[:num_non_zero]
    row_indices = flat_indices // N_UNITS_PER_LAYER[0]
    col_indices = flat_indices % N_UNITS_PER_LAYER[0]
    
    magnitude_min = trial.suggest_float("magnitude_min", -10.0, 0.0)
    magnitude_max = trial.suggest_float("magnitude_max", magnitude_min, 10.0)
    random_values = torch.rand(num_non_zero) * (magnitude_max - magnitude_min) + magnitude_min
    lin_input_map[row_indices, col_indices] = random_values

    # Angular input map for first layer
    ang_input_map = torch.zeros(n_inp, N_UNITS_PER_LAYER[0])
    if ang_input:
        non_zero_fraction_ang = trial.suggest_float("non_zero_fraction_ang", 0.1, 1.0, step=0.1)
        num_non_zero_ang = int(total_elements * non_zero_fraction_ang)
        flat_indices_ang = torch.randperm(total_elements)[:num_non_zero_ang]
        row_indices_ang = flat_indices_ang // N_UNITS_PER_LAYER[0]
        col_indices_ang = flat_indices_ang % N_UNITS_PER_LAYER[0]
        
        magnitude_min_ang = trial.suggest_float("magnitude_min_ang", -10.0, 0.0)
        magnitude_max_ang = trial.suggest_float("magnitude_max_ang", magnitude_min_ang, 10.0)
        random_values_ang = torch.rand(num_non_zero_ang) * (magnitude_max_ang - magnitude_min_ang) + magnitude_min_ang
        ang_input_map[row_indices_ang, col_indices_ang] = random_values_ang

    # Per-layer network parameters
    lin_stiff_min_list = []
    lin_stiff_max_list = []
    ang_stiff_min_list = []
    ang_stiff_max_list = []
    lin_damping_min_list = []
    lin_damping_max_list = []
    ang_damping_min_list = []
    ang_damping_max_list = []
    eq_dist_min_list = []
    eq_dist_max_list = []
    eq_dist_min_ang_list = []
    eq_dist_max_ang_list = []
    n_connections_list = []
    n_connections_anchor_list = []
    n_connections_ang_list = []
    n_connections_anchor_ang_list = []
    
    for layer_idx in range(N_LAYERS):
        # Stiffness parameters
        lin_stiff_min = trial.suggest_float(f'lin_stiff_min_layer{layer_idx}', 0.1, 1.0)
        lin_stiff_max = trial.suggest_float(f'lin_stiff_max_layer{layer_idx}', lin_stiff_min, 10.0)
        lin_stiff_min_list.append(lin_stiff_min)
        lin_stiff_max_list.append(lin_stiff_max)
        
        ang_stiff_min = trial.suggest_float(f'ang_stiff_min_layer{layer_idx}', 0.1, 1.0)
        ang_stiff_max = trial.suggest_float(f'ang_stiff_max_layer{layer_idx}', ang_stiff_min, 2.0)
        ang_stiff_min_list.append(ang_stiff_min)
        ang_stiff_max_list.append(ang_stiff_max)
        
        # Damping parameters
        lin_damping_min = trial.suggest_float(f'lin_damping_min_layer{layer_idx}', 0.1, 1.0)
        lin_damping_max = trial.suggest_float(f'lin_damping_max_layer{layer_idx}', lin_damping_min, 5.0)
        lin_damping_min_list.append(lin_damping_min)
        lin_damping_max_list.append(lin_damping_max)
        
        ang_damping_min = trial.suggest_float(f'ang_damping_min_layer{layer_idx}', 0.01, 1.0)
        ang_damping_max = trial.suggest_float(f'ang_damping_max_layer{layer_idx}', ang_damping_min, 10.0)
        ang_damping_min_list.append(ang_damping_min)
        ang_damping_max_list.append(ang_damping_max)
        
        # Equilibrium distance parameters
        eq_dist_min = trial.suggest_float(f"eq_dist_min_layer{layer_idx}", 0.2, 1.0)
        eq_dist_max = trial.suggest_float(f"eq_dist_max_layer{layer_idx}", eq_dist_min, 2.0)
        eq_dist_min_list.append(eq_dist_min)
        eq_dist_max_list.append(eq_dist_max)
        
        eq_dist_min_ang = trial.suggest_float(f"eq_dist_min_ang_layer{layer_idx}", -2*np.pi, 0.0)
        eq_dist_max_ang = trial.suggest_float(f"eq_dist_max_ang_layer{layer_idx}", eq_dist_min_ang, 2*np.pi)
        eq_dist_min_ang_list.append(eq_dist_min_ang)
        eq_dist_max_ang_list.append(eq_dist_max_ang)
        
        # Connectivity parameters
        n_connections_fraction = trial.suggest_float(f"n_connections_fraction_layer{layer_idx}", 0.2, 1.0, step=0.1)
        n_connections = int(N_UNITS_PER_LAYER[layer_idx] * n_connections_fraction)
        n_connections_list.append(n_connections)
        
        anchor_con_fraction = trial.suggest_float(f"anchor_con_fraction_layer{layer_idx}", 0.1, 1.0, step=0.1)
        n_connections_anchor = int(anchor_con_fraction * N_UNITS_PER_LAYER[layer_idx])
        n_connections_anchor_list.append(n_connections_anchor)
        
        # Angular connectivity
        if not ang_connections:
            n_connections_ang_list.append(0)
            n_connections_anchor_ang_list.append(0)
        else:
            n_connections_ang_fraction = trial.suggest_float(f"n_connections_ang_fraction_layer{layer_idx}", 0., 1.0, step=0.1)
            n_connections_ang = int(N_UNITS_PER_LAYER[layer_idx] * n_connections_ang_fraction)
            n_connections_ang_list.append(n_connections_ang)
            
            anchor_con_fraction_ang = trial.suggest_float(f"anchor_con_fraction_ang_layer{layer_idx}", 0.1, 1.0, step=0.1)
            n_connections_anchor_ang = int(anchor_con_fraction_ang * N_UNITS_PER_LAYER[layer_idx])
            n_connections_anchor_ang_list.append(n_connections_anchor_ang)
    
    # Optional: linear transform scale for inter-layer connections
    linear_transform_scale = trial.suggest_float("linear_transform_scale", 0.1, 10.0, log=True)
    
    # Optional: angular transform scale for angular state coupling between layers
    angular_transform_scale = trial.suggest_float("angular_transform_scale", 0.1, 10.0, log=True)
    
    # Optional: concatenate activations from all layers
    concatenate_all_layers = trial.suggest_categorical("concatenate_all_layers", [True, False])
    concatenate_all_layers = True
    
    n_steps_readout = 0

    # Initialize the deep model
    model = DeepUnicycleReservoir(
        n_inp=n_inp, n_units_per_layer=N_UNITS_PER_LAYER, dt=dt, n_out=n_inp,
        lin_stiff_min=lin_stiff_min_list, lin_stiff_max=lin_stiff_max_list,
        ang_stiff_min=ang_stiff_min_list, ang_stiff_max=ang_stiff_max_list,
        lin_damping_min=lin_damping_min_list, lin_damping_max=lin_damping_max_list,
        ang_damping_min=ang_damping_min_list, ang_damping_max=ang_damping_max_list,
        eq_dist_min=eq_dist_min_list, eq_dist_max=eq_dist_max_list, 
        eq_dist_min_ang=eq_dist_min_ang_list, eq_dist_max_ang=eq_dist_max_ang_list,
        n_connections=n_connections_list, inp_bias=inp_bias, lin_input_map=lin_input_map, 
        n_connections_anchor=n_connections_anchor_list, ang_input_map=ang_input_map,
        n_connections_ang=n_connections_ang_list, n_connections_anchor_ang=n_connections_anchor_ang_list,
        n_past_steps_readout=n_steps_readout, use_linear_transform=True,
        linear_transform_scale=linear_transform_scale, use_angular_transform=True,
        angular_transform_scale=angular_transform_scale,
        concatenate_all_layers=concatenate_all_layers).to(device)

    # Generate Lorenz datasets
    train_dataset = get_lorenz(N=5, F=8, lag=lag, washout=washout)  # Shape: (batch, time_steps, 5)
    valid_dataset = get_lorenz(N=5, F=8, lag=lag, washout=washout)  # Shape: (batch, time_steps, 5)
    test_dataset = get_lorenz(N=5, F=8, lag=lag, washout=washout)   # Shape: (batch, time_steps, 5)

    # Set random initial states for batch processing
    batch_size = train_dataset.shape[0]
    model.set_init_states_random(batch_size)
    
    # Move states to device
    for layer_idx in range(model.n_layers):
        model.x_init_per_layer[layer_idx] = model.x_init_per_layer[layer_idx].to(device)
        model.z_init_per_layer[layer_idx] = model.z_init_per_layer[layer_idx].to(device)
        model.theta_init_per_layer[layer_idx] = model.theta_init_per_layer[layer_idx].to(device)
        model.s_init_per_layer[layer_idx] = model.s_init_per_layer[layer_idx].to(device)
        model.omega_init_per_layer[layer_idx] = model.omega_init_per_layer[layer_idx].to(device)
    
    model.lin_input_map = model.lin_input_map.to(device)
    model.ang_input_map = model.ang_input_map.to(device)
    
    # Move network parameters to device for all layers
    for layer_idx in range(model.n_layers):
        unicycle_net = model.unicycle_layers[layer_idx]
        unicycle_net.lin_damping = unicycle_net.lin_damping.to(device)
        unicycle_net.ang_damping = unicycle_net.ang_damping.to(device)
        unicycle_net.mass_vector = unicycle_net.mass_vector.to(device)
        unicycle_net.j_vector = unicycle_net.j_vector.to(device)

    # Configure initial states for all layers
    for layer_idx in range(model.n_layers):
        model.s_init_per_layer[layer_idx][:, 0] = 0
        model.omega_init_per_layer[layer_idx][:, :] = 0
        if not aligned_orientations:
            model.theta_init_per_layer[layer_idx][:, :] = torch.rand(model.theta_init_per_layer[layer_idx].size()) * (4*torch.pi) - 2*torch.pi
            model.theta_init_per_layer[layer_idx][:, 0] = 0
        else:
            model.theta_init_per_layer[layer_idx][:, :] = torch.rand(1) * (4*torch.pi) - 2*torch.pi

    # Training: Extract reservoir activations
    try:
        # For 2000 training pairs with lag=25 lookahead:
        # activation[t] at time (washout+t) predicts target at time (washout+t+lag)
        target = train_dataset[:, (lag+washout):(2000+lag+washout)].numpy()  # Shape: (batch, 2000, 5)
        dataset = train_dataset[:, :(2000+washout)].to(device)  # Shape: (batch, 2000+washout, 5)
        
        print(f"train_dataset shape: {train_dataset.shape}")
        print(f"dataset shape: {dataset.shape}")
        print(f"target shape: {target.shape}")
        
        # Get reservoir activations (no gradients needed)
        with torch.no_grad():
            states_list, _, _ = model(dataset, dataset)
        
        # Process activations - concatenate all states from states_list
        activations = torch.stack(states_list, dim=1)  # (batch, time, n_units*5)
        activations = activations.cpu().numpy()
        
        print(f"Raw activations shape: {activations.shape}")
        
        # Perform per-layer stability analysis BEFORE removing washout
        # (while we still have the proper 3D structure)
        if len(states_list) > 1:
            print("Per-layer stability analysis:")
            n_units = N_UNITS_PER_LAYER[0]
            for layer_idx in range(N_LAYERS):
                # Get activations for this layer, AFTER washout is removed
                layer_activations = activations[:, washout:, layer_idx*n_units*5:(layer_idx+1)*n_units*5]
                # Safely compute std, handling edge cases
                if layer_activations.size > 1:
                    layer_std = np.std(layer_activations)
                    print(f"  Layer {layer_idx}: std={layer_std:.4f}", end="")
                    if layer_std > MAX_STD_THRESHOLD:
                        print(f" [TOO HIGH - exceeds {MAX_STD_THRESHOLD}]")
                    else:
                        print(f" [OK]")
                else:
                    print(f"  Layer {layer_idx}: insufficient data")
        
        # Remove washout period from activations to match target
        activations = activations[:, washout:, :]  # Remove first washout time steps
        
        # Reshape for Ridge regression: (batch * time_steps, features)
        activations = activations.reshape(-1, activations.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        
        print(f"Activations shape after washout removal: {activations.shape}")
        print(f"Target shape: {target.shape}")
        
        # Validate shape alignment
        if activations.shape[0] != target.shape[0]:
            print(f"ERROR: Shape mismatch! Activations have {activations.shape[0]} samples, target has {target.shape[0]}")
            return float('inf')
        
        # Check for stability in the raw (unbatched, unwashout-removed) activations
        # This gives us per-layer information
        activations_for_stability = activations.reshape(len(train_dataset), -1, activations.shape[-1])
        
        # Check overall stability
        is_stable, stability_reason = check_dynamics_stability(activations_for_stability)
        if not is_stable:
            print(f"Dynamics instability detected: {stability_reason}")
            return float('inf')  # Return high error for unstable dynamics
        
        # Train Ridge regression on reservoir activations
        scaler = preprocessing.StandardScaler().fit(activations)
        activations_scaled = scaler.transform(activations)
        classifier = Ridge(alpha=ridge_alpha, max_iter=1000).fit(activations_scaled, target)
        
        # Validation
        def test_esn(dataset, classifier, scaler):
            # For 2000 validation pairs with lag=25 lookahead:
            # activation[t] at time (washout+t) predicts target at time (washout+t+lag)
            target = dataset[:, (lag+washout):(2000+lag+washout)].numpy()  # Shape: (batch, 2000, 5)
            dataset = dataset[:, :(2000+washout)].to(device)  # Shape: (batch, 2000+washout, 5)

            with torch.no_grad():
                states_list, _, _ = model(dataset, dataset)
            
            # Process activations - concatenate all states from states_list
            activations = torch.stack(states_list, dim=1)  # (batch, time, n_units*5)
            activations = activations.cpu().numpy()
            
            # Check validation set stability
            activations_for_check = activations[:, washout:, :]
            is_stable, stability_reason = check_dynamics_stability(activations_for_check)
            if not is_stable:
                print(f"Validation set instability detected: {stability_reason}")
                return float('inf')
            
            # Remove washout period from activations to match target
            activations = activations[:, washout:, :]  # Remove first washout time steps
            
            # Reshape for prediction: (batch * time_steps, features)
            activations = activations.reshape(-1, activations.shape[-1])
            target = target.reshape(-1, target.shape[-1])
            
            activations = scaler.transform(activations)
            predictions = classifier.predict(activations)
            
            mse = np.mean(np.square(predictions - target))
            rmse = np.sqrt(mse)
            norm = np.sqrt(np.square(target).mean())
            nrmse = rmse / (norm + 1e-9)
            return nrmse
        
        valid_nrmse = test_esn(valid_dataset, classifier, scaler)
        
        print(f"Validation NRMSE: {valid_nrmse:.6f}")
        
        # Return NRMSE (lower is better, so Optuna will minimize)
        return valid_nrmse
        
    except Exception as e:
        import traceback
        print(f"Error in trial: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return float('inf')  # Return high error for failed trials


# Define the search space and start the optimization
if __name__ == '__main__':
    # Define the GPU or CPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Deep network configuration: N_LAYERS={N_LAYERS}, N_UNITS_PER_LAYER={N_UNITS_PER_LAYER}")

    database_name = "unicycle_nets_lorenz_deep"
    study_name = f"lorenz_prediction_deep_lag25_{N_LAYERS}layers_ang_connected_layers_concatenated_instability_check"
    storage_name = "sqlite:///optuna_databases/{}.db".format(database_name)
    study = optuna.create_study(storage=storage_name, study_name=study_name, direction='minimize', load_if_exists=True)
    
    # Run optimization
    study.optimize(
        partial(objective, aligned_orientations=False, ang_input=True, ang_connections=True), 
        timeout=3600*8
    )

    # Get the best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    print(f"Best NRMSE: {best_value:.6f}")
    print(f"Best hyperparameters: {best_params}")

    # Save results
    with open('lorenz_unicycle_deep_results.txt', 'w') as f:
        f.write(f"Deep Network Configuration:\n")
        f.write(f"  N_LAYERS: {N_LAYERS}\n")
        f.write(f"  N_UNITS_PER_LAYER: {N_UNITS_PER_LAYER}\n\n")
        f.write(f"Best NRMSE: {best_value:.6f}\n")
        f.write(f"Best hyperparameters:\n")
        for key, value in sorted(best_params.items()):
            f.write(f"  {key}: {value}\n")
