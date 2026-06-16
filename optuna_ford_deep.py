from functools import partial
import torch
from torch import nn, optim
from tqdm import tqdm
import optuna
from deep_unicycle_network import DeepUnicycleReservoir
from utils import get_FordA_data
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import traceback

# Configuration for deep network
N_LAYERS = 2  # Number of layers to explore
N_UNITS_PER_LAYER = [50, 50]  # Units for each layer

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
    
    # LogisticRegression hyperparameter
    c_param = trial.suggest_float("c_param", 1e-4, 1e2, log=True)  # Inverse of regularization strength
    
    # Input map parameters (for first layer)
    n_inp = 1  # FordA has 1-D time series
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
    
    n_steps_readout = 0
    # Warmup (washup) steps before collecting activations
    washup = trial.suggest_int("washup", 0, 4000, step=500)

    # Initialize the deep model
    model = DeepUnicycleReservoir(
        n_inp=n_inp, n_units_per_layer=N_UNITS_PER_LAYER, dt=dt, n_out=2,
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

    # Load FordA datasets
    train_loader, valid_loader, test_loader = get_FordA_data(bs, bs)

    # Set random initial states for batch processing
    model.set_init_states_random(bs)
    
    # Perform washup warmup using zero inputs and update initial states accordingly
    
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

    if washup > 0:
        # Extract initial states for batch size 1
        x_per_layer = [model.x_init_per_layer[i][0:1, :] for i in range(model.n_layers)]
        z_per_layer = [model.z_init_per_layer[i][0:1, :] for i in range(model.n_layers)]
        theta_per_layer = [model.theta_init_per_layer[i][0:1, :] for i in range(model.n_layers)]
        s_per_layer = [model.s_init_per_layer[i][0:1, :] for i in range(model.n_layers)]
        omega_per_layer = [model.omega_init_per_layer[i][0:1, :] for i in range(model.n_layers)]
        
        # Manually iterate through washup steps
        u_lin = torch.zeros((1, washup, n_inp), device=device)
        u_ang = torch.zeros_like(u_lin, device=device)
        
        for t in range(washup):
            linear_input = (u_lin[:, t] + inp_bias) @ model.lin_input_map
            angular_input = u_ang[:, t] @ model.ang_input_map
            
            # Process through each layer in sequence
            for layer_idx in range(model.n_layers):
                if layer_idx == 0:
                    lin_in = linear_input
                    ang_in = angular_input
                else:
                    # Extract positions from previous layer and apply transform
                    layer_output = torch.cat([x_per_layer[layer_idx - 1], z_per_layer[layer_idx - 1]], dim=-1)
                    if model.use_linear_transform:
                        lin_in = layer_output @ model.position_transforms[layer_idx - 1]
                    else:
                        lin_in = layer_output
                    
                    if model.use_angular_transform:
                        ang_in = theta_per_layer[layer_idx - 1] @ model.angular_transforms[layer_idx - 1]
                    else:
                        ang_in = lin_in
                
                # Update states for this layer
                x_per_layer[layer_idx], z_per_layer[layer_idx], theta_per_layer[layer_idx], \
                    s_per_layer[layer_idx], omega_per_layer[layer_idx] = model.unicycle_layers[layer_idx](
                    lin_in, ang_in,
                    x_per_layer[layer_idx], z_per_layer[layer_idx], theta_per_layer[layer_idx],
                    s_per_layer[layer_idx], omega_per_layer[layer_idx]
                )
        
        # Set the final washup states as initial states for full batch
        model.set_init_states(bs, x_per_layer, z_per_layer, theta_per_layer, s_per_layer, omega_per_layer)
    

    # Training: Extract reservoir activations
    try:
        activations, ys = [], []
        
        for x, labels in train_loader:
            x = x.to(device)
            labels = labels.to(device)
            
            # Get reservoir activations (no gradients needed)
            with torch.no_grad():
                states_list, _, _ = model(x, x)
            
            # Process activations - concatenate all states from states_list
            batch_activations = torch.stack(states_list, dim=1)  # (batch, time, n_units*5)
            
            # Check for stability
            batch_activations_np = batch_activations.cpu().numpy()
            is_stable, stability_reason = check_dynamics_stability(batch_activations_np)
            if not is_stable:
                print(f"Dynamics instability detected: {stability_reason}")
                return 0.0  # Return low accuracy for unstable dynamics
            
            # Take only the final time step as features for classification
            final_activations = batch_activations[:, -1, :]  # (batch, features)
            
            activations.append(final_activations.detach().cpu())
            ys.append(labels.cpu())
        
        activations = torch.cat(activations, dim=0).numpy()
        ys = torch.cat(ys, dim=0).numpy()
        
        print(f"Activations shape: {activations.shape}, Labels shape: {ys.shape}")
        
        # Check for NaN/Inf in activations
        if np.isnan(activations).any() or np.isinf(activations).any():
            print("NaN or Inf values detected in activations, ending trial")
            return 0.0
        
        # Train LogisticRegression classifier
        scaler = preprocessing.StandardScaler().fit(activations)
        activations_scaled = scaler.transform(activations)
        classifier = LogisticRegression(C=c_param, max_iter=1000, solver='lbfgs').fit(activations_scaled, ys)
        
        # Validation
        def test_esn(data_loader, classifier, scaler):
            activations, ys = [], []
            for x, labels in data_loader:
                x = x.to(device)
                labels = labels.to(device)
                
                with torch.no_grad():
                    states_list, _, _ = model(x, x)
                
                batch_activations = torch.stack(states_list, dim=1)  # (batch, time, features)
                
                # Check validation set stability
                batch_activations_np = batch_activations.cpu().numpy()
                is_stable, stability_reason = check_dynamics_stability(batch_activations_np)
                if not is_stable:
                    print(f"Validation set instability detected: {stability_reason}")
                    return 0.0
                
                # Take only the final time step
                final_activations = batch_activations[:, -1, :]
                
                activations.append(final_activations.cpu())
                ys.append(labels.cpu())
            
            activations = torch.cat(activations, dim=0).numpy()
            activations = scaler.transform(activations)
            ys = torch.cat(ys, dim=0).numpy()
            
            return classifier.score(activations, ys)
        
        valid_accuracy = test_esn(valid_loader, classifier, scaler)
        
        print(f"Validation Accuracy: {valid_accuracy:.6f}")
        
        # Return accuracy (higher is better, so Optuna will maximize)
        return valid_accuracy
        
    except Exception as e:
        import traceback
        print(f"Error in trial: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return 0.0  # Return low accuracy for failed trials


# Define the search space and start the optimization
if __name__ == '__main__':
    # Define the GPU or CPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Deep network configuration: N_LAYERS={N_LAYERS}, N_UNITS_PER_LAYER={N_UNITS_PER_LAYER}")

    database_name = "unicycle_nets_ford_deep"
    study_name = f"ford_prediction_deep_{N_LAYERS}layers_{N_UNITS_PER_LAYER}_ang_connected_layers_concatenated_instability_check_not_aligned_washup"
    storage_name = "sqlite:///optuna_databases/{}.db".format(database_name)
    study = optuna.create_study(storage=storage_name, study_name=study_name, direction='maximize', load_if_exists=True)
    
    # Run optimization
    study.optimize(
        partial(objective, aligned_orientations=False, ang_input=True, ang_connections=True), 
        timeout=3600*8
    )

    # Get the best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    print(f"Best Validation Accuracy: {best_value:.6f}")
    print(f"Best hyperparameters: {best_params}")

    # Save results
    with open('ford_unicycle_deep_results.txt', 'w') as f:
        f.write(f"Deep Network Configuration:\n")
        f.write(f"  N_LAYERS: {N_LAYERS}\n")
        f.write(f"  N_UNITS_PER_LAYER: {N_UNITS_PER_LAYER}\n\n")
        f.write(f"Best Validation Accuracy: {best_value:.6f}\n")
        f.write(f"Best hyperparameters:\n")
        for key, value in sorted(best_params.items()):
            f.write(f"  {key}: {value}\n")
