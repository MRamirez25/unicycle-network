from functools import partial
import torch
from torch import nn, optim
from tqdm import tqdm
import optuna
from unicycle_network_class import UnicycleReservoir
from utils import get_lorenz
import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn import preprocessing
import traceback

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

    n_units = 200
    lin_stiff_min = trial.suggest_float('lin_stiff_min', 0.1, 1.0)
    lin_stiff_max = trial.suggest_float('lin_stiff_max', lin_stiff_min, 10.0)  # lin_stiff_max >= lin_stiff_min
    ang_stiff_min = trial.suggest_float('ang_stiff_min', 0.1, 1.0)
    ang_stiff_max = trial.suggest_float('ang_stiff_max', ang_stiff_min, 2.0)  # ang_stiff_max >= ang_stiff_min
    lin_damping_min = trial.suggest_float('lin_damping_min', 0.1, 1.0)
    lin_damping_max = trial.suggest_float('lin_damping_max', lin_damping_min, 5.0)
    ang_damping_min = trial.suggest_float('ang_damping_min', 0.01, 1.0)
    ang_damping_max = trial.suggest_float('ang_damping_max', ang_damping_min, 10.0)
    bs = 500
    dt = trial.suggest_float("dt", 0.0001, 0.01, log=True)
    inp_bias = trial.suggest_float("inp_bias", -1,1)
    n_connections_anchor_fraction = trial.suggest_float("anchor_con_fraction", 0, 1.0, step=0.1)
    
    # Ridge regression hyperparameter
    ridge_alpha = trial.suggest_float("ridge_alpha", 1e-6, 1e2, log=True)
    
    # Lorenz dataset parameters
    lag = 25
    washout = trial.suggest_int("washout", 0, 4000, step=500)
    
    # Input map parameters
    n_inp = 5  # Lorenz system has 5 dimensions
    non_zero_fraction = trial.suggest_float("non_zero_fraction", 0.1, 1.0, step=0.1)
    total_elements = n_inp * n_units
    num_non_zero = int(total_elements * non_zero_fraction)
    
    # Create sparse input maps
    lin_input_map = torch.zeros(n_inp, n_units)
    flat_indices = torch.randperm(total_elements)[:num_non_zero]
    row_indices = flat_indices // n_units
    col_indices = flat_indices % n_units
    
    magnitude_min = trial.suggest_float("magnitude_min", -10.0, 0.0)
    magnitude_max = trial.suggest_float("magnitude_max", magnitude_min, 10.0)
    random_values = torch.rand(num_non_zero) * (magnitude_max - magnitude_min) + magnitude_min
    lin_input_map[row_indices, col_indices] = random_values

    # Angular input map
    ang_input_map = torch.zeros(n_inp, n_units)
    if ang_input:
        non_zero_fraction_ang = trial.suggest_float("non_zero_fraction_ang", 0.1, 1.0, step=0.1)
        num_non_zero_ang = int(total_elements * non_zero_fraction_ang)
        flat_indices_ang = torch.randperm(total_elements)[:num_non_zero_ang]
        row_indices_ang = flat_indices_ang // n_units
        col_indices_ang = flat_indices_ang % n_units
        
        magnitude_min_ang = trial.suggest_float("magnitude_min_ang", -10.0, 0.0)
        magnitude_max_ang = trial.suggest_float("magnitude_max_ang", magnitude_min_ang, 10.0)
        random_values_ang = torch.rand(num_non_zero_ang) * (magnitude_max_ang - magnitude_min_ang) + magnitude_min_ang
        ang_input_map[row_indices_ang, col_indices_ang] = random_values_ang

    # Network connectivity parameters
    n_connections_fraction = trial.suggest_float("n_connections_fraction", 0.2, 1.0, step=0.1)
    n_connections = int(n_units * n_connections_fraction)
    n_connections_anchor_fraction = trial.suggest_float("anchor_con_fraction", 0, 1.0, step=0.1)
    n_connections_anchor = int(n_connections_anchor_fraction * n_units)
    
    n_steps_readout = trial.suggest_int("steps_readout", 0, 20, step=5)
    
    if not ang_connections:
        n_connections_ang = 0
        n_connections_anchor_ang = 0
    else:
        n_connections_ang_fraction = trial.suggest_float("n_connections_ang_fraction", 0., 1.0, step=0.1)
        n_connections_ang = int(n_units * n_connections_ang_fraction)
        n_connections_anchor_fraction_ang = trial.suggest_float("anchor_con_fraction_ang", 0, 1.0, step=0.1)
        n_connections_anchor_ang = int(n_connections_anchor_fraction_ang * n_units)

    eq_dist_min = trial.suggest_float("eq_dist_min", 0.2, 1.0)
    eq_dist_max = trial.suggest_float("eq_dist_max", eq_dist_min, 2.0)
    eq_dist_min_ang = trial.suggest_float("eq_dist_min_ang", -2*torch.pi, 0.0)
    eq_dist_max_ang = trial.suggest_float("eq_dist_max_ang", eq_dist_min_ang, 2*torch.pi)

    # Initialize the model
    model = UnicycleReservoir(
        n_inp=n_inp, n_units=n_units, dt=dt, n_out=n_inp,  # n_out = n_inp for time series prediction
        lin_stiff_min=lin_stiff_min, lin_stiff_max=lin_stiff_max,
        ang_stiff_min=ang_stiff_min, ang_stiff_max=ang_stiff_max,
        lin_damping_min=lin_damping_min, lin_damping_max=lin_damping_max,
        ang_damping_min=ang_damping_min, ang_damping_max=ang_damping_max,
        eq_dist_min=eq_dist_min, eq_dist_max=eq_dist_max, 
        eq_dist_min_ang=eq_dist_min_ang, eq_dist_max_ang=eq_dist_max_ang,
        n_connections=n_connections, inp_bias=inp_bias, lin_input_map=lin_input_map, 
        n_connections_anchor=n_connections_anchor, ang_input_map=ang_input_map,
        n_connections_ang=n_connections_ang, n_connections_anchor_ang=n_connections_anchor_ang, 
        n_past_steps_readout=n_steps_readout).to(device)

    # Generate Lorenz datasets - use multiple samples for better training
    train_dataset = get_lorenz(N=5, F=8, lag=lag, washout=washout)  # Shape: (batch, time_steps, 5)
    valid_dataset = get_lorenz(N=5, F=8, lag=lag, washout=washout)  # Shape: (batch, time_steps, 5)
    test_dataset = get_lorenz(N=5, F=8, lag=lag, washout=washout)   # Shape: (batch, time_steps, 5)

    # Set random initial states for batch processing
    batch_size = train_dataset.shape[0]  # Use the batch size from dataset
    model.set_init_states_random(batch_size)
    model.x_init = model.x_init.to(device)
    model.z_init = model.z_init.to(device)
    model.theta_init = model.theta_init.to(device)
    model.s_init = model.s_init.to(device)
    model.omega_init = model.omega_init.to(device)
    model.lin_input_map = model.lin_input_map.to(device)
    model.ang_input_map = model.ang_input_map.to(device)
    
    # Move network parameters to device
    model.unicycle_network.lin_damping = model.unicycle_network.lin_damping.to(device)
    model.unicycle_network.ang_damping = model.unicycle_network.ang_damping.to(device)
    model.unicycle_network.mass_vector = model.unicycle_network.mass_vector.to(device)
    model.unicycle_network.j_vector = model.unicycle_network.j_vector.to(device)

    # Configure initial states
    model.s_init[:,0] = 0
    model.omega_init[:,:] = 0
    if not aligned_orientations:
        model.theta_init[:,:] = torch.rand(model.theta_init.size()) * (4*torch.pi) - 2*torch.pi
        model.theta_init[:,0] = 0
    else:
        model.theta_init[:,:] = torch.rand(1) * (4*torch.pi) - 2*torch.pi

    # Training: Extract reservoir activations
    try:
        target = train_dataset[:, (lag+washout):].numpy()  # Shape: (batch, time_steps, 5)
        dataset = train_dataset[:, :(2000+washout)].to(device)  # Shape: (batch, time_steps, 5)
        
        print(f"train_dataset shape: {train_dataset.shape}")
        print(f"dataset shape: {dataset.shape}")
        print(f"target shape: {target.shape}")
        
        # Get reservoir activations (no gradients needed)
        with torch.no_grad():
            # Model expects (batch, time_steps, features)
            states_list, _, _ = model(dataset, dataset)
        
        # Process activations - concatenate all states from states_list
        activations = torch.stack(states_list, dim=1)  # (batch, time, n_units*5)
        activations = activations.cpu().numpy()
        
        print(f"Raw activations shape: {activations.shape}")
        
        # Remove washout period from activations to match target
        activations = activations[:, washout:, :]  # Remove first washout time steps
        
        # Reshape for Ridge regression: (batch * time_steps, features)
        activations = activations.reshape(-1, activations.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        
        print(f"Activations shape after washout removal: {activations.shape}")
        print(f"Target shape: {target.shape}")
        
        # Check for NaN values
        if np.isnan(activations).any():
            print("NaN values detected in activations, ending trial")
            return float('inf')  # Return high error for failed trial
        
        # Train Ridge regression on reservoir activations
        scaler = preprocessing.StandardScaler().fit(activations)
        activations_scaled = scaler.transform(activations)
        classifier = Ridge(alpha=ridge_alpha, max_iter=1000).fit(activations_scaled, target)
        
        # Validation
        def test_esn(dataset, classifier, scaler):
            target = dataset[:, (lag+washout):].numpy()  # Shape: (batch, time_steps, 5)
            dataset = dataset[:, :(2000+washout)].to(device)  # Shape: (batch, time_steps, 5)

            with torch.no_grad():
                # Model expects (batch, time_steps, features)
                states_list, _, _ = model(dataset, dataset)
            
            # Process activations - concatenate all states from states_list
            activations = torch.stack(states_list, dim=1)  # (batch, time, n_units*5)
            activations = activations.cpu().numpy()
            
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

    database_name = "unicycle_nets_lorenz"
    study_name = f"lorenz_prediction_esn_lag25_200_units_notanh"
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
    with open('lorenz_unicycle_results.txt', 'w') as f:
        f.write(f"Best NRMSE: {best_value:.6f}\n")
        f.write(f"Best hyperparameters: {best_params}\n")
