#%%
import os
import sys
import time
import numpy as np
import torch
import random
from tqdm import tqdm

#%%
# Fix matplotlib backend issues - use non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no display required)
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn import preprocessing
import optuna
from matplotlib.animation import FuncAnimation

#%%
# Add the parent directory to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import n_params
from unicycle_network_class import UnicycleReservoir
from plot_robot_data import load_robot_data, extract_robot_ids, get_robot_states

#%%
# NARMA generation functions
def u(t, period_ratio=2, amplitude=0.2):
    """Input signal: sum of three sinusoids"""
    f1 = 2.11  # frequencies
    f2 = 3.73
    f3 = 4.33
    T = period_ratio
    u_val = amplitude * np.sin(2*np.pi * f1 * t / T) * np.sin(2*np.pi * f2 * t / T) * np.sin(2*np.pi * f3 * t / T)
    return u_val


def generate_narma(n_samples, narma_order=10, period_ratio=2, amplitude=0.2):
    """
    Generate NARMA-n time series.
    
    Args:
        n_samples: Number of time steps
        narma_order: Order of NARMA system (2, 3, or n >= 4)
        amplitude: Amplitude of input signal (will be normalized to [0, 0.5] for NARMA)
    
    Returns:
        u_input: Raw input signal from u(t) function
        u_normalized: Normalized input [0, 0.5] used for NARMA generation
        y_narma: NARMA output series
    """
    # Generate input signal using the sinusoidal function
    t = np.arange(n_samples) * 0.05
    u_input = np.array([u(ti, period_ratio=period_ratio, amplitude=amplitude) for ti in t])
    
    # Normalize input to [0, 0.5] for NARMA generation (standard NARMA input range)
    u_normalized = u_input - u_input.min()
    u_normalized = 0.5 * u_normalized / u_normalized.max()
    
    # Initialize NARMA output
    y_narma = np.zeros(n_samples)
    
    # Generate NARMA series based on order using normalized input
    if narma_order == 2:
        # NARMA-2 equations
        for k in range(1, n_samples - 1):
            y_narma[k + 1] = (0.4 * y_narma[k] + 
                            0.4 * y_narma[k] * y_narma[k - 1] + 
                            0.6 * u_normalized[k]**3 + 0.1)
                            
    elif narma_order == 3:
        # NARMA-3 equations
        a, b, c, d = 0.3, 0.05, 1.5, 0.1
        for k in range(2, n_samples - 1):
            y_sum = np.sum(y_narma[k - 2:k + 1])
            y_narma[k + 1] = (a * y_narma[k] + 
                            b * y_narma[k] * y_sum + 
                            c * u_normalized[k - 2] * u_normalized[k] + d)
                            
    else:
        # NARMA-n for n >= 4
        n = narma_order
        a, b, c, d = 0.3, 0.05, 1.5, 0.1
        for k in range(n - 1, n_samples - 1):
            y_sum = np.sum(y_narma[k - (n - 1):k + 1])
            y_narma[k + 1] = (a * y_narma[k] + 
                            b * y_narma[k] * y_sum + 
                            c * u_normalized[k - n + 1] * u_normalized[k] + d)
    
    # Check for numerical issues and normalize if needed
    if np.max(np.abs(y_narma)) > 10:
        print(f'WARNING: NARMA output exceeded safe range (max={np.max(np.abs(y_narma)):.2f}). Normalizing...')
        y_narma = y_narma / np.max(np.abs(y_narma))
    
    # Return BOTH raw input AND normalized input (the one used for NARMA)
    return u_input, u_normalized, y_narma



# Robot data configuration (set to None to use random initialization)
ROBOT_DATA_FILE = "/home/mariano/phd_code/unicycle-network/reference10.csv.gz"  # Path to robot data CSV file (e.g., "reference10.csv.gz" or "../reference10.csv.gz")
ROBOT_DATA_START_IDX = 100  # Start index in robot data for initial positions
USE_INITIAL_DISTANCES = True  # Set spring equilibrium distances from initial robot positions
#%%
# Load best parameters from Optuna study
NARMA_ORDER = 5
N_SAMPLES = 10000
POSITION_NOISE_STD = "0.001"
database_name = "unicycle_nets_narma_real_data_init"
study_name = f"narma{NARMA_ORDER}_noise_{POSITION_NOISE_STD}"
#%%
# Configuration
storage_name = f"sqlite:///{parent_dir}/optuna_databases/{database_name}.db"
study = optuna.create_study(storage=storage_name, study_name=study_name, directions=['minimize', 'minimize'], load_if_exists=True)
params = study.best_params
WASHOUT_FRACTION = params['washout_fraction']  # Fraction of samples to use as washout (default 0.1)
TRAIN_FRACTION = 0.8
POSITION_QUANTIZATION = None  # Position sensor resolution in meters (e.g., 0.01 = 1cm, 0.001 = 1mm, None = no quantization)
print(f"Loaded best parameters from study: {study_name}")

#%%
# Extract parameters
aligned_orientations = False#params.get('aligned_orientations', False)
ang_input = False #params.get('ang_input', True)
ang_connections = False#params.get('ang_connections', True)

n_units = 4
lin_stiff_min = params['lin_stiff_min']  # Match optuna_narma.py scaling
lin_stiff_max = params['lin_stiff_max']  # Match optuna_narma.py scaling
ang_stiff_min = params['ang_stiff_min']
ang_stiff_max = params['ang_stiff_max']
lin_damping_min = params['lin_damping_min']
lin_damping_max = params['lin_damping_max']
ang_damping_min = params['ang_damping_min']
ang_damping_max = params['ang_damping_max']
dt = 0.01#params['dt']
inp_bias = 0
anchor_con_fraction = params['anchor_con_fraction']
# magnitude_min = params['magnitude_min']
magnitude_max = params['magnitude_max']
n_connections_fraction = params['n_connections_fraction']
n_connections = int(n_units * n_connections_fraction)
washout_fraction = params['washout_fraction']
eq_dist_min = params['eq_dist_min']
eq_dist_max = params['eq_dist_max']
eq_dist_min_ang = params['eq_dist_min_ang']
eq_dist_max_ang = params['eq_dist_max_ang']
n_connections_anchor = int(n_units * anchor_con_fraction)
ridge_alpha = params.get('ridge_alpha', 0)
non_zero_fraction = params['non_zero_fraction']
period_ratio = params.get('period_ratio', 2) # Default to 2 if not in params
reservoir_input_scale = params.get('reservoir_input_scale', 1.0)*0.5  # Default to 1.0 if not in params

if ang_input:
    non_zero_fraction_ang = params['non_zero_fraction_ang']
    magnitude_min_ang = params['magnitude_min_ang']
    magnitude_max_ang = params['magnitude_max_ang']

if ang_connections:
    n_connections_ang_fraction = params["n_connections_ang_fraction"]
    n_connections_ang = int(n_connections_ang_fraction * n_units)
    anchor_con_fraction_ang = params['anchor_con_fraction_ang']
    n_connections_anchor_ang = int(anchor_con_fraction_ang * n_units)
else:
    n_connections_ang = 0
    n_connections_anchor_ang = 0

n_steps_readout = params.get('steps_readout', 0)

print(f"\nModel configuration:")
print(f"  n_units: {n_units}")
print(f"  aligned_orientations: {aligned_orientations}")
print(f"  ang_input: {ang_input}")
print(f"  ang_connections: {ang_connections}")
print(f"  robot_data: {ROBOT_DATA_FILE}")
print(f"  use_initial_distances: {USE_INITIAL_DISTANCES}")

#%%
# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#%%
# Generate NARMA series with fixed amplitude of 0.5
print(f'\n=== Generating NARMA-{NARMA_ORDER} target series ===')
print(f'NARMA amplitude: 0.5, Period ratio: {period_ratio}, Reservoir input scale: {reservoir_input_scale:.3f}')
u_input, u_input_norm, y_narma = generate_narma(N_SAMPLES, narma_order=5, period_ratio=period_ratio, amplitude=0.5)

# Check NARMA generation
print(f'NARMA target stats:')
print(f'  Range: [{y_narma.min():.6f}, {y_narma.max():.6f}]')
print(f'  Mean: {y_narma.mean():.6f}, Std: {y_narma.std():.6f}')
print(f'  First 10 values: {y_narma[:10]}')
print(f'  Period ratio: {period_ratio} (higher = slower input variation)')
print(f'  Input u[k]^3 range: [{(u_input**3).min():.6f}, {(u_input**3).max():.6f}]')

# WARNING: If period_ratio is very large, NARMA converges to near-constant
if period_ratio > 50:
    print(f'  ⚠ WARNING: period_ratio={period_ratio} is very large!')
    print(f'     Input changes slowly → NARMA converges to equilibrium')
    print(f'     This makes the prediction task trivial (just predict mean)')

# Split data
washout_samples = int(washout_fraction * N_SAMPLES)
valid_idx = np.arange(washout_samples, N_SAMPLES)
n_valid = len(valid_idx)
n_train = int(TRAIN_FRACTION * n_valid)

train_idx = valid_idx[:n_train]
test_idx = valid_idx[n_train:]

print(f'Total samples: {N_SAMPLES}')
print(f'Washout: {washout_samples} samples ({washout_fraction * 100:.1f}%)')
print(f'Training: {len(train_idx)} samples ({TRAIN_FRACTION * 100:.1f}%)')
print(f'Testing: {len(test_idx)} samples ({(1 - TRAIN_FRACTION) * 100:.1f}%)')

train_u = u_input[train_idx]
train_y = y_narma[train_idx]
test_u = u_input[test_idx]
test_y = y_narma[test_idx]
#%%
plt.plot(train_u)
plt.show()
#%%
# Create input maps
n_inp = 1
total_elements = n_inp * n_units
num_non_zero = int(total_elements * non_zero_fraction)

lin_input_map = torch.zeros(n_inp, n_units)
flat_indices = torch.randperm(total_elements)[:num_non_zero]
row_indices = flat_indices // n_units
col_indices = flat_indices % n_units
random_values = torch.rand(num_non_zero) * 2 * (magnitude_max) - magnitude_max
lin_input_map[row_indices, col_indices] = random_values

#%%
# Angular input map
ang_input_map = torch.zeros(n_inp, n_units)
if ang_input:
    num_non_zero_ang = int(total_elements * non_zero_fraction_ang)
    flat_indices_ang = torch.randperm(total_elements)[:num_non_zero_ang]
    row_indices_ang = flat_indices_ang // n_units
    col_indices_ang = flat_indices_ang % n_units
    random_values_ang = torch.rand(num_non_zero_ang) * (magnitude_max_ang - magnitude_min_ang) + magnitude_min_ang
    ang_input_map[row_indices_ang, col_indices_ang] = random_values_ang

#%%#%%
position_noise_std = float(POSITION_NOISE_STD)
# Initialize model
model = UnicycleReservoir(
    n_inp=n_inp, n_units=n_units, dt=dt, n_out=1,
    lin_input_map=lin_input_map, 
    lin_stiff_min=lin_stiff_min, lin_stiff_max=lin_stiff_max,
    lin_damping_min=lin_damping_min, lin_damping_max=lin_damping_max,
    ang_stiff_min=ang_stiff_min, ang_stiff_max=ang_stiff_max,
    ang_damping_min=ang_damping_min, ang_damping_max=ang_damping_max,
    eq_dist_min=eq_dist_min, eq_dist_max=eq_dist_max,
    eq_dist_min_ang=eq_dist_min_ang, eq_dist_max_ang=eq_dist_max_ang,  
    n_connections=n_connections, n_connections_anchor=n_connections_anchor, 
    n_past_steps_readout=n_steps_readout,
    n_connections_ang=n_connections_ang, n_connections_anchor_ang=n_connections_anchor_ang,
    inp_bias=inp_bias, ang_input_map=ang_input_map, use_capped_dynamics=True, max_speed=0.3, max_acceleration=3.0,
    position_noise_std=position_noise_std
)

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
model = model.to(device)

#%%
# Initialize model states
batch_size = 1  # Time series processing

# Option 1: Use robot data for initial positions
if ROBOT_DATA_FILE is not None:
    print(f"\n=== Loading robot data for initial positions ===")
    print(f"File: {ROBOT_DATA_FILE}")
    print(f"Start index: {ROBOT_DATA_START_IDX}")
    
    df = load_robot_data(ROBOT_DATA_FILE)
    robot_ids = extract_robot_ids(df)
    
    # Use only first n_units robots
    if len(robot_ids) < n_units:
        raise ValueError(f"Need {n_units} robots but only {len(robot_ids)} found in data")
    robot_ids = robot_ids[:n_units]
    
    # Extract initial states from robot data
    x_positions = np.zeros(n_units)
    z_positions = np.zeros(n_units)
    linear_velocities = np.zeros(n_units)
    angular_velocities = np.zeros(n_units)
    orientations = np.zeros(n_units)
    
    for i, robot_id in enumerate(robot_ids):
        states = get_robot_states(df, robot_id)
        x_positions[i] = states['pos_x'][ROBOT_DATA_START_IDX]
        z_positions[i] = states['pos_y'][ROBOT_DATA_START_IDX]
        linear_velocities[i] = states['linear_x'][ROBOT_DATA_START_IDX]
        angular_velocities[i] = states['omega'][ROBOT_DATA_START_IDX]
        
        # Compute orientation from quaternion
        qz = states['qz'][ROBOT_DATA_START_IDX]
        qw = states['qw'][ROBOT_DATA_START_IDX]
        theta = 2 * np.arctan2(qz, qw)
        orientations[i] = theta
    
    # Set initial states from robot data
    model.set_init_states(batch_size, x_positions, z_positions, orientations, 
                         linear_velocities, angular_velocities)
    
    print(f"Loaded initial positions from robot data")
    print(f"  X range: [{x_positions.min():.3f}, {x_positions.max():.3f}]")
    print(f"  Z range: [{z_positions.min():.3f}, {z_positions.max():.3f}]")
    
    # Optionally set equilibrium distances based on initial positions
    if USE_INITIAL_DISTANCES:
        print(f"Setting spring equilibrium distances from initial positions...")
        model.set_eq_distances_from_initial_positions()

# Option 2: Random initialization (default)
else:
    model.set_init_states_random(batch_size)
    print("\n=== Using random initial positions ===")

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

model.s_init[:, :] = 0
model.omega_init[:, :] = 0
if not aligned_orientations:
    # Only override orientations if NOT using robot data
    # if args.robot_data is None:
    model.theta_init[:, :] = torch.rand(model.theta_init.size()) * (4*torch.pi) - 2*torch.pi
    model.theta_init[:, 0] = 0
else:
    print("Using aligned orientations")
    model.theta_init[:, :] = torch.rand(1) * (4*torch.pi) - 2*torch.pi

#%%
# Washup period: Run model with zero input to let it settle near equilibrium
washup_steps = params.get('washup_steps', 0)
if washup_steps > 0:
    print(f"\n=== Running washup period ({washup_steps} steps) ===")
    x = model.x_init.clone()
    z = model.z_init.clone()
    theta = model.theta_init.clone()
    s = model.s_init.clone()
    omega = model.omega_init.clone()
    
    u_washup = torch.zeros((1, washup_steps, 1), device=device)
    
    with torch.no_grad():
        for t in range(washup_steps):
            linear_input = (u_washup[:, t]) @ model.lin_input_map
            angular_input = (u_washup[:, t]) @ model.ang_input_map
            
            x, z, theta, s, omega = model.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)
    
    # Update model's initial states to the equilibrated states
    model.set_init_states(batch_size, x, z, theta, s, omega)
    print(f"Washup complete - State ranges: x=[{x.min():.3f}, {x.max():.3f}], s=[{s.min():.3f}, {s.max():.3f}]")
else:
    print("\n=== No washup period (washup_steps=0) ===")

#%%
# No explicit washup phase needed - it's handled in data split
print("\n=== Training Ridge Regression ===")

# We need to include washout period in model forward pass for reservoir to stabilize,
# Extract reservoir activations for ALL samples (including washout)
print("\n=== Extracting reservoir activations ===")
full_input = torch.from_numpy(u_input * reservoir_input_scale).float().reshape(1, -1, 1).to(device)
print(f"Full input shape: {full_input.shape}")
print(f"Input range after scaling: [{full_input.min():.4f}, {full_input.max():.4f}]")
print(f"Washout samples: {washout_samples}")

# Get reservoir activations (no gradients needed)
with torch.no_grad():
    states_list, _, _ = model(full_input, full_input)

# Process activations - concatenate all states from states_list
activations_all = torch.stack(states_list, dim=1)  # (1, time_steps, n_units*5)
activations_all = activations_all.squeeze(0).cpu().numpy()  # (time_steps, n_units*5)

print(f"All activations shape: {activations_all.shape}")
print(f"All activations stats - mean: {activations_all.mean():.6f}, std: {activations_all.std():.6f}")

# Check for NaN values
if np.isnan(activations_all).any():
    print("WARNING: NaN values detected in activations")
else:
    print("No NaN values detected, proceeding with training")

# Check for zero variance
if activations_all.std() < 1e-10:
    print("WARNING: Activations have zero variance, reservoir not responding properly")

#%%
# Apply position quantization to simulate realistic sensor measurements
if POSITION_QUANTIZATION is not None and POSITION_QUANTIZATION > 0:
    print(f"\n=== Applying Position Quantization (resolution: {POSITION_QUANTIZATION}m) ===")
    
    # Quantize x and y positions (first 2*n_units features)
    activations_all_quantized = activations_all.copy()
    activations_all_quantized[:, :2*n_units] = np.round(activations_all[:, :2*n_units] / POSITION_QUANTIZATION) * POSITION_QUANTIZATION
    
    # Check impact of quantization
    quantization_error_x = np.abs(activations_all[:, :n_units] - activations_all_quantized[:, :n_units])
    quantization_error_y = np.abs(activations_all[:, n_units:2*n_units] - activations_all_quantized[:, n_units:2*n_units])
    print(f"Quantization error (X) - mean: {quantization_error_x.mean():.6f}, max: {quantization_error_x.max():.6f}")
    print(f"Quantization error (Y) - mean: {quantization_error_y.mean():.6f}, max: {quantization_error_y.max():.6f}")
    
    # Calculate information loss
    position_variance = activations_all[:, :2*n_units].var()
    quantization_noise_variance = ((activations_all[:, :2*n_units] - activations_all_quantized[:, :2*n_units])**2).mean()
    snr_db = 10 * np.log10(position_variance / (quantization_noise_variance + 1e-12))
    print(f"Signal-to-Quantization-Noise Ratio: {snr_db:.2f} dB")
    
    # Replace activations with quantized version
    activations_all = activations_all_quantized
    print(f"✓ Position quantization applied")
else:
    print(f"\n=== No Position Quantization (using continuous values) ===")

#%%
# Compare reservoir evolution with input vs without input
print("\n=== Comparing Input vs No-Input Evolution ===")
zero_input = torch.zeros_like(full_input).to(device)

# Reset model to same initial state for fair comparison
model.set_init_states(batch_size, model.x_init, model.z_init, model.theta_init, model.s_init, model.omega_init)

with torch.no_grad():
    states_list_zero, _, _ = model(zero_input, zero_input)

activations_zero = torch.stack(states_list_zero, dim=1)
activations_zero = activations_zero.squeeze(0).cpu().numpy()
activations_zero = activations_zero[:, :]

print(f"Zero-input activations shape: {activations_zero.shape}")
print(f"Zero-input activations stats - mean: {activations_zero.mean():.6f}, std: {activations_zero.std():.6f}")

# Create comparison plots with overlays and differences
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
time_steps = np.arange(activations_all.shape[0])

# X positions - overlay and difference
for i in range(n_units):
    axes[0, 0].plot(time_steps, activations_all[:, i], alpha=0.4, color='blue', linewidth=0.8)
    axes[0, 0].plot(time_steps, activations_zero[:, i], alpha=0.4, color='red', linewidth=0.8, linestyle='--')
axes[0, 0].set_title('X Positions (Blue: With Input, Red: Zero Input)')
axes[0, 0].set_ylabel('x')
axes[0, 0].grid(True, alpha=0.3)

x_diff = activations_all[:, :n_units] - activations_zero[:, :n_units]
axes[0, 1].plot(time_steps, x_diff, alpha=0.5)
axes[0, 1].set_title('X Position Difference (With Input - Zero Input)')
axes[0, 1].set_ylabel('Δx')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Y positions - overlay and difference
for i in range(n_units):
    axes[1, 0].plot(time_steps, activations_all[:, n_units + i], alpha=0.4, color='blue', linewidth=0.8)
    axes[1, 0].plot(time_steps, activations_zero[:, n_units + i], alpha=0.4, color='red', linewidth=0.8, linestyle='--')
axes[1, 0].set_title('Y Positions (Blue: With Input, Red: Zero Input)')
axes[1, 0].set_ylabel('y')
axes[1, 0].grid(True, alpha=0.3)

y_diff = activations_all[:, n_units:2*n_units] - activations_zero[:, n_units:2*n_units]
axes[1, 1].plot(time_steps, y_diff, alpha=0.5)
axes[1, 1].set_title('Y Position Difference (With Input - Zero Input)')
axes[1, 1].set_ylabel('Δy')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Theta angles - overlay and difference
for i in range(n_units):
    axes[2, 0].plot(time_steps, activations_all[:, 2*n_units + i], alpha=0.4, color='blue', linewidth=0.8)
    axes[2, 0].plot(time_steps, activations_zero[:, 2*n_units + i], alpha=0.4, color='red', linewidth=0.8, linestyle='--')
axes[2, 0].set_title('Theta Angles (Blue: With Input, Red: Zero Input)')
axes[2, 0].set_xlabel('Time Step')
axes[2, 0].set_ylabel('θ')
axes[2, 0].grid(True, alpha=0.3)

theta_diff = activations_all[:, 2*n_units:] - activations_zero[:, 2*n_units:]
axes[2, 1].plot(time_steps, theta_diff, alpha=0.5)
axes[2, 1].set_title('Theta Difference (With Input - Zero Input)')
axes[2, 1].set_xlabel('Time Step')
axes[2, 1].set_ylabel('Δθ')
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

plt.suptitle(f'NARMA-{NARMA_ORDER}: Reservoir Evolution Comparison (Input vs No Input)', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(f'{parent_dir}/plots/narma{NARMA_ORDER}_input_comparison.png', dpi=150, bbox_inches='tight')
print(f"Input comparison plot saved to: {parent_dir}/plots/narma{NARMA_ORDER}_input_comparison.png")
plt.close()

# Split activations using same indices as targets (after washout)
valid_idx = np.arange(washout_samples, N_SAMPLES)
n_valid = len(valid_idx)
n_train = int(TRAIN_FRACTION * n_valid)

train_idx = valid_idx[:n_train]
test_idx = valid_idx[n_train:]

X_train = activations_all[train_idx, :n_units*2]
y_train_full = y_narma[train_idx]  # Keep as 1D array
X_test = activations_all[test_idx, :n_units*2]
y_test_full = y_narma[test_idx]  # Keep as 1D array

print(f"\n=== Train/Test Split ===")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train_full.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test_full.shape}")
print(f"Train target stats - mean: {y_train_full.mean():.6f}, std: {y_train_full.std():.6f}")

# Train Ridge regression on reservoir activations
# Standard per-feature scaling (each column independently)
from sklearn.preprocessing import StandardScaler

X_train_features = X_train[:, :]
X_test_features = X_test[:, :]

# Fit scaler on training data and transform both train and test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

print(f"Feature scaling - each coordinate normalized independently")
print(f"Scaled train stats - mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")

# Train Ridge regression with intercept (more standard approach)
# StandardScaler already centered features, so intercept captures target mean
regressor = Ridge(alpha=ridge_alpha, max_iter=1000, solver='lsqr', fit_intercept=True).fit(X_train_scaled, y_train_full)

# Print regressor weights statistics
weights = regressor.coef_.ravel()  # Flatten to 1D array
intercept = regressor.intercept_
# Handle both scalar and array intercepts
intercept_val = intercept[0] if isinstance(intercept, np.ndarray) and intercept.size > 0 else float(intercept)
print(f"Regressor weights - mean: {weights.mean():.6f}, std: {weights.std():.6f}, max: {weights.max():.6f}, min: {weights.min():.6f}")
print(f"Regressor intercept: {intercept_val:.6f}")
print(f"Ridge alpha used: {ridge_alpha}")

#%%
# DIAGNOSTIC ANALYSIS: Why is performance poor?
print("\n" + "="*60)
print("DIAGNOSTIC ANALYSIS: Activation-Target Relationship")
print("="*60)

# Print key hyperparameters first
print("\n0. Key Hyperparameters:")
print(f"   dt: {dt:.6f}")
print(f"   lin_stiff_min: {lin_stiff_min:.6f}, lin_stiff_max: {lin_stiff_max:.6f}")
print(f"   ang_stiff_min: {ang_stiff_min:.6f}, ang_stiff_max: {ang_stiff_max:.6f}")
print(f"   lin_damping_min: {lin_damping_min:.6f}, lin_damping_max: {lin_damping_max:.6f}")
print(f"   ang_damping_min: {ang_damping_min:.6f}, ang_damping_max: {ang_damping_max:.6f}")
# print(f"   magnitude_min: {magnitude_min:.6f}, magnitude_max: {magnitude_max:.6f}")
if ang_input:
    print(f"   magnitude_min_ang: {magnitude_min_ang:.6f}, magnitude_max_ang: {magnitude_max_ang:.6f}")
print(f"   ridge_alpha: {ridge_alpha:.6e}")
print(f"   period_ratio: {period_ratio}")
print(f"   Input map non-zeros: {(lin_input_map != 0).sum()}/{lin_input_map.numel()}")
print(f"   Input map range: [{lin_input_map.min():.6f}, {lin_input_map.max():.6f}]")

# Calculate magnitude_min from input map if not in params (for backward compatibility)
try:
    magnitude_min_val = magnitude_min
except NameError:
    # magnitude_min not defined, calculate from input map
    non_zero_weights = lin_input_map[lin_input_map != 0]
    magnitude_min_val = non_zero_weights.min().item() if len(non_zero_weights) > 0 else 0.0

# Generate LaTeX table for easy insertion into documents
latex_table_file = f'{parent_dir}/plots/narma{NARMA_ORDER}_hyperparameters_table.tex'
with open(latex_table_file, 'w') as f:
    f.write("% Hyperparameters table for NARMA experiment\n")
    f.write("% Generated automatically from narma_ang_input_coupled.py\n\n")
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Hyperparameters for NARMA-" + str(NARMA_ORDER) + " Experiment}\n")
    f.write("\\label{tab:narma" + str(NARMA_ORDER) + "_hyperparameters}\n")
    f.write("\\begin{tabular}{lcc}\n")
    f.write("\\hline\n")
    f.write("\\textbf{Parameter} & \\textbf{Value} & \\textbf{Description} \\\\\n")
    f.write("\\hline\n")
    
    # Network architecture
    f.write(f"$n_{{\\text{{units}}}}$ & {n_units} & Number of unicycle units \\\\\n")
    f.write(f"$\\Delta t$ & {dt:.4f} & Time step (s) \\\\\n")
    f.write("\\hline\n")
    
    # Linear stiffness
    f.write(f"$k_{{\\text{{lin,min}}}}$ & {lin_stiff_min:.4f} & Min linear spring stiffness \\\\\n")
    f.write(f"$k_{{\\text{{lin,max}}}}$ & {lin_stiff_max:.4f} & Max linear spring stiffness \\\\\n")
    
    # Angular stiffness
    f.write(f"$k_{{\\text{{ang,min}}}}$ & {ang_stiff_min:.4f} & Min angular spring stiffness \\\\\n")
    f.write(f"$k_{{\\text{{ang,max}}}}$ & {ang_stiff_max:.4f} & Max angular spring stiffness \\\\\n")
    
    # Linear damping
    f.write(f"$c_{{\\text{{lin,min}}}}$ & {lin_damping_min:.4f} & Min linear damping coefficient \\\\\n")
    f.write(f"$c_{{\\text{{lin,max}}}}$ & {lin_damping_max:.4f} & Max linear damping coefficient \\\\\n")
    
    # Angular damping
    f.write(f"$c_{{\\text{{ang,min}}}}$ & {ang_damping_min:.4f} & Min angular damping coefficient \\\\\n")
    f.write(f"$c_{{\\text{{ang,max}}}}$ & {ang_damping_max:.4f} & Max angular damping coefficient \\\\\n")
    f.write("\\hline\n")
    
    # Input mapping
    f.write(f"$w_{{\\text{{min}}}}$ & {magnitude_min_val:.4f} & Min input weight magnitude \\\\\n")
    f.write(f"$w_{{\\text{{max}}}}$ & {magnitude_max:.4f} & Max input weight magnitude \\\\\n")
    if ang_input:
        f.write(f"$w_{{\\text{{ang,min}}}}$ & {magnitude_min_ang:.4f} & Min angular input weight \\\\\n")
        f.write(f"$w_{{\\text{{ang,max}}}}$ & {magnitude_max_ang:.4f} & Max angular input weight \\\\\n")
    
    non_zero_frac = (lin_input_map != 0).sum().item() / lin_input_map.numel()
    f.write(f"Input sparsity & {non_zero_frac:.2%} & Fraction of non-zero input weights \\\\\n")
    f.write("\\hline\n")
    
    # Ridge regression
    f.write(f"$\\alpha_{{\\text{{ridge}}}}$ & {ridge_alpha:.2e} & Ridge regression regularization \\\\\n")
    
    # Signal parameters
    f.write(f"Period ratio & {period_ratio} & Input signal period scaling \\\\\n")
    f.write(f"NARMA order & {NARMA_ORDER} & Order of NARMA system \\\\\n")
    
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"\n✓ LaTeX table saved to: {latex_table_file}")

# Also generate a CSV version for easier editing
csv_table_file = f'{parent_dir}/plots/narma{NARMA_ORDER}_hyperparameters_table.csv'
with open(csv_table_file, 'w') as f:
    f.write("Parameter,Value,Description\n")
    f.write(f"n_units,{n_units},Number of unicycle units\n")
    f.write(f"dt,{dt:.6f},Time step (s)\n")
    f.write(f"lin_stiff_min,{lin_stiff_min:.6f},Min linear spring stiffness\n")
    f.write(f"lin_stiff_max,{lin_stiff_max:.6f},Max linear spring stiffness\n")
    f.write(f"ang_stiff_min,{ang_stiff_min:.6f},Min angular spring stiffness\n")
    f.write(f"ang_stiff_max,{ang_stiff_max:.6f},Max angular spring stiffness\n")
    f.write(f"lin_damping_min,{lin_damping_min:.6f},Min linear damping coefficient\n")
    f.write(f"lin_damping_max,{lin_damping_max:.6f},Max linear damping coefficient\n")
    f.write(f"ang_damping_min,{ang_damping_min:.6f},Min angular damping coefficient\n")
    f.write(f"ang_damping_max,{ang_damping_max:.6f},Max angular damping coefficient\n")
    f.write(f"magnitude_min,{magnitude_min_val:.6f},Min input weight magnitude\n")
    f.write(f"magnitude_max,{magnitude_max:.6f},Max input weight magnitude\n")
    if ang_input:
        f.write(f"magnitude_min_ang,{magnitude_min_ang:.6f},Min angular input weight\n")
        f.write(f"magnitude_max_ang,{magnitude_max_ang:.6f},Max angular input weight\n")
    f.write(f"input_sparsity,{non_zero_frac:.4f},Fraction of non-zero input weights\n")
    f.write(f"ridge_alpha,{ridge_alpha:.6e},Ridge regression regularization\n")
    f.write(f"period_ratio,{period_ratio},Input signal period scaling\n")
    f.write(f"narma_order,{NARMA_ORDER},Order of NARMA system\n")

print(f"✓ CSV table saved to: {csv_table_file}")

# 1. Check correlation between activations and target
from scipy.stats import pearsonr, spearmanr

print("\n1. Individual Feature Correlations with Target:")
correlations = []
for i in range(X_train_scaled.shape[1]):
    if X_train_scaled[:, i].std() > 1e-10:  # Only compute for non-constant features
        corr, p_value = pearsonr(X_train_scaled[:, i], y_train_full)  # Use original target, not centered
        correlations.append(corr)
    else:
        correlations.append(0.0)  # Zero correlation for constant features

correlations = np.array(correlations)
valid_corrs = correlations[~np.isnan(correlations)]
print(f"   Max absolute correlation: {np.abs(valid_corrs).max():.4f}")
print(f"   Mean absolute correlation: {np.abs(valid_corrs).mean():.4f}")
print(f"   Features with |corr| > 0.1: {np.sum(np.abs(correlations) > 0.1)}/{len(correlations)}")
print(f"   Features with |corr| > 0.2: {np.sum(np.abs(correlations) > 0.2)}/{len(correlations)}")

# 2. Check temporal dynamics in activations
print("\n2. Temporal Dynamics of Activations:")
temporal_changes = np.diff(X_train, axis=0)
print(f"   Mean temporal change: {np.abs(temporal_changes).mean():.6f}")
print(f"   Std of temporal change: {np.abs(temporal_changes).std():.6f}")
print(f"   Activation range: [{X_train.min():.4f}, {X_train.max():.4f}]")
print(f"   Target range: [{y_train_full.min():.4f}, {y_train_full.max():.4f}]")

# 3. Check for constant or near-constant features
print("\n3. Feature Variability Check:")
feature_stds = X_train.std(axis=0)
n_low_variance = np.sum(feature_stds < 0.01)
n_constant = np.sum(feature_stds < 1e-10)
print(f"   Features with std < 0.01: {n_low_variance}/{len(feature_stds)}")
print(f"   Features with std < 1e-10 (constant): {n_constant}/{len(feature_stds)}")
print(f"   Min feature std: {feature_stds.min():.6f}")
print(f"   Max feature std: {feature_stds.max():.6f}")

# 4. Check condition number of feature matrix
print("\n4. Feature Matrix Conditioning:")
U, s, Vt = np.linalg.svd(X_train_scaled, full_matrices=False)
condition_number = s[0] / s[-1]
print(f"   Condition number: {condition_number:.2e}")
print(f"   Largest singular value: {s[0]:.4f}")
print(f"   Smallest singular value: {s[-1]:.4f}")
print(f"   Effective rank (s > 1e-10): {np.sum(s > 1e-10)}/{len(s)}")

# 5. Analyze input signal properties
print("\n5. Input Signal Analysis:")
print(f"   Input range: [{u_input.min():.4f}, {u_input.max():.4f}]")
print(f"   Input mean: {u_input.mean():.6f}, std: {u_input.std():.6f}")
input_changes = np.abs(np.diff(u_input))
print(f"   Input temporal changes - mean: {input_changes.mean():.6f}, max: {input_changes.max():.6f}")

# 6. Check if activations are responsive to input
print("\n6. Input-Activation Relationship:")
# Compare activations from different time windows
early_acts = X_train[:100, :].std()
late_acts = X_train[-100:, :].std()
print(f"   Early activation std: {early_acts:.6f}")
print(f"   Late activation std: {late_acts:.6f}")
print(f"   Ratio (late/early): {late_acts/early_acts:.4f}")

# 7. Analyze target properties
print("\n7. Target Signal Analysis:")
target_changes = np.abs(np.diff(y_train_full.flatten()))
print(f"   Target temporal changes - mean: {target_changes.mean():.6f}, max: {target_changes.max():.6f}")
print(f"   Target autocorrelation (lag 1): {np.corrcoef(y_train_full[:-1].flatten(), y_train_full[1:].flatten())[0, 1]:.4f}")

# 8. Memory capacity check
print("\n8. Memory Capacity Indicators:")
# Check if activations maintain information over time
# Remove constant features for this calculation
active_features = feature_stds > 1e-10
X_train_active = X_train[:, active_features]
lag_correlations = []
for lag in [1, 5, 10, 20, 50]:
    if lag < len(X_train_active):
        try:
            corr_matrix = np.corrcoef(X_train_active[:-lag].T, X_train_active[lag:].T)
            cross_corr = corr_matrix[:X_train_active.shape[1], X_train_active.shape[1]:]
            mean_corr = np.nanmean(np.abs(cross_corr))
            lag_correlations.append(mean_corr)
            print(f"   Mean |correlation| at lag {lag}: {mean_corr:.4f}")
        except:
            print(f"   Mean |correlation| at lag {lag}: Could not compute")

print("\n" + "="*60)
print("POTENTIAL ISSUES TO INVESTIGATE:")
print("="*60)

issues = []
if np.abs(valid_corrs).max() < 0.01:
    issues.append("❌ CRITICAL: No correlation between features and target (max < 0.01)")
elif np.abs(valid_corrs).max() < 0.1:
    issues.append("⚠ Very weak correlations between features and target (< 0.1)")
elif np.abs(valid_corrs).max() < 0.3:
    issues.append("⚠ Weak correlations between features and target (< 0.3)")

if condition_number > 1e10:
    issues.append(f"⚠ Extremely poorly conditioned feature matrix ({condition_number:.2e})")
if n_constant > 0:
    issues.append(f"⚠ {n_constant} completely constant features (zero variance)")
if n_low_variance > len(feature_stds) * 0.1:
    issues.append(f"⚠ {n_low_variance} features with very low variance (> 10% of total)")

# Check if reservoir is responding to input
input_change_rate = input_changes.mean()
activation_change_rate = np.abs(temporal_changes).mean()
target_change_rate = target_changes.mean()

if activation_change_rate < input_change_rate / 100:
    issues.append(f"❌ CRITICAL: Reservoir not responding to input!")
    issues.append(f"   Input changes {input_change_rate:.6f}/step, activations only {activation_change_rate:.6f}/step")
    issues.append(f"   Ratio: {activation_change_rate/input_change_rate:.6f} (should be closer to 0.1-1.0)")

if activation_change_rate < target_change_rate / 10:
    issues.append(f"⚠ Activations changing much slower than target")
    issues.append(f"   Target: {target_change_rate:.6f}/step, Activations: {activation_change_rate:.6f}/step")

if late_acts/early_acts < 0.5 or late_acts/early_acts > 2.0:
    issues.append(f"⚠ Activation dynamics changing significantly over time (ratio: {late_acts/early_acts:.2f})")
if y_train_full.std() < 0.01:
    issues.append(f"⚠ Target has very low variance (std={y_train_full.std():.6f})")

# Check lag correlations
if len(lag_correlations) > 0 and all(lc > 0.7 for lc in lag_correlations):
    issues.append(f"⚠ Reservoir has excessive memory (lag correlations > 0.7 up to lag 50)")
    issues.append(f"   This suggests overdamped or stuck dynamics")

if issues:
    for issue in issues:
        print(issue)
else:
    print("✓ No obvious issues detected in diagnostics")

print("="*60 + "\n")

# Visualize regressor weights as heatmap
fig = plt.figure(figsize=(16, 10))

# Determine how many features were used (check actual training data shape)
n_features_used = X_train_features.shape[1]
n_states_per_unit = n_features_used // n_units

# Map indices to state names
state_names = ['x', 'z', 'θ', 's', 'ω'][:n_states_per_unit]

# Reshape weights for better visualization
weights_reshaped = weights.reshape(n_states_per_unit, n_units)

# Extract individual state weights
state_weights = [weights[i*n_units:(i+1)*n_units] for i in range(n_states_per_unit)]

# Print detailed weight statistics
print("\n" + "="*60)
print("RIDGE REGRESSOR WEIGHT ANALYSIS")
print("="*60)
print(f"\nFeature information:")
print(f"  Total features used: {n_features_used}")
print(f"  States per unit: {n_states_per_unit} {state_names}")
print(f"  Number of units: {n_units}")
print(f"\nOverall statistics:")
print(f"  Total weights: {len(weights)}")
print(f"  Mean: {weights.mean():.6f}")
print(f"  Std: {weights.std():.6f}")
print(f"  Max: {weights.max():.6f} (at index {weights.argmax()})")
print(f"  Min: {weights.min():.6f} (at index {weights.argmin()})")
print(f"  Median: {np.median(weights):.6f}")
print(f"  Non-zero weights: {np.sum(np.abs(weights) > 1e-6)}/{len(weights)}")
print(f"  Intercept: {intercept_val:.6f}")

print(f"\nBy state type:")
for i, (name, w) in enumerate(zip(state_names, state_weights)):
    print(f"  {name:>5} weights - mean: {w.mean():+.6f}, std: {w.std():.6f}, "
          f"max: {w.max():+.6f}, min: {w.min():+.6f}, "
          f"|mean|: {np.abs(w).mean():.6f}")

print(f"\nTop 10 largest magnitude weights:")
top_indices = np.argsort(np.abs(weights))[-10:][::-1]
for rank, idx in enumerate(top_indices, 1):
    state_type = state_names[idx // n_units]
    unit_idx = idx % n_units
    print(f"  {rank:2d}. {state_type}[{unit_idx}] = {weights[idx]:+.6f}  (|w| = {np.abs(weights[idx]):.6f})")

# Save weights to text file
weights_file = f'{parent_dir}/plots/narma{NARMA_ORDER}_regressor_weights.txt'
with open(weights_file, 'w') as f:
    f.write("RIDGE REGRESSOR WEIGHTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Intercept: {intercept_val:.6f}\n")
    f.write(f"Ridge alpha: {ridge_alpha:.6e}\n")
    f.write(f"Features used: {n_states_per_unit} states per unit {state_names}\n\n")
    f.write(f"Overall statistics:\n")
    f.write(f"  Mean: {weights.mean():.6f}\n")
    f.write(f"  Std: {weights.std():.6f}\n")
    f.write(f"  Max: {weights.max():.6f}\n")
    f.write(f"  Min: {weights.min():.6f}\n\n")
    
    f.write("Weights by state and unit:\n")
    for i in range(n_units):
        weight_str = ", ".join([f"{name}={state_weights[j][i]:+.6f}" 
                               for j, name in enumerate(state_names)])
        f.write(f"  Unit {i}: {weight_str}\n")

print(f"Saved weights to {weights_file}")
print("="*60 + "\n")

# Plot 1: Heatmap
plt.subplot(2, 3, 1)
im = plt.imshow(weights_reshaped, aspect='auto', cmap='RdBu_r', 
                vmin=-np.abs(weights).max(), vmax=np.abs(weights).max())
plt.colorbar(im, label='Weight value')
plt.xlabel('Unicycle index')
plt.ylabel('State type')
plt.yticks(range(n_states_per_unit), state_names)
plt.title('Ridge Regressor Weights Heatmap')

# Plot 2: Grouped bar plot
plt.subplot(2, 3, 2)
x_pos = np.arange(n_units)
width = 0.8 / n_states_per_unit
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, (name, w) in enumerate(zip(state_names, state_weights)):
    offset = width * (i - n_states_per_unit/2 + 0.5)
    plt.bar(x_pos + offset, w, width, label=f'{name} weights', 
            alpha=0.8, color=colors[i % len(colors)])

plt.xlabel('Unicycle index')
plt.ylabel('Weight value')
plt.title('Weights by State Type')
plt.legend(loc='best')
plt.grid(True, alpha=0.3, axis='y')

# Plot 3: Absolute magnitude comparison
plt.subplot(2, 3, 3)
for i, (name, w) in enumerate(zip(state_names, state_weights)):
    offset = width * (i - n_states_per_unit/2 + 0.5)
    plt.bar(x_pos + offset, np.abs(w), width, label=f'|{name} weights|', 
            alpha=0.8, color=colors[i % len(colors)])

plt.xlabel('Unicycle index')
plt.ylabel('Absolute weight value')
plt.title('Absolute Weight Magnitudes')
plt.legend(loc='best')
plt.grid(True, alpha=0.3, axis='y')

# Plot 4: Weight distribution histogram
plt.subplot(2, 3, 4)
plt.hist(weights, bins=30, alpha=0.7, edgecolor='black')
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
plt.axvline(weights.mean(), color='green', linestyle='--', linewidth=2, label='Mean')
plt.xlabel('Weight value')
plt.ylabel('Count')
plt.title('Weight Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Cumulative weight contribution
plt.subplot(2, 3, 5)
sorted_abs_weights = np.sort(np.abs(weights))[::-1]
cumulative_contrib = np.cumsum(sorted_abs_weights) / np.sum(sorted_abs_weights)
plt.plot(cumulative_contrib, linewidth=2)
plt.axhline(0.9, color='red', linestyle='--', alpha=0.7, label='90%')
plt.axhline(0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
plt.xlabel('Number of weights (sorted by magnitude)')
plt.ylabel('Cumulative contribution')
plt.title('Cumulative Weight Contribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Weight magnitude by state type (box plot)
plt.subplot(2, 3, 6)
weight_data = [np.abs(w) for w in state_weights]
bp = plt.boxplot(weight_data, labels=state_names, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
plt.ylabel('Absolute weight magnitude')
plt.xlabel('State type')
plt.title('Weight Distribution by State Type')
plt.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Ridge Regressor Weights Analysis (α={ridge_alpha:.2e}, intercept={intercept_val:.3f}, {n_states_per_unit} states)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{parent_dir}/plots/narma{NARMA_ORDER}_regressor_weights.png', dpi=150, bbox_inches='tight')
print(f"Saved regressor weights visualization to {parent_dir}/plots/narma{NARMA_ORDER}_regressor_weights.png")
plt.close()

#%%
# Additional diagnostic visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Correlation heatmap
axes[0, 0].bar(range(len(correlations)), np.abs(correlations))
axes[0, 0].set_xlabel('Feature Index')
axes[0, 0].set_ylabel('|Correlation| with Target')
axes[0, 0].set_title('Feature-Target Correlations')
axes[0, 0].axhline(y=0.1, color='r', linestyle='--', label='0.1 threshold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Sample activations vs target
sample_size = min(500, len(X_train))
axes[0, 1].plot(y_train_full[:sample_size], label='Target', linewidth=2)  # Use original target
# Show a few most correlated features
top_features = np.argsort(np.abs(correlations))[-3:]
for i, feat_idx in enumerate(top_features):
    axes[0, 1].plot(X_train_scaled[:sample_size, feat_idx], 
                    alpha=0.6, label=f'Feature {feat_idx} (corr={correlations[feat_idx]:.3f})')
axes[0, 1].set_xlabel('Time Step')
axes[0, 1].set_ylabel('Scaled Value')
axes[0, 1].set_title('Target vs Most Correlated Features')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Feature variance distribution
axes[1, 0].hist(feature_stds, bins=30, edgecolor='black')
axes[1, 0].set_xlabel('Feature Standard Deviation')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Distribution of Feature Variability')
axes[1, 0].axvline(x=0.01, color='r', linestyle='--', label='Low variance threshold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Singular value spectrum
axes[1, 1].semilogy(s, 'o-')
axes[1, 1].set_xlabel('Singular Value Index')
axes[1, 1].set_ylabel('Singular Value')
axes[1, 1].set_title(f'Singular Value Spectrum (Condition: {condition_number:.2e})')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{parent_dir}/plots/narma{NARMA_ORDER}_diagnostics.png', dpi=150, bbox_inches='tight')
print(f"Saved diagnostic plots to {parent_dir}/plots/narma{NARMA_ORDER}_diagnostics.png")
plt.close()

# Check regressor performance on training data
train_predictions = regressor.predict(X_train_scaled).ravel()  # Flatten to 1D
train_std = train_predictions.std()
print(f"Train predictions stats - mean: {train_predictions.mean():.6f}, std: {train_std:.6f}")

if train_std < 1e-10:
    print("WARNING: Regressor predicting constant values, may be over-regularized")
    print(f"         Consider reducing ridge_alpha (currently {ridge_alpha})")

#%%
# Test evaluation
print("\n=== Testing ===")
# Scaler already fitted on X_train_features, just transform test features
# No need to apply manual scaling - scaler handles it
predictions = regressor.predict(X_test_scaled)#.ravel()  # Flatten to 1D
# y_test_full = y_narma[test_idx].reshape(-1, 1)
# Calculate metrics
mse = np.mean(np.square(predictions - y_test_full))
target_var = np.var(y_test_full)
nmse = mse / (target_var + 1e-9)
rmse = np.sqrt(mse)
target_std = np.std(y_test_full)
nrmse = rmse / (target_std + 1e-9)

print(f"Test MSE: {mse:.6f}")
print(f"Test NMSE: {nmse:.6f}")
print(f"Test NRMSE: {nrmse:.6f}")
# print(f"Number of trainable parameters: {n_params(regressor)}")

#%%
# Visualization: Plot predictions vs targets
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Plot 1: Input signal (use test indices from original u_input)
test_u_plot = u_input[test_idx]
axes[0].plot(test_u_plot, label='Input Signal', alpha=0.7)
axes[0].set_ylabel('Input')
axes[0].set_title('NARMA Input Signal (Test Set)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Target vs Prediction
axes[1].plot(y_test_full, label='Target', alpha=0.7)
axes[1].plot(predictions, label='Prediction', alpha=0.7)
axes[1].set_ylabel('Output')
axes[1].set_title(f'NARMA-{NARMA_ORDER} Prediction (NMSE={nmse:.4f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Prediction Error
error = predictions - y_test_full
axes[2].plot(error, label='Prediction Error', alpha=0.7, color='red')
axes[2].set_xlabel('Time Step')
axes[2].set_ylabel('Error')
axes[2].set_title('Prediction Error')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{parent_dir}/plots/narma{NARMA_ORDER}_prediction.png', dpi=150)
print(f"\nPrediction plot saved to: {parent_dir}/plots/narma{NARMA_ORDER}_prediction.png")
plt.close()

#%%
# Visualization: State evolution
# Extract states from training data for visualization
print("\n=== Analyzing State Evolution ===")
all_states_time_res = torch.stack(states_list, dim=1)  # (batch=1, time, n_units*5)
print(f"States tensor shape: {all_states_time_res.shape}")

sample_idx = 0  # Only one sample for time series
x_states = all_states_time_res[sample_idx, :, 0:n_units].detach().cpu().numpy().T
y_states = all_states_time_res[sample_idx, :, n_units:2*n_units].detach().cpu().numpy().T
theta_states = all_states_time_res[sample_idx, :, 2*n_units:3*n_units].detach().cpu().numpy().T
s_states = all_states_time_res[sample_idx, :, 3*n_units:4*n_units].detach().cpu().numpy().T
omega_states = all_states_time_res[sample_idx, :, 4*n_units:5*n_units].detach().cpu().numpy().T

# Apply quantization to positions for visualization (same as used for training)
if POSITION_QUANTIZATION is not None and POSITION_QUANTIZATION > 0:
    print(f"Applying quantization to position plots (resolution: {POSITION_QUANTIZATION}m)")
    x_states = np.round(x_states / POSITION_QUANTIZATION) * POSITION_QUANTIZATION
    y_states = np.round(y_states / POSITION_QUANTIZATION) * POSITION_QUANTIZATION

# Save individual state plots
fig, axes = plt.subplots(5, 1, figsize=(15, 12))

axes[0].plot(x_states.T)
axes[0].set_ylabel('x states')
title_suffix = f' (quantized: {POSITION_QUANTIZATION}m)' if POSITION_QUANTIZATION else ''
axes[0].set_title('Position States (x)' + title_suffix)
axes[0].grid(True, alpha=0.3)

axes[1].plot(y_states.T)
axes[1].set_ylabel('y states')
axes[1].set_title('Position States (y)' + title_suffix)
axes[1].grid(True, alpha=0.3)

axes[2].plot(theta_states.T)
axes[2].set_ylabel('theta states')
axes[2].set_title('Orientation States (theta)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(s_states.T)
axes[3].set_ylabel('s states')
axes[3].set_title('Linear Velocity States (s)')
axes[3].grid(True, alpha=0.3)

axes[4].plot(omega_states.T)
axes[4].set_xlabel('Time Step')
axes[4].set_ylabel('omega states')
axes[4].set_title('Angular Velocity States (omega)')
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{parent_dir}/plots/narma{NARMA_ORDER}_states_evolution.png', dpi=150)
print(f"State evolution plot saved to: {parent_dir}/plots/narma{NARMA_ORDER}_states_evolution.png")
plt.close()

#%%
# Individual unit position plots (separate subplots for x and y)
print("\n=== Creating Individual Unit Position Plots ===")
time_steps = np.arange(x_states.shape[1])

# Determine grid layout for subplots
n_cols = min(3, n_units)  # Max 3 columns
n_rows = int(np.ceil(n_units / n_cols))

# Create title suffix for quantization info
quant_suffix = f' (quantized: {POSITION_QUANTIZATION}m)' if POSITION_QUANTIZATION else ''

# X positions
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
if n_units == 1:
    axes = np.array([axes])
axes = axes.flatten()

for i in range(n_units):
    ax = axes[i]
    ax.plot(time_steps[:2000], x_states[i, :2000], alpha=0.8, linewidth=1.5, color='C0')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('x position (m)')
    anchor_text = ' (Anchor)' if i == 0 else ''
    ax.set_title(f'Unit {i} - X Position{anchor_text}{quant_suffix}')
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for i in range(n_units, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(f'{parent_dir}/plots/narma{NARMA_ORDER}_individual_unit_x_positions.png', dpi=150, bbox_inches='tight')
print(f"Individual unit X position plots saved to: {parent_dir}/plots/narma{NARMA_ORDER}_individual_unit_x_positions.png")
plt.close()

# Y positions
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
if n_units == 1:
    axes = np.array([axes])
axes = axes.flatten()

for i in range(n_units):
    ax = axes[i]
    ax.plot(time_steps, y_states[i, :], alpha=0.8, linewidth=1.5, color='C1')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('y position (m)')
    anchor_text = ' (Anchor)' if i == 0 else ''
    ax.set_title(f'Unit {i} - Y Position{anchor_text}{quant_suffix}')
    ax.grid(True, alpha=0.3)
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for i in range(n_units, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(f'{parent_dir}/plots/narma{NARMA_ORDER}_individual_unit_y_positions.png', dpi=150, bbox_inches='tight')
print(f"Individual unit Y position plots saved to: {parent_dir}/plots/narma{NARMA_ORDER}_individual_unit_y_positions.png")
plt.close()

#%%
# Plot velocities of units receiving input, with input signal overlay
print("\n=== Creating Velocity Plots for Input-Receiving Units ===")

# Find which units receive linear input (non-zero weights in lin_input_map)
input_receiving_units = torch.nonzero(lin_input_map[0, :]).cpu().numpy().flatten()
print(f"Units receiving input: {input_receiving_units}")
print(f"Input weights: {lin_input_map[0, input_receiving_units].cpu().numpy()}")

if len(input_receiving_units) > 0:
    # Create plot with input signal and velocities
    n_input_units = len(input_receiving_units)
    fig, axes = plt.subplots(n_input_units + 1, 1, figsize=(15, 3*(n_input_units + 1)))
    
    if n_input_units == 0:
        axes = [axes]
    
    # Plot 1: Input signal
    axes[0].plot(time_steps, u_input[:len(time_steps)], color='black', linewidth=2, label='Input signal')
    axes[0].set_ylabel('Input u(t)')
    axes[0].set_title('NARMA Input Signal')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2+: Velocity of each input-receiving unit
    for idx, unit_idx in enumerate(input_receiving_units):
        ax = axes[idx + 1]
        
        # Plot linear velocity (s)
        ax.plot(time_steps, s_states[unit_idx, :], color='C0', linewidth=1.5, label=f'Linear velocity (s)', alpha=0.8)
        ax.set_ylabel('Velocity (m/s)', color='C0')
        ax.tick_params(axis='y', labelcolor='C0')
        ax.grid(True, alpha=0.3)
        
        # Overlay input signal on secondary y-axis
        ax2 = ax.twinx()
        input_weight = lin_input_map[0, unit_idx].item()
        ax2.plot(time_steps, u_input[:len(time_steps)], color='red', linewidth=1.0, 
                 alpha=0.5, linestyle='--', label='Input signal')
        ax2.set_ylabel('Input u(t)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        anchor_text = ' (Anchor)' if unit_idx == 0 else ''
        ax.set_title(f'Unit {unit_idx}{anchor_text} - Linear Velocity (input weight: {input_weight:.3f})')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    axes[-1].set_xlabel('Time Step')
    
    plt.tight_layout()
    plt.savefig(f'{parent_dir}/plots/narma{NARMA_ORDER}_input_unit_velocities.png', dpi=150, bbox_inches='tight')
    print(f"Input unit velocity plots saved to: {parent_dir}/plots/narma{NARMA_ORDER}_input_unit_velocities.png")
    plt.close()
else:
    print("No units receive input (all weights are zero)")

#%%
# Animation: Create state evolution animation
print("\n=== Creating State Evolution Animation ===")

t_steps = len(train_u)
# Subsample for animation (every 10th step to make it manageable)
subsample = 10
t_steps_anim = t_steps // subsample

fig, ax = plt.subplots(figsize=(10, 10))

# Calculate the data ranges
x_range = x_states.max() - x_states.min()
y_range = y_states.max() - y_states.min()

# Set axis limits with a margin
ax.set_xlim(x_states.min() - 0.2, x_states.max() + 0.2)
ax.set_ylim(y_states.min() - 0.2, y_states.max() + 0.2)
ax.set_xlabel('x position')
ax.set_ylabel('y position')
ax.set_title(f'NARMA-{NARMA_ORDER} State Evolution (Unicycle Reservoir)')
ax.set_aspect('equal', adjustable='box')  # Force equal aspect ratio for consistent arrow sizes

# Fixed arrow size (all arrows same length since they represent direction only)
# Adjust the multiplier (0.10) to change arrow size: larger = bigger arrows
arrow_length = 0.10 * min(x_range, y_range).item() if min(x_range, y_range).item() > 0 else 0.10

# Initialize the positions and directions for the arrows
x_pos = x_states[:, 0]
y_pos = y_states[:, 0]
# Scale the unit vectors to fixed length
u_arrow = arrow_length * np.cos(theta_states[:, 0])
v_arrow = arrow_length * np.sin(theta_states[:, 0])

# Color coding for different units
colors = np.linspace(0, 1, n_units)
cmap = plt.get_cmap("hsv")
arrow_colors = cmap(colors)

# Initialize a Quiver object for the arrows
# Use scale=1 and scale_units='xy' to interpret u,v directly as data coordinates
quiver = ax.quiver(x_pos, y_pos, u_arrow, v_arrow, 
                   angles='xy', scale_units='xy', scale=1,
                   width=0.006, headwidth=4, headlength=5,
                   color=arrow_colors)

def update_animation(frame):
    frame_idx = frame * subsample
    if frame_idx >= t_steps:
        frame_idx = t_steps - 1
    
    # Get the current positions
    x_pos = x_states[:, frame_idx]
    y_pos = y_states[:, frame_idx]
    
    # Get the current orientations (scaled to fixed length)
    u_arrow = arrow_length * np.cos(theta_states[:, frame_idx])
    v_arrow = arrow_length * np.sin(theta_states[:, frame_idx])
    
    # Update the quiver
    quiver.set_offsets(np.c_[x_pos, y_pos])
    quiver.set_UVC(u_arrow, v_arrow)
    
    return quiver,

# Create animation
ani = FuncAnimation(fig, update_animation, frames=t_steps_anim, blit=True, interval=50)

# Save as MP4
animation_path_mp4 = f'{parent_dir}/plots/narma{NARMA_ORDER}_state_evolution.mp4'
ani.save(animation_path_mp4)
print(f"MP4 animation saved to: {animation_path_mp4}")

# Save as GIF (more compatible, easier to embed in documents/presentations)
animation_path_gif = f'{parent_dir}/plots/narma{NARMA_ORDER}_state_evolution.gif'
ani.save(animation_path_gif, writer='pillow', fps=20)
print(f"GIF animation saved to: {animation_path_gif}")

plt.close()

print("\n=== Analysis Complete ===")
# %%

#%% Force breakdown visualization function
def plot_force_breakdown(model, u_input, n_steps=500, unit_idx=0, save_path=None):
    """
    Visualize the breakdown of forces acting on a specific unit over time.
    
    Args:
        model: UnicycleReservoir model
        u_input: Input signal (n_samples,)
        n_steps: Number of timesteps to simulate
        unit_idx: Which unit to visualize (default: 0, the anchor)
        save_path: Path to save the plot (optional)
    """
    # Prepare input
    u_torch = torch.from_numpy(u_input[:n_steps] * reservoir_input_scale).float().reshape(1, -1, 1).to(device)
    
    # Run model with force tracking
    print(f"\n=== Computing force breakdown for {n_steps} steps ===")
    with torch.no_grad():
        states_list, _, _, force_history = model.forward_with_forces(u_torch, u_torch)
    
    # Extract forces for the specified unit
    time_steps = np.arange(n_steps)
    input_forces = np.array([f['input_force'][0, unit_idx] for f in force_history])
    spring_forces = np.array([f['spring_force'][0, unit_idx] for f in force_history])
    damping_forces = np.array([f['damping_force'][0, unit_idx] for f in force_history])
    total_forces = np.array([f['total_force'][0, unit_idx] for f in force_history])
    
    input_torques = np.array([f['input_torque'][0, unit_idx] for f in force_history])
    angular_coupling = np.array([f['angular_coupling'][0, unit_idx] for f in force_history])
    angular_damping = np.array([f['angular_damping'][0, unit_idx] for f in force_history])
    total_torques = np.array([f['total_torque'][0, unit_idx] for f in force_history])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Linear forces
    ax = axes[0]
    ax.plot(time_steps, input_forces, label='Input force', linewidth=2, alpha=0.8)
    ax.plot(time_steps, spring_forces, label='Spring force', linewidth=2, alpha=0.8)
    ax.plot(time_steps, damping_forces, label='Damping force', linewidth=2, alpha=0.8)
    ax.plot(time_steps, total_forces, label='Total force', linewidth=2, color='black', linestyle='--')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Force [N]')
    ax.set_title(f'Linear Force Breakdown - Unit {unit_idx}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Angular forces (torques)
    ax = axes[1]
    ax.plot(time_steps, input_torques, label='Input torque', linewidth=2, alpha=0.8)
    ax.plot(time_steps, angular_coupling, label='Angular coupling', linewidth=2, alpha=0.8)
    ax.plot(time_steps, angular_damping, label='Angular damping', linewidth=2, alpha=0.8)
    ax.plot(time_steps, total_torques, label='Total torque', linewidth=2, color='black', linestyle='--')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Torque [N·m]')
    ax.set_title(f'Angular Torque Breakdown - Unit {unit_idx}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Force breakdown plot saved to: {save_path}")
    else:
        plt.savefig(f'{parent_dir}/plots/narma{NARMA_ORDER}_force_breakdown_unit{unit_idx}.png', dpi=150, bbox_inches='tight')
        print(f"Force breakdown plot saved to: {parent_dir}/plots/narma{NARMA_ORDER}_force_breakdown_unit{unit_idx}.png")
    
    plt.close()
    
    # Print statistics
    print(f"\n=== Force Statistics for Unit {unit_idx} ===")
    print(f"Input force:    mean={input_forces.mean():.4f}, std={input_forces.std():.4f}, max={np.abs(input_forces).max():.4f}")
    print(f"Spring force:   mean={spring_forces.mean():.4f}, std={spring_forces.std():.4f}, max={np.abs(spring_forces).max():.4f}")
    print(f"Damping force:  mean={damping_forces.mean():.4f}, std={damping_forces.std():.4f}, max={np.abs(damping_forces).max():.4f}")
    print(f"Total force:    mean={total_forces.mean():.4f}, std={total_forces.std():.4f}, max={np.abs(total_forces).max():.4f}")
    print(f"\nInput torque:   mean={input_torques.mean():.4f}, std={input_torques.std():.4f}, max={np.abs(input_torques).max():.4f}")
    print(f"Angular coupling: mean={angular_coupling.mean():.4f}, std={angular_coupling.std():.4f}, max={np.abs(angular_coupling).max():.4f}")
    print(f"Angular damping: mean={angular_damping.mean():.4f}, std={angular_damping.std():.4f}, max={np.abs(angular_damping).max():.4f}")
    print(f"Total torque:   mean={total_torques.mean():.4f}, std={total_torques.std():.4f}, max={np.abs(total_torques).max():.4f}")

# Example usage - uncomment to generate force breakdown plots
plot_force_breakdown(model, u_input, n_steps=500, unit_idx=0)
plot_force_breakdown(model, u_input, n_steps=500, unit_idx=5)
plot_force_breakdown(model, u_input, n_steps=500, unit_idx=3)  # For a non-anchor unit
  # For a non-anchor unit
# %%
fig,
plt.plot(activations_all[:,0])
# %%
