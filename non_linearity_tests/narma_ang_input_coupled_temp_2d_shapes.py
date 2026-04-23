#%%
# ============================================================================
# TEMPORARY TESTING SCRIPT: OLD SHAPE HANDLING (2D ARRAYS)
# This version uses the OLD approach with 2D target arrays to test if
# shape inconsistencies were causing performance issues.
# 
# Changes from working version:
# - y_train_full and y_test_full are 2D: shape (n, 1) instead of (n,)
# - predictions not flattened with .ravel()
# - weights not flattened with .ravel()
# ============================================================================

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
    t = np.arange(n_samples)
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


#%%
# Configuration
NARMA_ORDER = 2
N_SAMPLES = 10000
WASHOUT_FRACTION = 0.1
TRAIN_FRACTION = 0.7

#%%
# Load best parameters from Optuna study
database_name = "unicycle_nets_narma"
study_name = f"narma{NARMA_ORDER}_centered_no_intercept_new_scaling"
storage_name = f"sqlite:///{parent_dir}/optuna_databases/{database_name}.db"
study = optuna.create_study(storage=storage_name, study_name=study_name, direction='minimize', load_if_exists=True)
params = study.best_params
print(f"Loaded best parameters from study: {study_name}")

#%%
# Extract parameters
aligned_orientations = params.get('aligned_orientations', False)
ang_input = params.get('ang_input', True)
ang_connections = params.get('ang_connections', True)

n_units = 20
lin_stiff_min = params['lin_stiff_min']
lin_stiff_max = params['lin_stiff_max']
ang_stiff_min = params['ang_stiff_min']
ang_stiff_max = params['ang_stiff_max']
lin_damping_min = params['lin_damping_min']
lin_damping_max = params['lin_damping_max']
ang_damping_min = params['ang_damping_min']
ang_damping_max = params['ang_damping_max']
dt = params['dt']*0.1
inp_bias = 0
anchor_con_fraction = params['anchor_con_fraction']
magnitude_min = params['magnitude_min']
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
period_ratio = params.get('period_ratio', 2)  # Default to 2 if not in params
reservoir_input_scale = params.get('reservoir_input_scale', 1.0)  # Default to 1.0 if not in params

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
u_raw, u_input, y_narma = generate_narma(N_SAMPLES, narma_order=NARMA_ORDER, period_ratio=period_ratio, amplitude=0.5)

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
#%%
# Create input maps
n_inp = 1
total_elements = n_inp * n_units
num_non_zero = int(total_elements * non_zero_fraction)

lin_input_map = torch.zeros(n_inp, n_units)
flat_indices = torch.randperm(total_elements)[:num_non_zero]
row_indices = flat_indices // n_units
col_indices = flat_indices % n_units
random_values = torch.rand(num_non_zero) * (magnitude_max - magnitude_min) + magnitude_min
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
    inp_bias=inp_bias, ang_input_map=ang_input_map
)

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
model = model.to(device)

#%%
# Initialize model states
batch_size = 1  # Time series processing
model.set_init_states_random(batch_size)
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
activations_all = activations_all[:, :n_units*3]  # Use only position (x, y) and angle (theta) states

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
# Compare reservoir evolution with input vs without input
print("\n=== Comparing Input vs No-Input Evolution ===")
zero_input = torch.zeros_like(full_input).to(device)

# Reset model to same initial state for fair comparison
model.set_init_states(batch_size, x, z, theta, s, omega)

with torch.no_grad():
    states_list_zero, _, _ = model(zero_input, zero_input)

activations_zero = torch.stack(states_list_zero, dim=1)
activations_zero = activations_zero.squeeze(0).cpu().numpy()
activations_zero = activations_zero[:, :n_units*3]

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

X_train = activations_all[train_idx, :]
y_train_full = y_narma[train_idx].reshape(-1, 1)  # OLD: Make 2D array
X_test = activations_all[test_idx, :]
y_test_full = y_narma[test_idx].reshape(-1, 1)  # OLD: Make 2D array

print(f"\n=== Train/Test Split (OLD SHAPE HANDLING - 2D ARRAYS) ===")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train_full.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test_full.shape}")
print(f"Train target stats - mean: {y_train_full.mean():.6f}, std: {y_train_full.std():.6f}")

# Train Ridge regression on reservoir activations
# Standard per-feature scaling (each column independently)
from sklearn.preprocessing import StandardScaler

X_train_features = X_train[:, :3*n_units]
X_test_features = X_test[:, :3*n_units]

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
weights = regressor.coef_  # OLD: Don't flatten, keep as returned (could be 2D)
intercept = regressor.intercept_
# Handle both scalar and array intercepts
intercept_val = intercept[0] if isinstance(intercept, np.ndarray) and intercept.size > 0 else float(intercept)
print(f"Regressor weights shape: {weights.shape}")  # Show shape to see if it's 2D
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
print(f"   magnitude_min: {magnitude_min:.6f}, magnitude_max: {magnitude_max:.6f}")
if ang_input:
    print(f"   magnitude_min_ang: {magnitude_min_ang:.6f}, magnitude_max_ang: {magnitude_max_ang:.6f}")
print(f"   ridge_alpha: {ridge_alpha:.6e}")
print(f"   period_ratio: {period_ratio}")
print(f"   Input map non-zeros: {(lin_input_map != 0).sum()}/{lin_input_map.numel()}")
print(f"   Input map range: [{lin_input_map.min():.6f}, {lin_input_map.max():.6f}]")

# 1. Check correlation between activations and target
from scipy.stats import pearsonr, spearmanr

print("\n1. Individual Feature Correlations with Target:")
correlations = []
for i in range(X_train_scaled.shape[1]):
    if X_train_scaled[:, i].std() > 1e-10:  # Only compute for non-constant features
        corr, p_value = pearsonr(X_train_scaled[:, i], y_train_full.flatten())  # OLD: Need to flatten 2D target
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
plt.figure(figsize=(12, 4))

# Reshape weights for better visualization: [x_weights, y_weights, theta_weights]
weights_flat = weights.ravel()  # Flatten in case it's 2D
weights_reshaped = weights_flat.reshape(3, n_units)

plt.subplot(1, 2, 1)
im = plt.imshow(weights_reshaped, aspect='auto', cmap='RdBu_r', 
                vmin=-np.abs(weights).max(), vmax=np.abs(weights).max())
plt.colorbar(im, label='Weight value')
plt.xlabel('Unicycle index')
plt.ylabel('State type')
plt.yticks([0, 1, 2], ['x', 'y', 'θ'])
plt.title('Ridge Regressor Weights Heatmap')

# Also show as grouped bar plot for better readability
plt.subplot(1, 2, 2)
x_weights = weights_flat[:n_units]
y_weights = weights_flat[n_units:2*n_units]
theta_weights = weights_flat[2*n_units:]
x_pos = np.arange(n_units)
width = 0.25

plt.bar(x_pos - width, x_weights, width, label='x weights', alpha=0.8)
plt.bar(x_pos, y_weights, width, label='y weights', alpha=0.8)
plt.bar(x_pos + width, theta_weights, width, label='θ weights', alpha=0.8)
plt.xlabel('Unicycle index')
plt.ylabel('Weight value')
plt.title('Ridge Regressor Weights by State Type')
plt.legend()
plt.grid(True, alpha=0.3)

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
axes[0, 1].plot(y_train_full[:sample_size].flatten(), label='Target', linewidth=2)  # OLD: Need to flatten 2D
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
train_predictions = regressor.predict(X_train_scaled)  # OLD: Don't flatten
train_std = train_predictions.std()
print(f"Train predictions shape: {train_predictions.shape}")
print(f"Train predictions stats - mean: {train_predictions.mean():.6f}, std: {train_std:.6f}")

if train_std < 1e-10:
    print("WARNING: Regressor predicting constant values, may be over-regularized")
    print(f"         Consider reducing ridge_alpha (currently {ridge_alpha})")

#%%
# Test evaluation
print("\n=== Testing (OLD SHAPE HANDLING) ===")
# Scaler already fitted on X_train_features, just transform test features
# No need to apply manual scaling - scaler handles it
predictions = regressor.predict(X_test_scaled)  # OLD: Don't flatten
print(f"Predictions shape: {predictions.shape}")
print(f"y_test_full shape: {y_test_full.shape}")

# SHAPE BUG: predictions might be (n,) while y_test_full is (n, 1)
# This causes broadcasting issues in metric calculations!
# Let's flatten both to ensure consistent shapes for comparison
predictions_flat = predictions.ravel()
y_test_flat = y_test_full.ravel()
print(f"After flattening - Predictions: {predictions_flat.shape}, Target: {y_test_flat.shape}")

# Calculate metrics (using flattened arrays)
mse = np.mean(np.square(predictions_flat - y_test_flat))
target_var = np.var(y_test_flat)
nmse = mse / (target_var + 1e-9)
rmse = np.sqrt(mse)
target_std = np.std(y_test_flat)
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

# Save individual state plots
fig, axes = plt.subplots(5, 1, figsize=(15, 12))

axes[0].plot(x_states.T)
axes[0].set_ylabel('x states')
axes[0].set_title('Position States (x)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(y_states.T)
axes[1].set_ylabel('y states')
axes[1].set_title('Position States (y)')
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

# Dynamically set scale and width based on the data range
scale_factor = 0.3 * min(x_range, y_range).item() if min(x_range, y_range).item() > 0 else 0.3
width_factor = 0.002 * min(x_range, y_range).item() if min(x_range, y_range).item() > 0 else 0.002

# Initialize the positions and directions for the arrows
x_pos = x_states[:, 0]
y_pos = y_states[:, 0]
u_arrow = np.cos(theta_states[:, 0])
v_arrow = np.sin(theta_states[:, 0])

# Color coding for different units
colors = np.linspace(0, 1, n_units)
cmap = plt.get_cmap("hsv")
arrow_colors = cmap(colors)

# Initialize a Quiver object for the arrows
quiver = ax.quiver(x_pos, y_pos, u_arrow, v_arrow, angles='xy', scale_units='xy', 
                   scale=scale_factor, width=width_factor, color=arrow_colors)

def update_animation(frame):
    frame_idx = frame * subsample
    if frame_idx >= t_steps:
        frame_idx = t_steps - 1
    
    # Get the current positions
    x_pos = x_states[:, frame_idx]
    y_pos = y_states[:, frame_idx]
    
    # Get the current orientations
    u_arrow = np.cos(theta_states[:, frame_idx])
    v_arrow = np.sin(theta_states[:, frame_idx])
    
    # Update the quiver
    quiver.set_offsets(np.c_[x_pos, y_pos])
    quiver.set_UVC(u_arrow, v_arrow)
    
    return quiver,

# Create animation
ani = FuncAnimation(fig, update_animation, frames=t_steps_anim, blit=True, interval=50)
animation_path = f'{parent_dir}/plots/narma{NARMA_ORDER}_state_evolution.mp4'
ani.save(animation_path)
print(f"Animation saved to: {animation_path}")
plt.close()

print("\n=== Analysis Complete ===")
# %%
