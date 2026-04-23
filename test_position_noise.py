"""
Test script demonstrating position noise in UnicycleReservoir.

This script shows how to use the position_noise_std parameter to add
Gaussian noise to the unicycle positions during simulation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from unicycle_network_class import UnicycleReservoir

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters
n_inp = 1
n_units = 5
dt = 0.05
n_out = 1
time_steps = 200

# Create two reservoirs - one without noise, one with noise
reservoir_no_noise = UnicycleReservoir(
    n_inp=n_inp,
    n_units=n_units,
    dt=dt,
    n_out=n_out,
    position_noise_std=0.0  # No noise
)

reservoir_with_noise = UnicycleReservoir(
    n_inp=n_inp,
    n_units=n_units,
    dt=dt,
    n_out=n_out,
    position_noise_std=0.01  # 1 cm standard deviation
)

# Set initial states (grid layout)
bs = 1
reservoir_no_noise.set_init_states_grid(bs, num_rows=1, num_cols=n_units, spacing=(1.0, 1.0))
reservoir_with_noise.set_init_states_grid(bs, num_rows=1, num_cols=n_units, spacing=(1.0, 1.0))

# Generate sinusoidal input signal
t = np.linspace(0, time_steps * dt, time_steps)
u_lin = torch.tensor(0.5 * np.sin(2 * np.pi * 0.5 * t), dtype=torch.float32).reshape(bs, time_steps, n_inp)
u_ang = torch.zeros(bs, time_steps, n_inp)

# Run simulations
print("Running simulation without noise...")
states_no_noise, _, _ = reservoir_no_noise.forward(u_lin, u_ang)

print("Running simulation with noise (std=0.01m)...")
states_with_noise, _, _ = reservoir_with_noise.forward(u_lin, u_ang)

# Extract positions for first robot
x_no_noise = [state[0, 0].item() for state in states_no_noise]
z_no_noise = [state[0, n_units].item() for state in states_no_noise]

x_with_noise = [state[0, 0].item() for state in states_with_noise]
z_with_noise = [state[0, n_units].item() for state in states_with_noise]

# Plot trajectories
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot without noise
axes[0].plot(x_no_noise, z_no_noise, 'b-', linewidth=2, label='Robot 0')
axes[0].plot(x_no_noise[0], z_no_noise[0], 'go', markersize=10, label='Start')
axes[0].plot(x_no_noise[-1], z_no_noise[-1], 'ro', markersize=10, label='End')
axes[0].set_xlabel('X Position (m)', fontsize=12)
axes[0].set_ylabel('Z Position (m)', fontsize=12)
axes[0].set_title('Trajectory WITHOUT Position Noise', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axis('equal')

# Plot with noise
axes[1].plot(x_with_noise, z_with_noise, 'r-', linewidth=2, alpha=0.7, label='Robot 0 (noisy)')
axes[1].plot(x_no_noise, z_no_noise, 'b--', linewidth=1, alpha=0.5, label='Reference (no noise)')
axes[1].plot(x_with_noise[0], z_with_noise[0], 'go', markersize=10, label='Start')
axes[1].plot(x_with_noise[-1], z_with_noise[-1], 'ro', markersize=10, label='End')
axes[1].set_xlabel('X Position (m)', fontsize=12)
axes[1].set_ylabel('Z Position (m)', fontsize=12)
axes[1].set_title('Trajectory WITH Position Noise (σ=0.01m)', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axis('equal')

plt.tight_layout()
plt.savefig('position_noise_comparison.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'position_noise_comparison.png'")

# Compute and display statistics
position_error = np.sqrt((np.array(x_with_noise) - np.array(x_no_noise))**2 + 
                        (np.array(z_with_noise) - np.array(z_no_noise))**2)

print(f"\n{'='*60}")
print(f"Position Noise Statistics:")
print(f"{'='*60}")
print(f"Configured noise std: 0.01 m")
print(f"Mean position error:  {np.mean(position_error):.4f} m")
print(f"Std position error:   {np.std(position_error):.4f} m")
print(f"Max position error:   {np.max(position_error):.4f} m")
print(f"{'='*60}\n")

# Time series comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# X position over time
axes[0].plot(t, x_no_noise, 'b-', linewidth=2, label='Without noise')
axes[0].plot(t, x_with_noise, 'r-', linewidth=1, alpha=0.7, label='With noise (σ=0.01m)')
axes[0].set_xlabel('Time (s)', fontsize=12)
axes[0].set_ylabel('X Position (m)', fontsize=12)
axes[0].set_title('X Position Over Time', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Z position over time
axes[1].plot(t, z_no_noise, 'b-', linewidth=2, label='Without noise')
axes[1].plot(t, z_with_noise, 'r-', linewidth=1, alpha=0.7, label='With noise (σ=0.01m)')
axes[1].set_xlabel('Time (s)', fontsize=12)
axes[1].set_ylabel('Z Position (m)', fontsize=12)
axes[1].set_title('Z Position Over Time', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('position_noise_timeseries.png', dpi=150, bbox_inches='tight')
print("Time series plot saved as 'position_noise_timeseries.png'")

plt.show()

print("\nTest completed successfully!")
