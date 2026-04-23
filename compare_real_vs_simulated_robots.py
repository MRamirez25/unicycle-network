#!/usr/bin/env python3
"""
Compare real robot data with simulated unicycle reservoir.

This script:
1. Loads real robot data (positions, velocities, input)
2. Creates a unicycle reservoir with same number of units
3. Initializes simulator with real robot initial states
4. Runs simulation with the same input signal as real robots
5. Compares and visualizes real vs simulated trajectories
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
SCRIPT_DIR = '/home/mariano/phd_code/unicycle-network/'
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from plot_robot_data import load_robot_data, extract_robot_ids, get_robot_states
from unicycle_network_class import UnicycleReservoir


def extract_initial_states(df, robot_ids, start_idx=0):
    """
    Extract initial states (positions, velocities, orientations) from real robot data.
    
    Args:
        df: DataFrame with robot data
        robot_ids: List of robot IDs
        start_idx: Index to extract initial states from
    
    Returns:
        initial_states: Dictionary with initial state tensors
    """
    n_robots = len(robot_ids)
    
    # Initialize arrays
    x_positions = np.zeros(n_robots)
    y_positions = np.zeros(n_robots)
    linear_velocities = np.zeros(n_robots)
    angular_velocities = np.zeros(n_robots)
    orientations = np.zeros(n_robots)  # Will compute from quaternions
    
    print(f"\nExtracting initial states at index {start_idx}:")
    print(f"{'Robot':<10} {'x':<10} {'y':<10} {'v':<10} {'ω':<10} {'θ':<10}")
    print("-" * 60)
    
    for i, robot_id in enumerate(robot_ids):
        states = get_robot_states(df, robot_id)
        
        # Get initial states
        x_positions[i] = states['pos_x'][start_idx]
        y_positions[i] = states['pos_y'][start_idx]
        linear_velocities[i] = states['linear_x'][start_idx]
        angular_velocities[i] = states['omega'][start_idx]
        
        # Compute orientation from quaternion (qz, qw)
        # For 2D: theta = 2 * atan2(qz, qw)
        qz = states['qz'][start_idx]
        qw = states['qw'][start_idx]
        theta = 2 * np.arctan2(qz, qw)
        orientations[i] = theta
        
        robot_label = robot_id[:8] if len(robot_id) > 8 else robot_id
        print(f"{robot_label:<10} {x_positions[i]:<10.4f} {y_positions[i]:<10.4f} "
              f"{linear_velocities[i]:<10.4f} {angular_velocities[i]:<10.4f} {theta:<10.4f}")
    
    # Convert to tensors
    initial_states = {
        'x_positions': torch.tensor(x_positions, dtype=torch.float32),
        'y_positions': torch.tensor(y_positions, dtype=torch.float32),
        'linear_velocities': torch.tensor(linear_velocities, dtype=torch.float32),
        'angular_velocities': torch.tensor(angular_velocities, dtype=torch.float32),
        'orientations': torch.tensor(orientations, dtype=torch.float32)
    }
    
    return initial_states


def extract_input_signal(df, robot_ids, start_idx=0, end_idx=None):
    """
    Extract global input signal from robot data.
    
    Args:
        df: DataFrame with robot data
        robot_ids: List of robot IDs
        start_idx: Start index
        end_idx: End index
    
    Returns:
        input_signal: numpy array of input values
        time_vector: numpy array of time values
    """
    # Get states from first robot (global_u should be same for all)
    states = get_robot_states(df, robot_ids[0])
    
    if states['global_u'] is None:
        raise ValueError("No global_u found in robot data!")
    
    # Extract input signal
    input_signal = states['global_u'][start_idx:end_idx]
    time_vector = states['t'][start_idx:end_idx]
    
    print(f"\nInput signal extracted:")
    print(f"  Length: {len(input_signal)} timesteps")
    print(f"  Range: [{input_signal.min():.6f}, {input_signal.max():.6f}]")
    print(f"  Mean: {input_signal.mean():.6f}, Std: {input_signal.std():.6f}")
    print(f"  Time range: [{time_vector[0]:.3f}, {time_vector[-1]:.3f}] s")
    
    return input_signal, time_vector


def run_simulation(reservoir, input_signal, initial_states, dt=0.01, device='cpu', 
                   use_initial_distances=False):
    """
    Run unicycle reservoir simulation with given input and initial states.
    
    Args:
        reservoir: UnicycleReservoir instance
        input_signal: Array of input values
        initial_states: Dictionary with initial state tensors
        dt: Time step
        device: torch device (unused, kept for backwards compatibility)
        use_initial_distances: If True, set spring equilibrium distances from initial positions
    
    Returns:
        sim_states: Dictionary with simulated trajectories
    """
    n_steps = len(input_signal)
    n_units = len(initial_states['x_positions'])
    
    # Set initial states using the correct API
    # Note: x_positions is actually x, y_positions is actually z in the unicycle frame
    reservoir.set_init_states(
        bs=1,
        x=initial_states['x_positions'].numpy(),
        z=initial_states['y_positions'].numpy(),
        theta=initial_states['orientations'].numpy(),
        s=initial_states['linear_velocities'].numpy(),
        omega=initial_states['angular_velocities'].numpy()
    )
    
    # Optionally set equilibrium distances based on initial positions
    if use_initial_distances:
        print("\nSetting spring equilibrium distances from initial robot positions...")
        reservoir.set_eq_distances_from_initial_positions()
    
    # Prepare input signals
    # u_lin shape: (batch_size, time_steps, n_inp)
    u_lin = torch.tensor(input_signal, dtype=torch.float32).reshape(1, n_steps, 1)
    u_ang = torch.zeros(1, n_steps, 1)
    
    print(f"\nRunning simulation for {n_steps} timesteps...")
    
    # Run simulation using forward method
    states_list, _, _ = reservoir.forward(u_lin, u_ang)
    
    # Extract states from states_list
    # states_list is a list of tensors with shape (batch_size, n_units*5)
    # where the 5 state variables are: [x, z, theta, s, omega]
    x_traj = np.zeros((n_steps, n_units))
    y_traj = np.zeros((n_steps, n_units))
    v_traj = np.zeros((n_steps, n_units))
    omega_traj = np.zeros((n_steps, n_units))
    theta_traj = np.zeros((n_steps, n_units))
    
    for step in range(n_steps):
        state = states_list[step][0]  # Get batch 0
        x_traj[step, :] = state[:n_units].detach().cpu().numpy()
        y_traj[step, :] = state[n_units:2*n_units].detach().cpu().numpy()
        theta_traj[step, :] = state[2*n_units:3*n_units].detach().cpu().numpy()
        v_traj[step, :] = state[3*n_units:4*n_units].detach().cpu().numpy()
        omega_traj[step, :] = state[4*n_units:5*n_units].detach().cpu().numpy()
    
    sim_states = {
        'pos_x': x_traj,
        'pos_y': y_traj,
        'linear_x': v_traj,
        'omega': omega_traj,
        'theta': theta_traj
    }
    
    print(f"Simulation complete!")
    
    return sim_states


def plot_comparison(real_states_list, sim_states, robot_ids, time_vector, 
                   input_signal, save_dir, prefix="comparison"):
    """
    Create comparison plots of real vs simulated robot states.
    
    Args:
        real_states_list: List of dictionaries with real robot states
        sim_states: Dictionary with simulated states
        robot_ids: List of robot IDs
        time_vector: Time array
        input_signal: Input signal array
        save_dir: Directory to save plots
        prefix: Prefix for saved files
    """
    n_robots = len(robot_ids)
    
    # Plot 1: Positions (x, y) for all robots
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_robots))
    
    # X positions
    for i, (robot_id, real_states) in enumerate(zip(robot_ids, real_states_list)):
        robot_label = robot_id[:8] if len(robot_id) > 8 else robot_id
        axes[0].plot(time_vector, real_states['pos_x'], 
                    label=f'Real {robot_label}', color=colors[i], linestyle='-', alpha=0.7)
        axes[0].plot(time_vector, sim_states['pos_x'][:, i], 
                    label=f'Sim {robot_label}', color=colors[i], linestyle='--', alpha=0.7)
    
    axes[0].set_ylabel('X Position (m)', fontsize=12)
    axes[0].set_title('X Position: Real vs Simulated', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    if n_robots <= 5:
        axes[0].legend(loc='upper right', fontsize=8, ncol=2)
    
    # Y positions
    for i, (robot_id, real_states) in enumerate(zip(robot_ids, real_states_list)):
        robot_label = robot_id[:8] if len(robot_id) > 8 else robot_id
        axes[1].plot(time_vector, real_states['pos_y'], 
                    label=f'Real {robot_label}', color=colors[i], linestyle='-', alpha=0.7)
        axes[1].plot(time_vector, sim_states['pos_y'][:, i], 
                    label=f'Sim {robot_label}', color=colors[i], linestyle='--', alpha=0.7)
    
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Y Position (m)', fontsize=12)
    axes[1].set_title('Y Position: Real vs Simulated', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    if n_robots <= 5:
        axes[1].legend(loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_positions.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {prefix}_positions.png")
    plt.close()
    
    # Plot 2: Velocities (linear and angular)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Linear velocities
    for i, (robot_id, real_states) in enumerate(zip(robot_ids, real_states_list)):
        robot_label = robot_id[:8] if len(robot_id) > 8 else robot_id
        axes[0].plot(time_vector, real_states['linear_x'], 
                    label=f'Real {robot_label}', color=colors[i], linestyle='-', alpha=0.7)
        axes[0].plot(time_vector, sim_states['linear_x'][:, i], 
                    label=f'Sim {robot_label}', color=colors[i], linestyle='--', alpha=0.7)
    
    axes[0].set_ylabel('Linear Velocity (m/s)', fontsize=12)
    axes[0].set_title('Linear Velocity: Real vs Simulated', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    if n_robots <= 5:
        axes[0].legend(loc='upper right', fontsize=8, ncol=2)
    
    # Angular velocities
    for i, (robot_id, real_states) in enumerate(zip(robot_ids, real_states_list)):
        robot_label = robot_id[:8] if len(robot_id) > 8 else robot_id
        axes[1].plot(time_vector, real_states['omega'], 
                    label=f'Real {robot_label}', color=colors[i], linestyle='-', alpha=0.7)
        axes[1].plot(time_vector, sim_states['omega'][:, i], 
                    label=f'Sim {robot_label}', color=colors[i], linestyle='--', alpha=0.7)
    
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    axes[1].set_title('Angular Velocity: Real vs Simulated', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    if n_robots <= 5:
        axes[1].legend(loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_velocities.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {prefix}_velocities.png")
    plt.close()
    
    # Plot 3: 2D trajectories
    fig, ax = plt.subplots(figsize=(12, 12))
    
    for i, (robot_id, real_states) in enumerate(zip(robot_ids, real_states_list)):
        robot_label = robot_id[:8] if len(robot_id) > 8 else robot_id
        ax.plot(real_states['pos_x'], real_states['pos_y'], 
               label=f'Real {robot_label}', color=colors[i], linestyle='-', alpha=0.7, linewidth=2)
        ax.plot(sim_states['pos_x'][:, i], sim_states['pos_y'][:, i], 
               label=f'Sim {robot_label}', color=colors[i], linestyle='--', alpha=0.7, linewidth=2)
        
        # Mark start and end points
        ax.plot(real_states['pos_x'][0], real_states['pos_y'][0], 
               'o', color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=1)
        ax.plot(sim_states['pos_x'][0, i], sim_states['pos_y'][0, i], 
               's', color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('2D Trajectories: Real vs Simulated\n(○ = real start, □ = sim start)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    if n_robots <= 5:
        ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_trajectories_2d.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {prefix}_trajectories_2d.png")
    plt.close()
    
    # Plot 4: Input signal
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(time_vector, input_signal, color='black', linewidth=2)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Input Signal', fontsize=12)
    ax.set_title('Input Signal Used for Both Real and Simulated Systems', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_input.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {prefix}_input.png")
    plt.close()


def compute_errors(real_states_list, sim_states, robot_ids):
    """
    Compute error metrics between real and simulated states.
    
    Args:
        real_states_list: List of dictionaries with real robot states
        sim_states: Dictionary with simulated states
        robot_ids: List of robot IDs
    
    Returns:
        errors: Dictionary with error metrics
    """
    n_robots = len(robot_ids)
    
    print(f"\n{'='*70}")
    print("ERROR METRICS: Real vs Simulated")
    print(f"{'='*70}")
    print(f"{'Robot':<10} {'RMSE_x':<12} {'RMSE_y':<12} {'RMSE_v':<12} {'RMSE_ω':<12}")
    print("-"*70)
    
    errors = {
        'rmse_x': [],
        'rmse_y': [],
        'rmse_v': [],
        'rmse_omega': []
    }
    
    for i, (robot_id, real_states) in enumerate(zip(robot_ids, real_states_list)):
        robot_label = robot_id[:8] if len(robot_id) > 8 else robot_id
        
        # Compute RMSE for each state variable
        rmse_x = np.sqrt(np.mean((real_states['pos_x'] - sim_states['pos_x'][:, i])**2))
        rmse_y = np.sqrt(np.mean((real_states['pos_y'] - sim_states['pos_y'][:, i])**2))
        rmse_v = np.sqrt(np.mean((real_states['linear_x'] - sim_states['linear_x'][:, i])**2))
        rmse_omega = np.sqrt(np.mean((real_states['omega'] - sim_states['omega'][:, i])**2))
        
        errors['rmse_x'].append(rmse_x)
        errors['rmse_y'].append(rmse_y)
        errors['rmse_v'].append(rmse_v)
        errors['rmse_omega'].append(rmse_omega)
        
        print(f"{robot_label:<10} {rmse_x:<12.6f} {rmse_y:<12.6f} {rmse_v:<12.6f} {rmse_omega:<12.6f}")
    
    # Print averages
    print("-"*70)
    print(f"{'AVERAGE':<10} {np.mean(errors['rmse_x']):<12.6f} {np.mean(errors['rmse_y']):<12.6f} "
          f"{np.mean(errors['rmse_v']):<12.6f} {np.mean(errors['rmse_omega']):<12.6f}")
    print(f"{'='*70}\n")
    
    return errors


def main(data_file, start_idx=1000, n_steps=5000, save_dir=None,
         spring_stiffness=1.0, damping=0.1, dt=0.01, 
         lin_input_fraction=0.3, lin_input_magnitude=1.0,
         exclude_robots=None, position_noise_std=0.0,
         use_initial_distances=False):
    """
    Main function to compare real robot data with simulated unicycle reservoir.
    
    Args:
        data_file: Path to robot data CSV file
        start_idx: Starting index in robot data
        n_steps: Number of steps to simulate
        save_dir: Directory to save plots
        spring_stiffness: Spring stiffness for unicycle connections
        damping: Damping coefficient
        dt: Time step
        lin_input_fraction: Fraction of robots receiving linear input
        lin_input_magnitude: Magnitude of linear input
        exclude_robots: List of robot IDs to exclude
        position_noise_std: Standard deviation of Gaussian noise added to positions (m)
        use_initial_distances: If True, set spring equilibrium distances from initial robot positions
    """
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, 'real_vs_sim_comparison')
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"{'='*70}")
    print("REAL vs SIMULATED ROBOT COMPARISON")
    print(f"{'='*70}")
    
    # Load robot data
    print(f"\nLoading robot data from: {data_file}")
    df = load_robot_data(data_file)
    print(f"Data loaded: {len(df)} timesteps")
    
    # Extract robot IDs
    robot_ids = extract_robot_ids(df)
    
    # Filter excluded robots
    if exclude_robots:
        robot_ids = [rid for rid in robot_ids if rid not in exclude_robots]
    
    n_robots = len(robot_ids)
    print(f"Number of robots: {n_robots}")
    print(f"Robot IDs: {[rid[:8] for rid in robot_ids]}")
    
    # Determine end index
    end_idx = start_idx + n_steps
    if end_idx > len(df):
        end_idx = len(df)
        n_steps = end_idx - start_idx
        print(f"Adjusted n_steps to {n_steps} (reached end of data)")
    
    # Extract initial states
    initial_states = extract_initial_states(df, robot_ids, start_idx=start_idx)
    
    # Extract input signal
    input_signal, time_vector = extract_input_signal(df, robot_ids, 
                                                     start_idx=start_idx, 
                                                     end_idx=end_idx)
    
    # Extract real robot trajectories
    print(f"\nExtracting real robot trajectories...")
    real_states_list = []
    for robot_id in robot_ids:
        states = get_robot_states(df, robot_id)
        real_states = {
            'pos_x': states['pos_x'][start_idx:end_idx],
            'pos_y': states['pos_y'][start_idx:end_idx],
            'linear_x': states['linear_x'][start_idx:end_idx],
            'omega': states['omega'][start_idx:end_idx]
        }
        real_states_list.append(real_states)
    
    # Create unicycle reservoir
    print(f"\nCreating unicycle reservoir with {n_robots} units...")
    print(f"  Spring stiffness: {spring_stiffness}")
    print(f"  Damping: {damping}")
    print(f"  dt: {dt}")
    if position_noise_std > 0:
        print(f"  Position noise std: {position_noise_std} m")
    
    # Create input map (linear input only for simplicity)
    n_inp = 1
    lin_input_map = torch.zeros(n_inp, n_robots)
    n_inputs = int(lin_input_fraction * n_robots)
    input_indices = torch.randperm(n_robots)[:n_inputs]
    lin_input_map[0, input_indices] = lin_input_magnitude
    
    print(f"  Linear input: {n_inputs} robots receiving input (magnitude: {lin_input_magnitude})")
    
    # No angular input or connections for simplicity (can be added later)
    ang_input_map = torch.zeros(n_inp, n_robots)
    
    # Create reservoir with correct API
    reservoir = UnicycleReservoir(
        n_inp=n_inp,
        n_units=n_robots,
        dt=dt,
        n_out=1,  # Not used for forward simulation
        lin_stiff_min=spring_stiffness,
        lin_stiff_max=spring_stiffness,
        lin_damping_min=damping,
        lin_damping_max=damping,
        lin_input_map=lin_input_map,
        ang_input_map=ang_input_map,
        n_connections=int(0.5 * n_robots),  # ~50% connectivity
        n_connections_anchor=0,  # No anchor for now
        n_connections_ang=0,  # No angular connections
        n_connections_anchor_ang=0,
        position_noise_std=position_noise_std
    )
    
    # Run simulation
    sim_states = run_simulation(reservoir, input_signal, initial_states, dt=dt, 
                               use_initial_distances=use_initial_distances)
    
    # Compute errors
    errors = compute_errors(real_states_list, sim_states, robot_ids)
    
    # Plot comparisons
    print(f"\nGenerating comparison plots...")
    plot_comparison(real_states_list, sim_states, robot_ids, time_vector, 
                   input_signal, save_dir, prefix="real_vs_sim")
    
    print(f"\n{'='*70}")
    print(f"COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"All plots saved to: {save_dir}")
    print(f"{'='*70}\n")
    
    return real_states_list, sim_states, errors


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare real vs simulated robot dynamics')
    parser.add_argument('--data', type=str, 
                       default='/home/mariano/phd_code/unicycle-network/reference10.csv.gz',
                       help='Path to robot data CSV file')
    parser.add_argument('--start-idx', type=int, default=100,
                       help='Starting index in robot data')
    parser.add_argument('--n-steps', type=int, default=5000,
                       help='Number of simulation steps')
    parser.add_argument('--spring', type=float, default=1.0,
                       help='Spring stiffness')
    parser.add_argument('--damping', type=float, default=0.1,
                       help='Damping coefficient')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step')
    parser.add_argument('--position-noise-std', type=float, default=0.0,
                       help='Standard deviation of Gaussian noise added to positions (m)')
    parser.add_argument('--use-initial-distances', action='store_true',
                       help='Set spring equilibrium distances from initial robot positions')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    real_states, sim_states, errors = main(
        data_file=args.data,
        start_idx=args.start_idx,
        n_steps=args.n_steps,
        save_dir=args.save_dir,
        spring_stiffness=args.spring,
        damping=args.damping,
        dt=args.dt,
        position_noise_std=args.position_noise_std,
        use_initial_distances=args.use_initial_distances
    )
    
    print("\n✓ Comparison complete!")
