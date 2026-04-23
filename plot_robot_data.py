#!/usr/bin/env python3
"""
Load and visualize robot state data from robot_state_log.csv

This script loads the compressed CSV file containing robot trajectories
and provides functions to plot individual robot states or comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import os

# Path to the data file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "robot_state_log.csv")


def load_robot_data(filepath=DATA_FILE):
    """
    Load robot state data from compressed CSV file.
    
    Args:
        filepath: Path to the robot_state_log.csv file (gzipped)
    
    Returns:
        pandas DataFrame with all robot states
    """
    # Check if file is gzipped
    with open(filepath, 'rb') as f:
        is_gzipped = f.read(2) == b'\x1f\x8b'
    
    if is_gzipped:
        print(f"Loading gzipped data from {filepath}...")
        with gzip.open(filepath, 'rt') as f:
            df = pd.read_csv(f)
    else:
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
    
    print(f"Loaded {len(df)} timesteps")
    return df


def extract_robot_ids(df):
    """
    Extract unique robot IDs from column names.
    Supports multiple formats:
    1. 'robot_XXXXX_linear_x' format
    2. 'r##_linear_x' or 'r##_pos_x' format (where ## is a number)
    3. 'r##_linear_x_measured' or 'r##_linear_x_command' format
    
    Args:
        df: DataFrame with robot data
    
    Returns:
        List of unique robot IDs
    """
    robot_ids = []
    
    # Try format 1: robot_XXXXX_linear_x
    for col in df.columns:
        if col.startswith('robot_') and '_linear_x' in col:
            # Extract ID from 'robot_XXXXX_linear_x'
            robot_id = col.replace('robot_', '').replace('_linear_x', '')
            robot_ids.append(robot_id)
    
    # If no robots found, try format 2/3: r##_state format
    if len(robot_ids) == 0:
        import re
        seen_ids = set()
        for col in df.columns:
            # Match patterns like 'r01_linear_x', 'r02_pos_x', 'r20_linear_x_measured' etc.
            # Must have underscore and a state name after the ID
            match = re.match(r'^(r\d+)_', col)
            if match:
                robot_id = match.group(1)
                # Verify this robot has the expected state columns
                if robot_id not in seen_ids:
                    # Check if at least one state column exists for this robot
                    has_states = (f'{robot_id}_pos_x' in df.columns or 
                                 f'{robot_id}_linear_x' in df.columns or
                                 f'{robot_id}_linear_x_measured' in df.columns or
                                 f'{robot_id}_linear_x_command' in df.columns)
                    if has_states:
                        seen_ids.add(robot_id)
                        robot_ids.append(robot_id)
    
    return sorted(robot_ids)  # Sort for consistent ordering


def get_robot_states(df, robot_id, verbose=False):
    """
    Extract all state variables for a specific robot.
    Supports multiple column naming formats:
    1. 'robot_XXXXX_state' format
    2. 'r##_state' format (where ## is a number)
    3. 'r##_state_measured' or 'r##_state_command' format
    
    Priority order when multiple formats exist:
    - _measured columns are preferred over _command columns
    
    Args:
        df: DataFrame with robot data
        robot_id: Robot identifier (e.g., 'bf5fc8b9' or 'r01')
        verbose: If True, print which column format is being used
    
    Returns:
        Dictionary with numpy arrays for each state variable:
        - t: time
        - linear_x: linear velocity
        - pos_x: x position
        - pos_y: y position
        - qz: quaternion z component
        - qw: quaternion w component
        - omega: angular velocity
    """
    # Try format 1: robot_XXXXX_state
    if f'robot_{robot_id}_linear_x' in df.columns:
        prefix = f'robot_{robot_id}_'
        suffix = ''
        format_type = 'robot_XXXXX_state'
    # Try format 2: r##_state
    elif f'{robot_id}_linear_x' in df.columns:
        prefix = f'{robot_id}_'
        suffix = ''
        format_type = 'r##_state'
    # Try format 3: r##_state_measured (PREFERRED)
    elif f'{robot_id}_linear_x_measured' in df.columns:
        prefix = f'{robot_id}_'
        suffix = '_measured'
        format_type = 'r##_state_measured (MEASURED data)'
    # Try format 4: r##_state_command (FALLBACK)
    elif f'{robot_id}_linear_x_command' in df.columns:
        prefix = f'{robot_id}_'
        suffix = '_command'
        format_type = 'r##_state_command (COMMAND data)'
    else:
        raise ValueError(f"Could not find columns for robot '{robot_id}'. "
                        f"Available columns: {list(df.columns)[:20]}")
    
    if verbose:
        print(f"Robot '{robot_id}': Using column format '{format_type}' (e.g., '{prefix}linear_x{suffix}')")
    
    return {
        't': df['t'].values if 't' in df.columns else df.index.values,
        'linear_x': df[f'{prefix}linear_x{suffix}'].values,
        'pos_x': df[f'{prefix}pos_x'].values,
        'pos_y': df[f'{prefix}pos_y'].values,
        'qz': df[f'{prefix}qz'].values,
        'qw': df[f'{prefix}qw'].values,
        'omega': df[f'{prefix}omega'].values,
        'global_u': df['global_u'].values if 'global_u' in df.columns else None,
    }


def plot_robot_trajectory(df, robot_id, ax=None, **kwargs):
    """
    Plot the 2D trajectory (x-y path) of a robot.
    
    Args:
        df: DataFrame with robot data
        robot_id: Robot identifier
        ax: Matplotlib axis (creates new if None)
        **kwargs: Additional arguments for plt.plot
    
    Returns:
        Matplotlib axis
    """
    states = get_robot_states(df, robot_id)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(states['pos_x'], states['pos_y'], label=f'Robot {robot_id[:8]}', **kwargs)
    ax.scatter(states['pos_x'][0], states['pos_y'][0], marker='o', s=100, 
               label='Start', zorder=5)
    ax.scatter(states['pos_x'][-1], states['pos_y'][-1], marker='s', s=100,
               label='End', zorder=5)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Robot Trajectory: {robot_id[:8]}')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    
    return ax


def plot_robot_states_timeseries(df, robot_id, figsize=(14, 10)):
    """
    Plot all state variables over time for a robot.
    
    Args:
        df: DataFrame with robot data
        robot_id: Robot identifier
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    states = get_robot_states(df, robot_id)
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f'Robot States Over Time: {robot_id[:8]}', fontsize=14)
    
    # Position X
    axes[0, 0].plot(states['t'], states['pos_x'])
    axes[0, 0].set_ylabel('X Position')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Position Y
    axes[0, 1].plot(states['t'], states['pos_y'])
    axes[0, 1].set_ylabel('Y Position')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Linear velocity
    axes[1, 0].plot(states['t'], states['linear_x'])
    axes[1, 0].set_ylabel('Linear Velocity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Angular velocity
    axes[1, 1].plot(states['t'], states['omega'])
    axes[1, 1].set_ylabel('Angular Velocity (ω)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Quaternion components
    axes[2, 0].plot(states['t'], states['qz'], label='qz')
    axes[2, 0].plot(states['t'], states['qw'], label='qw')
    axes[2, 0].set_ylabel('Quaternion')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Global input (if available)
    if states['global_u'] is not None:
        axes[2, 1].plot(states['t'], states['global_u'])
        axes[2, 1].set_ylabel('Global Input (u)')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].grid(True, alpha=0.3)
    else:
        axes[2, 1].text(0.5, 0.5, 'No global input data', 
                       ha='center', va='center', transform=axes[2, 1].transAxes)
    
    plt.tight_layout()
    return fig


def plot_all_trajectories(df, figsize=(12, 12)):
    """
    Plot trajectories of all robots on the same plot.
    
    Args:
        df: DataFrame with robot data
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    robot_ids = extract_robot_ids(df)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(robot_ids)))
    
    for i, robot_id in enumerate(robot_ids):
        states = get_robot_states(df, robot_id)
        ax.plot(states['pos_x'], states['pos_y'], 
               label=f'Robot {robot_id[:8]}', color=colors[i], alpha=0.7)
        ax.scatter(states['pos_x'][0], states['pos_y'][0], 
                  marker='o', s=50, color=colors[i], zorder=5)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'All Robot Trajectories ({len(robot_ids)} robots)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig


def plot_all_robots_timeseries(df, figsize=(16, 12)):
    """
    Plot all state variables over time for all robots on the same plots.
    
    Args:
        df: DataFrame with robot data
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    robot_ids = extract_robot_ids(df)
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f'All Robot States Over Time ({len(robot_ids)} robots)', fontsize=14)
    
    # Get colors for all robots
    colors = plt.cm.tab10(np.linspace(0, 1, len(robot_ids)))
    
    # Plot each robot
    for i, robot_id in enumerate(robot_ids):
        states = get_robot_states(df, robot_id)
        label = f'Robot {robot_id[:8]}'
        color = colors[i]
        
        # Position X
        axes[0, 0].plot(states['t'], states['pos_x'], 
                       label=label, color=color, alpha=0.7)
        
        # Position Y
        axes[0, 1].plot(states['t'], states['pos_y'], 
                       label=label, color=color, alpha=0.7)
        
        # Linear velocity
        axes[1, 0].plot(states['t'], states['linear_x'], 
                       label=label, color=color, alpha=0.7)
        
        # Angular velocity
        axes[1, 1].plot(states['t'], states['omega'], 
                       label=label, color=color, alpha=0.7)
        
        # Quaternion qz
        axes[2, 0].plot(states['t'], states['qz'], 
                       label=label, color=color, alpha=0.7)
        
        # Quaternion qw
        axes[2, 1].plot(states['t'], states['qw'], 
                       label=label, color=color, alpha=0.7)
    
    # Configure axes
    axes[0, 0].set_ylabel('X Position')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    
    axes[0, 1].set_ylabel('Y Position')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_ylabel('Linear Velocity')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_ylabel('Angular Velocity (ω)')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 0].set_ylabel('Quaternion qz')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].set_ylabel('Quaternion qw')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def export_robot_data_as_numpy(df, robot_id):
    """
    Export a single robot's data as structured numpy array.
    
    Args:
        df: DataFrame with robot data
        robot_id: Robot identifier
    
    Returns:
        Structured numpy array with fields: t, linear_x, pos_x, pos_y, qz, qw, omega
    """
    states = get_robot_states(df, robot_id)
    
    # Create structured array
    dtype = [('t', 'f8'), ('linear_x', 'f8'), ('pos_x', 'f8'), ('pos_y', 'f8'),
             ('qz', 'f8'), ('qw', 'f8'), ('omega', 'f8')]
    
    n_timesteps = len(states['t'])
    data = np.zeros(n_timesteps, dtype=dtype)
    
    for key in ['t', 'linear_x', 'pos_x', 'pos_y', 'qz', 'qw', 'omega']:
        data[key] = states[key]
    
    return data


def export_all_robots_as_numpy(df):
    """
    Export all robots' data as a dictionary of numpy arrays.
    
    Args:
        df: DataFrame with robot data
    
    Returns:
        Dictionary mapping robot_id -> structured numpy array
    """
    robot_ids = extract_robot_ids(df)
    
    robot_data = {}
    for robot_id in robot_ids:
        robot_data[robot_id] = export_robot_data_as_numpy(df, robot_id)
    
    return robot_data


# Example usage
if __name__ == "__main__":
    # Load data
    df = load_robot_data()
    
    # Get robot IDs
    robot_ids = extract_robot_ids(df)
    print(f"\nFound {len(robot_ids)} robots:")
    for robot_id in robot_ids:
        print(f"  - {robot_id}")
    
    # Plot first robot's states
    if len(robot_ids) > 0:
        print(f"\nPlotting data for robot: {robot_ids[0]}")
        
        # Trajectory
        fig1 = plt.figure(figsize=(8, 8))
        plot_robot_trajectory(df, robot_ids[0])
        plt.savefig('robot_trajectory.png', dpi=150, bbox_inches='tight')
        print("Saved: robot_trajectory.png")
        plt.close()
        
        # Timeseries
        fig2 = plot_robot_states_timeseries(df, robot_ids[0])
        plt.savefig('robot_states_timeseries.png', dpi=150, bbox_inches='tight')
        print("Saved: robot_states_timeseries.png")
        plt.close()
        
        # All trajectories
        fig3 = plot_all_trajectories(df)
        plt.savefig('all_robot_trajectories.png', dpi=150, bbox_inches='tight')
        print("Saved: all_robot_trajectories.png")
        plt.close()
        
        # All robots timeseries
        fig4 = plot_all_robots_timeseries(df)
        plt.savefig('all_robots_timeseries.png', dpi=150, bbox_inches='tight')
        print("Saved: all_robots_timeseries.png")
        plt.close()
        
        # Export as numpy
        robot_data = export_all_robots_as_numpy(df)
        print(f"\nExported {len(robot_data)} robots as numpy arrays")
        print(f"Example - Robot {robot_ids[0]} data shape: {robot_data[robot_ids[0]].shape}")
        print(f"Available fields: {robot_data[robot_ids[0]].dtype.names}")
