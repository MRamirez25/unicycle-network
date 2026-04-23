#!/usr/bin/env python3
"""
Animate NARMA test predictions with robot movements.

This script:
1. Loads period 80 robot data
2. Trains NARMA model (configurable order)
3. Creates animation showing:
   - Test predictions over time (top plot)
   - Robot movements in 2D plane (bottom plot)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrow
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from plot_robot_data import load_robot_data, extract_robot_ids, get_robot_states

#=============================================================================
# CONFIGURATION
#=============================================================================

# Data file
DATA_FILE = "time-period-80.csv.gz"

# NARMA configuration
NARMA_ORDER = 3  # NARMA order to visualize (2, 3, 5, etc.)

# Ridge regression
RIDGE_ALPHA = 300
RIDGE_SOLVER = 'auto'

# Data processing
WASHOUT_FRACTION = 0.15
CUTOFF_FRACTION = 0.0
TRAIN_FRACTION = 0.8
N_ROBOTS = 10

# Features
FEATURE_NAMES = ['pos_x', 'pos_y']
USE_LOCAL_FRAME = False

# Animation settings
DT = 0.05  # Time step
FRAME_SKIP = 2  # Show every Nth frame (5 means 5x speedup)
FPS = 60  # Frames per second in saved video
TRAIL_LENGTH = 400  # Number of past positions to show as trail

# Output
OUTPUT_DIR = "animations"
ANIMATION_FILE = f"narma{NARMA_ORDER}_test_animation.mp4"

#=============================================================================
# NARMA generation
#=============================================================================

def generate_narma_from_input(u_normalized, narma_order):
    """Generate NARMA-n time series from normalized input."""
    n_samples = len(u_normalized)
    y_narma = np.zeros(n_samples)
    
    if narma_order == 2:
        for k in range(1, n_samples - 1):
            y_narma[k + 1] = (0.4 * y_narma[k] + 
                            0.4 * y_narma[k] * y_narma[k - 1] + 
                            0.6 * u_normalized[k]**3 + 0.1)
                            
    elif narma_order == 3:
        a, b, c, d = 0.3, 0.05, 1.5, 0.1
        for k in range(2, n_samples - 1):
            y_sum = np.sum(y_narma[k - 2:k + 1])
            y_narma[k + 1] = (a * y_narma[k] + 
                            b * y_narma[k] * y_sum + 
                            c * u_normalized[k - 2] * u_normalized[k] + d)
                            
    else:
        n = narma_order
        a, b, c, d = 0.3, 0.05, 1.5, 0.1
        for k in range(n - 1, n_samples - 1):
            y_sum = np.sum(y_narma[k - (n - 1):k + 1])
            y_narma[k + 1] = (a * y_narma[k] + 
                            b * y_narma[k] * y_sum + 
                            c * u_normalized[k - n + 1] * u_normalized[k] + d)
    
    return y_narma


def extract_robot_features(df, robot_ids, feature_names):
    """Extract feature matrix from robot states."""
    all_features = []
    
    for robot_id in robot_ids:
        states = get_robot_states(df, robot_id, verbose=False)
        robot_feature_list = [states[fname] for fname in feature_names]
        robot_features = np.stack(robot_feature_list, axis=1)
        all_features.append(robot_features)
    
    features = np.concatenate(all_features, axis=1)
    return features


#=============================================================================
# Main animation
#=============================================================================

def main():
    """Main execution."""
    
    print("\n" + "="*80)
    print("NARMA TEST ANIMATION")
    print("="*80)
    print(f"Data file: {DATA_FILE}")
    print(f"NARMA order: {NARMA_ORDER}")
    print(f"Animation: {ANIMATION_FILE}")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load robot data
    print("Loading robot data...")
    df = load_robot_data(DATA_FILE)
    robot_ids = extract_robot_ids(df)
    robot_ids = robot_ids[:N_ROBOTS]
    
    print(f"Found {len(robot_ids)} robots")
    
    # Extract features
    print("Extracting features...")
    features = extract_robot_features(df, robot_ids, FEATURE_NAMES)
    n_samples = features.shape[0]
    
    # Extract global_u and generate NARMA
    print(f"Generating NARMA-{NARMA_ORDER} target...")
    states = get_robot_states(df, robot_ids[0], verbose=False)
    global_u = states['global_u']
    
    u_normalized = global_u - global_u.min()
    u_normalized = 0.5 * u_normalized / u_normalized.max()
    
    y_narma = generate_narma_from_input(u_normalized, NARMA_ORDER)
    
    # Split data
    washout_samples = int(WASHOUT_FRACTION * n_samples)
    cutoff_samples = int(CUTOFF_FRACTION * n_samples)
    end_idx = n_samples - cutoff_samples if cutoff_samples > 0 else n_samples
    valid_idx = np.arange(washout_samples, end_idx)
    n_valid = len(valid_idx)
    n_train = int(TRAIN_FRACTION * n_valid)
    
    train_idx = valid_idx[:n_train]
    test_idx = valid_idx[n_train:]
    
    print(f"Data split: train={len(train_idx)}, test={len(test_idx)}")
    
    # Train model
    print("Training Ridge regression...")
    X_train = features[train_idx, :]
    y_train = y_narma[train_idx]
    X_test = features[test_idx, :]
    y_test = y_narma[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    regressor = Ridge(alpha=RIDGE_ALPHA, solver=RIDGE_SOLVER, fit_intercept=True)
    regressor.fit(X_train_scaled, y_train)
    
    test_pred = regressor.predict(X_test_scaled)
    test_mse = np.mean((test_pred - y_test)**2)
    test_nmse = test_mse / (np.var(y_test) + 1e-9)
    
    print(f"Test NMSE: {test_nmse:.6f}")
    
    # Extract robot positions for test period
    print("Extracting robot positions...")
    robot_positions = []
    robot_orientations = []
    for robot_id in robot_ids:
        robot_states = get_robot_states(df, robot_id, verbose=False)
        pos_x = robot_states['pos_x'][test_idx]
        pos_y = robot_states['pos_y'][test_idx]
        # Extract quaternion and convert to angle
        qz = robot_states['qz'][test_idx]
        qw = robot_states['qw'][test_idx]
        # Convert quaternion to yaw angle
        theta = 2 * np.arctan2(qz, qw)
        robot_positions.append((pos_x, pos_y))
        robot_orientations.append(theta)
    
    # Downsample for animation
    test_frames = np.arange(0, len(test_idx), FRAME_SKIP)
    n_frames = len(test_frames)
    
    print(f"Creating animation with {n_frames} frames...")
    
    # Setup figure (side by side layout)
    fig = plt.figure(figsize=(16, 6))
    
    # Left subplot: NARMA predictions
    ax_narma = plt.subplot(1, 2, 1)
    t_test = test_idx * DT
    
    # Lines that will be drawn progressively (no background)
    line_target, = ax_narma.plot([], [], 'b-', linewidth=2, label='Target', alpha=0.8)
    line_pred, = ax_narma.plot([], [], 'r--', linewidth=2, label='Prediction', alpha=0.8)
    marker_current, = ax_narma.plot([], [], 'ko', markersize=8)
    
    ax_narma.set_xlabel('Time (s)')
    ax_narma.set_ylabel('NARMA Output')
    ax_narma.set_title(f'NARMA-{NARMA_ORDER} Test Predictions (NMSE: {test_nmse:.4f})')
    ax_narma.legend(loc='upper right')
    ax_narma.grid(True, alpha=0.3)
    ax_narma.set_xlim(t_test[0], t_test[-1])
    ax_narma.set_ylim(min(y_test.min(), test_pred.min()) - 0.1, 
                      max(y_test.max(), test_pred.max()) + 0.1)
    
    # Right subplot: Robot positions in 2D
    ax_robots = plt.subplot(1, 2, 2)
    
    # Compute position bounds
    all_x = np.concatenate([pos[0] for pos in robot_positions])
    all_y = np.concatenate([pos[1] for pos in robot_positions])
    x_margin = (all_x.max() - all_x.min()) * 0.1
    y_margin = (all_y.max() - all_y.min()) * 0.05
    
    ax_robots.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
    ax_robots.set_ylim(all_y.min() - y_margin-0.3, all_y.max() + y_margin)
    ax_robots.set_xlabel('X Position (m)')
    ax_robots.set_ylabel('Y Position (m)')
    ax_robots.set_title('Robot Positions in 2D Plane')
    ax_robots.set_aspect('equal', adjustable='box')
    ax_robots.grid(True, alpha=0.3)
    
    # Create robot markers, trails, and orientation arrows
    colors = plt.cm.tab10(np.linspace(0, 1, len(robot_ids)))
    robot_markers = []
    robot_trails = []
    robot_arrows = []
    
    for idx, color in enumerate(colors):
        marker, = ax_robots.plot([], [], 'o', color=color, markersize=12, 
                                label=f'Robot {idx+1}', markeredgecolor='black', 
                                markeredgewidth=1.5)
        trail, = ax_robots.plot([], [], '-', color=color, linewidth=1.5, alpha=0.5)
        # Create quiver for orientation arrow (will be updated in animate)
        arrow = ax_robots.quiver([], [], [], [], color=color, scale=4.5, width=0.007, 
                                headwidth=10, headlength=13, headaxislength=11.5, alpha=0.8)
        robot_markers.append(marker)
        robot_trails.append(trail)
        robot_arrows.append(arrow)
    
    ax_robots.legend(loc='lower center', ncol=5, fontsize=9)
    
    # Time text
    time_text = ax_robots.text(0.02, 0.98, '', transform=ax_robots.transAxes,
                               fontsize=12, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Animation update function
    def update(frame_num):
        """Update animation frame."""
        frame_idx = test_frames[frame_num]
        current_time = t_test[frame_idx]
        
        # Update NARMA plot - draw progressively from start to current frame
        line_target.set_data(t_test[0:frame_idx+1], y_test[0:frame_idx+1])
        line_pred.set_data(t_test[0:frame_idx+1], test_pred[0:frame_idx+1])
        # marker_current.set_data([current_time], [y_test[frame_idx]])
        
        # Update robot positions and orientations
        for robot_idx, (marker, trail, arrow) in enumerate(zip(robot_markers, robot_trails, robot_arrows)):
            pos_x, pos_y = robot_positions[robot_idx]
            theta = robot_orientations[robot_idx]
            
            # Current position
            current_x = pos_x[frame_idx]
            current_y = pos_y[frame_idx]
            # marker.set_data([current_x], [current_y])
            
            # Trail (past positions)
            trail_start = max(0, frame_idx - TRAIL_LENGTH)
            trail.set_data(pos_x[trail_start:frame_idx+1], pos_y[trail_start:frame_idx+1])
            
            # Update orientation arrow
            arrow_length = 0.2  # meters
            dx = arrow_length * np.cos(theta[frame_idx])
            dy = arrow_length * np.sin(theta[frame_idx])
            arrow.set_offsets([[current_x, current_y]])
            arrow.set_UVC(dx, dy)
        
        # Update time text
        time_text.set_text(f'Time: {current_time:.2f} s')
        
        return [line_target, line_pred, marker_current, time_text] + robot_markers + robot_trails + robot_arrows
    
    # Create animation
    print("Rendering animation...")
    anim = animation.FuncAnimation(fig, update, frames=n_frames, 
                                  interval=1000/FPS, blit=True, repeat=True)
    
    # Save animation
    output_path = os.path.join(OUTPUT_DIR, ANIMATION_FILE)
    print(f"Saving animation to {output_path}...")
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=FPS, bitrate=2000)
    anim.save(output_path, writer=writer)
    
    print(f"✓ Animation saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
