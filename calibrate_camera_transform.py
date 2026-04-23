#!/usr/bin/env python3
"""
Camera calibration tool for mapping physical coordinates to pixel coordinates.

This script:
1. Loads the first frame from a video
2. Lets you click on robot positions in the image
3. Matches clicks with physical coordinates from robot data
4. Calculates transformation (affine or perspective) from physical to pixel coordinates
5. Saves the transformation matrix for later use

Usage:
1. Run the script
2. Click on robots in the video frame (in order)
3. Press 'q' when done
4. The transformation will be calculated and saved
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pickle

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from plot_robot_data import load_robot_data, extract_robot_ids, get_robot_states

#=============================================================================
# CONFIGURATION
#=============================================================================

# Video file (actual camera recording of robots)
VIDEO_FILE = "video_edits/tune-period-80-retake4(1).mkv"

# Robot data file
DATA_FILE = "period80-retakes/time-period-80-retake4.csv.gz"

# Which video frame to display for clicking (0 = first frame)
VIDEO_FRAME_NUMBER = 0

# Which data frame to use for physical coordinates (can be later to ensure robots are visible)
DATA_FRAME_NUMBER = 30

# Time step in data (seconds)
DT = 0.05

# Number of robots to calibrate
N_ROBOTS = 10

# Transform type: 'affine' or 'perspective'
TRANSFORM_TYPE = 'perspective'  # affine = 6 params, perspective = 8 params

# Output file for transformation
OUTPUT_FILE = "camera_transform_perspective.pkl"

#=============================================================================
# Global state for click handling
#=============================================================================

clicked_points = []
fig_handle = None
ax_handle = None
circles = []


def onclick(event):
    """Handle mouse click events."""
    global clicked_points, circles, fig_handle, ax_handle
    
    if event.inaxes != ax_handle:
        return
    
    if event.button == 1:  # Left click
        x, y = event.xdata, event.ydata
        clicked_points.append((x, y))
        
        # Draw a circle at the clicked location
        circle = Circle((x, y), radius=10, color='red', fill=False, linewidth=2)
        ax_handle.add_patch(circle)
        
        # Add text label
        text = ax_handle.text(x + 15, y, f'{len(clicked_points)}', 
                             color='red', fontsize=12, fontweight='bold')
        circles.append((circle, text))
        
        print(f"Point {len(clicked_points)}: ({x:.1f}, {y:.1f})")
        fig_handle.canvas.draw()
        
    elif event.button == 3:  # Right click - undo last point
        if clicked_points:
            clicked_points.pop()
            circle, text = circles.pop()
            circle.remove()
            text.remove()
            print(f"Removed last point. Total points: {len(clicked_points)}")
            fig_handle.canvas.draw()


def onkey(event):
    """Handle key press events."""
    global fig_handle
    
    if event.key == 'q':
        print(f"\nFinished! Collected {len(clicked_points)} points.")
        plt.close(fig_handle)


#=============================================================================
# Main calibration
#=============================================================================

def extract_first_frame(video_path):
    """Extract the first frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Read the first frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("Could not read first frame from video")
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame_rgb


def get_physical_coordinates(data_file, n_robots, frame_idx=0):
    """Get physical (x, y) coordinates of robots at specified frame."""
    print(f"Loading robot data from {data_file}...")
    df = load_robot_data(data_file)
    robot_ids = extract_robot_ids(df)
    robot_ids = robot_ids[:n_robots]
    
    physical_coords = []
    for robot_id in robot_ids:
        states = get_robot_states(df, robot_id, verbose=False)
        x = states['pos_x'][frame_idx]
        y = states['pos_y'][frame_idx]
        physical_coords.append((x, y))
    
    print(f"Extracted {len(physical_coords)} robot positions:")
    for i, (x, y) in enumerate(physical_coords, 1):
        print(f"  Robot {i}: ({x:.4f}, {y:.4f}) m")
    
    return np.array(physical_coords)


def calculate_affine_transform(physical_points, pixel_points):
    """
    Calculate affine transformation from physical to pixel coordinates.
    
    Affine transform: [u, v, 1]^T = M * [x, y, 1]^T
    where M is 3x3 with last row [0, 0, 1]
    
    Parameters need at least 3 point correspondences.
    """
    n_points = len(physical_points)
    
    if n_points < 3:
        raise ValueError("Need at least 3 point correspondences for affine transform")
    
    # Build the system of equations: A * params = b
    # params = [m11, m12, m13, m21, m22, m23]
    A = []
    b = []
    
    for (x, y), (u, v) in zip(physical_points, pixel_points):
        # Equation for u coordinate
        A.append([x, y, 1, 0, 0, 0])
        b.append(u)
        # Equation for v coordinate
        A.append([0, 0, 0, x, y, 1])
        b.append(v)
    
    A = np.array(A)
    b = np.array(b)
    
    # Solve using least squares
    params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Build transformation matrix
    M = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [0, 0, 1]
    ])
    
    return M, residuals


def calculate_perspective_transform(physical_points, pixel_points):
    """
    Calculate perspective (homography) transformation.
    
    Uses OpenCV's findHomography for robust estimation.
    Requires at least 4 point correspondences.
    """
    n_points = len(physical_points)
    
    if n_points < 4:
        raise ValueError("Need at least 4 point correspondences for perspective transform")
    
    # OpenCV expects float32 arrays
    src_points = np.float32(physical_points)
    dst_points = np.float32(pixel_points)
    
    # Calculate homography
    M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    return M, mask


def apply_transform(M, physical_points, is_perspective=False):
    """Apply transformation to physical points to get pixel coordinates."""
    points_homo = np.column_stack([physical_points, np.ones(len(physical_points))])
    
    if is_perspective:
        # For perspective transform, need to normalize by w
        transformed = (M @ points_homo.T).T
        pixel_points = transformed[:, :2] / transformed[:, 2:3]
    else:
        # For affine transform
        pixel_points = (M @ points_homo.T).T[:, :2]
    
    return pixel_points


def main():
    """Main execution."""
    global fig_handle, ax_handle
    
    print("\n" + "="*80)
    print("CAMERA CALIBRATION TOOL")
    print("="*80)
    print(f"Video: {VIDEO_FILE}")
    print(f"Video frame for clicking: {VIDEO_FRAME_NUMBER}")
    print(f"Robot data: {DATA_FILE}")
    print(f"Data frame for physical coords: {DATA_FRAME_NUMBER}")
    print(f"Transform type: {TRANSFORM_TYPE}")
    print("="*80 + "\n")
    
    # Extract first frame from video
    print(f"Extracting frame {VIDEO_FRAME_NUMBER} from video...")
    frame = extract_first_frame(VIDEO_FILE)
    print(f"Frame shape: {frame.shape}")
    
    # Get physical coordinates from data
    print(f"Getting physical coordinates at frame {DATA_FRAME_NUMBER}...")
    physical_coords = get_physical_coordinates(DATA_FILE, N_ROBOTS, DATA_FRAME_NUMBER)
    
    # Display frame and collect clicks
    print("\n" + "-"*80)
    print("INSTRUCTIONS:")
    print("  - Left click on each robot in order (1, 2, 3, ...)")
    print(f"  - Click on {len(physical_coords)} robots total")
    print("  - Right click to undo last point")
    print("  - Press 'q' when done")
    print("-"*80 + "\n")
    
    # Create figure with two subplots side by side
    fig_handle, (ax_handle, ax_positions) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left subplot: video frame for clicking
    ax_handle.imshow(frame)
    ax_handle.set_title(f'Click on robots (0/{len(physical_coords)} selected)', fontsize=14)
    ax_handle.axis('off')
    
    # Right subplot: physical positions with labels
    # X-axis pointing down (vertical), Y-axis pointing right (horizontal)
    colors = plt.cm.tab10(np.linspace(0, 1, len(physical_coords)))
    for i, ((x, y), color) in enumerate(zip(physical_coords, colors), 1):
        ax_positions.scatter(y, x, s=200, color=color, edgecolors='black', 
                           linewidths=2, zorder=3, label=f'Robot {i}')
        ax_positions.text(y, x, str(i), fontsize=12, fontweight='bold',
                         ha='center', va='center', color='white',
                         bbox=dict(boxstyle='circle', facecolor=color, 
                                  edgecolor='black', linewidth=2))
    
    ax_positions.set_xlabel('Y Position (m)', fontsize=12)
    ax_positions.set_ylabel('X Position (m)', fontsize=12)
    ax_positions.set_title(f'Robot Positions at Frame {DATA_FRAME_NUMBER}\n(Click in this order)', fontsize=14)
    ax_positions.invert_yaxis()  # Invert Y-axis so X increases downward
    ax_positions.grid(True, alpha=0.3)
    ax_positions.set_aspect('equal', adjustable='box')
    ax_positions.legend(loc='best', fontsize=10, ncol=2)
    
    plt.tight_layout()
    
    # Connect event handlers
    fig_handle.canvas.mpl_connect('button_press_event', onclick)
    fig_handle.canvas.mpl_connect('key_press_event', onkey)
    
    plt.show()
    
    # Process results
    if len(clicked_points) < 3:
        print("\nError: Need at least 3 points for calibration")
        return
    
    pixel_coords = np.array(clicked_points)
    
    # Use only the number of points we have
    n_points = min(len(clicked_points), len(physical_coords))
    physical_coords = physical_coords[:n_points]
    pixel_coords = pixel_coords[:n_points]
    
    print("\n" + "="*80)
    print("CALCULATING TRANSFORMATION")
    print("="*80)
    print(f"Using {n_points} point correspondences\n")
    
    # Calculate transformation
    if TRANSFORM_TYPE == 'perspective':
        if n_points < 4:
            print("Warning: Need at least 4 points for perspective transform")
            print("Falling back to affine transform")
            M, residuals = calculate_affine_transform(physical_coords, pixel_coords)
            is_perspective = False
        else:
            M, mask = calculate_perspective_transform(physical_coords, pixel_coords)
            is_perspective = True
    else:
        M, residuals = calculate_affine_transform(physical_coords, pixel_coords)
        is_perspective = False
    
    print("Transformation matrix:")
    print(M)
    print()
    
    # Validate transformation
    print("Validation (physical -> predicted pixel):")
    predicted_pixels = apply_transform(M, physical_coords, is_perspective)
    
    errors = []
    for i, (phys, actual_pix, pred_pix) in enumerate(zip(physical_coords, pixel_coords, predicted_pixels), 1):
        error = np.linalg.norm(actual_pix - pred_pix)
        errors.append(error)
        print(f"  Point {i}: Physical ({phys[0]:.4f}, {phys[1]:.4f}) -> "
              f"Actual ({actual_pix[0]:.1f}, {actual_pix[1]:.1f}) "
              f"Predicted ({pred_pix[0]:.1f}, {pred_pix[1]:.1f}) "
              f"Error: {error:.2f} px")
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    print(f"\nMean error: {mean_error:.2f} pixels")
    print(f"Max error: {max_error:.2f} pixels")
    
    # Save transformation
    transform_data = {
        'matrix': M,
        'type': 'perspective' if is_perspective else 'affine',
        'physical_points': physical_coords,
        'pixel_points': pixel_coords,
        'mean_error': mean_error,
        'max_error': max_error,
        'video_file': VIDEO_FILE,
        'data_file': DATA_FILE,
        'frame_shape': frame.shape
    }
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(transform_data, f)
    
    print(f"\n✓ Transformation saved to: {OUTPUT_FILE}")
    print("="*80)
    
    # Visualize results
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(frame)
    ax.set_title(f'Calibration Results (Mean error: {mean_error:.2f} px)', fontsize=14)
    ax.axis('off')
    
    # Draw actual clicks
    for i, (x, y) in enumerate(pixel_coords, 1):
        circle = Circle((x, y), radius=10, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.text(x + 15, y, f'{i}', color='red', fontsize=12, fontweight='bold')
    
    # Draw predicted positions
    for i, (x, y) in enumerate(predicted_pixels, 1):
        circle = Circle((x, y), radius=8, color='lime', fill=False, linewidth=2, linestyle='--')
        ax.add_patch(circle)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='red', label='Clicked', linewidth=2),
        Patch(facecolor='none', edgecolor='lime', label='Predicted', linewidth=2, linestyle='--')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('calibration_validation.png', dpi=150, bbox_inches='tight')
    print(f"✓ Validation plot saved to: calibration_validation.png")
    plt.show()


if __name__ == '__main__':
    main()
