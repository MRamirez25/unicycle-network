#!/usr/bin/env python3
"""
Iteratively refine camera calibration by adjusting clicked points.

This script:
1. Loads existing calibration
2. Shows video frame with current points and predictions
3. Lets you adjust individual point positions
4. Recalculates transformation after each change
5. Shows updated error metrics
6. Saves improved calibration

Controls:
- Click near an existing point to select it
- Click elsewhere to move the selected point
- Press 'n' to cycle to next point
- Press 'd' to delete selected point
- Press 'a' to add new point
- Press 't' to toggle between affine/perspective transform
- Press 's' to save current calibration
- Press 'q' to quit
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

# Input calibration file
INPUT_CALIBRATION = "camera_transform_perspective.pkl"

# Output calibration file (will be created)
OUTPUT_CALIBRATION = "camera_transform_refined.pkl"

# Selection radius (pixels)
SELECTION_RADIUS = 20

#=============================================================================
# Global state
#=============================================================================

clicked_points = []
physical_coords = []
frame = None
fig_handle = None
ax_handle = None
selected_idx = None
transform_type = 'perspective'
circles = []
pred_circles = []
lines = []


def calculate_affine_transform(physical_points, pixel_points):
    """Calculate affine transformation."""
    n_points = len(physical_points)
    
    if n_points < 3:
        raise ValueError("Need at least 3 point correspondences for affine transform")
    
    A = []
    b = []
    
    for (x, y), (u, v) in zip(physical_points, pixel_points):
        A.append([x, y, 1, 0, 0, 0])
        b.append(u)
        A.append([0, 0, 0, x, y, 1])
        b.append(v)
    
    A = np.array(A)
    b = np.array(b)
    
    params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    M = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [0, 0, 1]
    ])
    
    return M, residuals


def calculate_perspective_transform(physical_points, pixel_points):
    """Calculate perspective transformation."""
    n_points = len(physical_points)
    
    if n_points < 4:
        raise ValueError("Need at least 4 point correspondences for perspective transform")
    
    src_points = np.float32(physical_points)
    dst_points = np.float32(pixel_points)
    
    M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    return M, mask


def apply_transform(M, physical_points, is_perspective=False):
    """Apply transformation to physical points."""
    if len(physical_points.shape) == 1:
        physical_points = physical_points.reshape(1, -1)
    
    points_homo = np.column_stack([physical_points, np.ones(len(physical_points))])
    
    if is_perspective:
        transformed = (M @ points_homo.T).T
        pixel_points = transformed[:, :2] / transformed[:, 2:3]
    else:
        pixel_points = (M @ points_homo.T).T[:, :2]
    
    return pixel_points


def calculate_current_transform():
    """Calculate transformation with current points."""
    global clicked_points, physical_coords, transform_type
    
    if len(clicked_points) < 3:
        return None, None, None, None
    
    pixel_coords = np.array(clicked_points)
    phys_coords = physical_coords[:len(clicked_points)]
    
    try:
        if transform_type == 'perspective' and len(clicked_points) >= 4:
            M, mask = calculate_perspective_transform(phys_coords, pixel_coords)
            is_perspective = True
        else:
            M, residuals = calculate_affine_transform(phys_coords, pixel_coords)
            is_perspective = False
        
        # Calculate predictions and errors
        predicted = apply_transform(M, phys_coords, is_perspective)
        errors = np.linalg.norm(pixel_coords - predicted, axis=1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        return M, is_perspective, mean_error, max_error
    except Exception as e:
        print(f"Error calculating transform: {e}")
        return None, None, None, None


def update_display():
    """Update the display with current points and predictions."""
    global fig_handle, ax_handle, clicked_points, physical_coords
    global circles, pred_circles, lines, selected_idx, transform_type
    
    # Clear previous annotations
    for circle, text in circles:
        circle.remove()
        text.remove()
    for circle in pred_circles:
        circle.remove()
    for line in lines:
        line.remove()
    circles.clear()
    pred_circles.clear()
    lines.clear()
    
    # Calculate current transform
    M, is_perspective, mean_error, max_error = calculate_current_transform()
    
    # Update title
    if M is not None:
        title = (f'Refining Calibration - {transform_type.upper()} '
                f'({len(clicked_points)} pts, Mean error: {mean_error:.2f} px, '
                f'Max: {max_error:.2f} px)')
    else:
        title = f'Refining Calibration - {transform_type.upper()} ({len(clicked_points)} pts)'
    ax_handle.set_title(title, fontsize=12)
    
    # Draw clicked points and predictions
    if M is not None:
        phys_coords = physical_coords[:len(clicked_points)]
        predicted = apply_transform(M, phys_coords, is_perspective)
        
        for i, (clicked, pred) in enumerate(zip(clicked_points, predicted)):
            # Line from clicked to predicted
            line, = ax_handle.plot([clicked[0], pred[0]], [clicked[1], pred[1]], 
                                  'y-', linewidth=1, alpha=0.5)
            lines.append(line)
    
    # Draw points
    for i, (x, y) in enumerate(clicked_points):
        # Color based on selection
        if i == selected_idx:
            color = 'yellow'
            lw = 3
        else:
            color = 'red'
            lw = 2
        
        circle = Circle((x, y), radius=10, color=color, fill=False, linewidth=lw)
        ax_handle.add_patch(circle)
        
        text = ax_handle.text(x + 15, y, f'{i+1}', 
                             color=color, fontsize=12, fontweight='bold')
        circles.append((circle, text))
    
    # Draw predicted positions (green)
    if M is not None:
        for i, (x, y) in enumerate(predicted):
            circle = Circle((x, y), radius=8, color='lime', fill=False, 
                          linewidth=2, linestyle='--')
            ax_handle.add_patch(circle)
            pred_circles.append(circle)
    
    fig_handle.canvas.draw()


def find_nearest_point(x, y):
    """Find index of nearest clicked point within selection radius."""
    if not clicked_points:
        return None
    
    points = np.array(clicked_points)
    dists = np.linalg.norm(points - np.array([x, y]), axis=1)
    min_idx = np.argmin(dists)
    
    if dists[min_idx] <= SELECTION_RADIUS:
        return min_idx
    return None


def onclick(event):
    """Handle mouse click events."""
    global clicked_points, selected_idx
    
    if event.inaxes != ax_handle:
        return
    
    if event.button == 1:  # Left click
        x, y = event.xdata, event.ydata
        
        # Check if clicking near existing point
        nearest = find_nearest_point(x, y)
        
        if nearest is not None:
            # Select this point
            selected_idx = nearest
            print(f"Selected point {selected_idx + 1}")
        elif selected_idx is not None:
            # Move selected point to new location
            clicked_points[selected_idx] = (x, y)
            print(f"Moved point {selected_idx + 1} to ({x:.1f}, {y:.1f})")
            selected_idx = None
        else:
            # Add new point
            if len(clicked_points) < len(physical_coords):
                clicked_points.append((x, y))
                print(f"Added point {len(clicked_points)} at ({x:.1f}, {y:.1f})")
            else:
                print("All robots already have points. Delete one first to add more.")
        
        update_display()
    
    elif event.button == 3:  # Right click - delete point
        x, y = event.xdata, event.ydata
        nearest = find_nearest_point(x, y)
        
        if nearest is not None:
            clicked_points.pop(nearest)
            if selected_idx == nearest:
                selected_idx = None
            elif selected_idx is not None and selected_idx > nearest:
                selected_idx -= 1
            print(f"Deleted point. {len(clicked_points)} points remaining.")
            update_display()


def onkey(event):
    """Handle key press events."""
    global selected_idx, transform_type, fig_handle
    
    if event.key == 'n':  # Next point
        if clicked_points:
            if selected_idx is None:
                selected_idx = 0
            else:
                selected_idx = (selected_idx + 1) % len(clicked_points)
            print(f"Selected point {selected_idx + 1}")
            update_display()
    
    elif event.key == 't':  # Toggle transform type
        if transform_type == 'affine':
            transform_type = 'perspective'
        else:
            transform_type = 'affine'
        print(f"Switched to {transform_type} transform")
        update_display()
    
    elif event.key == 's':  # Save
        save_calibration()
    
    elif event.key == 'q':  # Quit
        print("\nQuitting...")
        plt.close(fig_handle)


def save_calibration():
    """Save current calibration to file."""
    global clicked_points, physical_coords, transform_type, frame
    
    M, is_perspective, mean_error, max_error = calculate_current_transform()
    
    if M is None:
        print("Cannot save: insufficient points for calibration")
        return
    
    pixel_coords = np.array(clicked_points)
    phys_coords = physical_coords[:len(clicked_points)]
    
    transform_data = {
        'matrix': M,
        'type': 'perspective' if is_perspective else 'affine',
        'physical_points': phys_coords,
        'pixel_points': pixel_coords,
        'mean_error': mean_error,
        'max_error': max_error,
        'video_file': 'refined',
        'data_file': 'refined',
        'frame_shape': frame.shape
    }
    
    with open(OUTPUT_CALIBRATION, 'wb') as f:
        pickle.dump(transform_data, f)
    
    print(f"\n✓ Refined calibration saved to: {OUTPUT_CALIBRATION}")
    print(f"  Transform type: {transform_data['type']}")
    print(f"  Points: {len(clicked_points)}")
    print(f"  Mean error: {mean_error:.2f} px")
    print(f"  Max error: {max_error:.2f} px")


def main():
    """Main execution."""
    global clicked_points, physical_coords, frame, fig_handle, ax_handle, transform_type
    
    print("\n" + "="*80)
    print("CAMERA CALIBRATION REFINEMENT TOOL")
    print("="*80)
    print(f"Input: {INPUT_CALIBRATION}")
    print(f"Output: {OUTPUT_CALIBRATION}")
    print("="*80 + "\n")
    
    # Load existing calibration
    print("Loading existing calibration...")
    with open(INPUT_CALIBRATION, 'rb') as f:
        calib_data = pickle.load(f)
    
    print(f"Current calibration:")
    print(f"  Type: {calib_data['type']}")
    print(f"  Points: {len(calib_data['pixel_points'])}")
    print(f"  Mean error: {calib_data['mean_error']:.2f} px")
    print(f"  Max error: {calib_data['max_error']:.2f} px")
    
    # Load data
    clicked_points = list(map(tuple, calib_data['pixel_points']))
    physical_coords = calib_data['physical_points']
    transform_type = calib_data['type']
    
    # Load frame
    print("\nLoading video frame...")
    video_file = calib_data['video_file']
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_file}")
        print("Creating blank frame...")
        frame = np.zeros(calib_data['frame_shape'], dtype=np.uint8)
    else:
        ret, frame_bgr = cap.read()
        cap.release()
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Display instructions
    print("\n" + "-"*80)
    print("INSTRUCTIONS:")
    print("  - Click near a point to select it (turns yellow)")
    print("  - Click elsewhere to move selected point")
    print("  - Right-click on point to delete it")
    print("  - Press 'n' to cycle through points")
    print("  - Press 't' to toggle affine/perspective transform")
    print("  - Press 's' to save refined calibration")
    print("  - Press 'q' to quit")
    print("-"*80 + "\n")
    
    # Create figure
    fig_handle, ax_handle = plt.subplots(figsize=(14, 10))
    ax_handle.imshow(frame)
    ax_handle.axis('off')
    
    # Initial display
    update_display()
    
    # Connect event handlers
    fig_handle.canvas.mpl_connect('button_press_event', onclick)
    fig_handle.canvas.mpl_connect('key_press_event', onkey)
    
    plt.show()
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
