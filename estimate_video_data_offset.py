#!/usr/bin/env python3
"""
Estimate temporal offset between video and robot data using camera calibration.

This script allows you to click on robots in video frames, then uses the
calibrated camera to project robot positions from the data and finds the
offset that best matches your clicks.
"""

import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# Import robot data loading functions from the existing utility module
from plot_robot_data import load_robot_data, extract_robot_ids, get_robot_states

#=============================================================================
# Configuration
#=============================================================================

# Input files
VIDEO_FILE = "video_edits/tune-period-80-retake4(1).mkv"
DATA_FILE = "period80-retakes/time-period-80-retake4.csv.gz"
CALIBRATION_FILE = "camera_transform_aruco.pkl"

# Frames to use for offset estimation (video frame numbers)
# Choose frames with clear robot visibility and good distribution in time
CALIBRATION_FRAMES = [1000, 2000,3000,4000]

# Which robots to use (0-indexed, or None for all)
ROBOT_IDS_TO_USE = [0,1,2,3,4,5,6,7,8,9]  # e.g., [0, 1, 2] or None for all

# Search range for offset (in frames)
# Positive offset means data is ahead of video (video needs more offset to catch up)
OFFSET_SEARCH_MIN = -60
OFFSET_SEARCH_MAX = 60

# Visualization
MARKER_SIZE = 10
SHOW_ROBOT_LABELS = True

#=============================================================================
# Data loading functions
#=============================================================================

def load_calibration(calib_file):
    """Load calibration data from pickle file."""
    with open(calib_file, 'rb') as f:
        return pickle.load(f)


#=============================================================================
# Projection functions
#=============================================================================

def project_3d_to_2d(points_3d, camera_matrix, rvec, tvec, dist_coeffs):
    """Project 3D world points to 2D image using camera pose."""
    if len(points_3d.shape) == 1:
        points_3d = points_3d.reshape(1, -1)
    
    points_3d = points_3d.astype(np.float32)
    projected_2d, _ = cv2.projectPoints(
        points_3d, rvec, tvec, camera_matrix, dist_coeffs
    )
    pixel_points = projected_2d.reshape(-1, 2)
    return pixel_points


def apply_transform_2d(M, physical_points, is_perspective=False):
    """Apply 2D transformation (legacy calibration method)."""
    if len(physical_points.shape) == 1:
        physical_points = physical_points.reshape(1, -1)
    
    points_homo = np.column_stack([physical_points, np.ones(len(physical_points))])
    
    if is_perspective:
        transformed = (M @ points_homo.T).T
        pixel_points = transformed[:, :2] / transformed[:, 2:3]
    else:
        pixel_points = (M @ points_homo.T).T[:, :2]
    
    return pixel_points


#=============================================================================
# Interactive clicking
#=============================================================================

class RobotClicker:
    """Interactive tool for clicking on robots in video frames using matplotlib."""
    
    def __init__(self, video_file, frames_to_click, n_robots, robot_data):
        self.video_file = video_file
        self.frames_to_click = sorted(frames_to_click)
        self.n_robots = n_robots
        self.robot_data = robot_data
        self.current_frame_idx = 0
        self.current_robot_idx = 0
        
        # Store clicks: clicks[frame_num][robot_idx] = (x, y)
        self.clicks = {frame: {} for frame in self.frames_to_click}
        
        # Colors for different robots (RGB for matplotlib)
        np.random.seed(42)
        self.colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow',
                       'orange', 'purple', 'pink']
        
        self.frame = None
        self.fig = None
        self.ax_video = None
        self.ax_positions = None
        self.current_data_frame = 0
    
    def get_frame(self, frame_number):
        """Get specific frame from video."""
        cap = cv2.VideoCapture(self.video_file)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if ret:
            # Convert BGR to RGB for matplotlib
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame if ret else None
    
    def update_display(self):
        """Update the display with current clicks."""
        # Clear both axes
        self.ax_video.clear()
        self.ax_positions.clear()
        
        frame_num = self.frames_to_click[self.current_frame_idx]
        
        # Left: Video frame with clicks
        self.ax_video.imshow(self.frame)
        self.ax_video.axis('off')
        
        # Draw all existing clicks for this frame
        for robot_idx, (x, y) in self.clicks[frame_num].items():
            color = self.colors[robot_idx % len(self.colors)]
            self.ax_video.plot(x, y, 'o', color=color, markersize=MARKER_SIZE, 
                        markeredgewidth=2, markerfacecolor='none')
            self.ax_video.plot(x, y, '.', color=color, markersize=3)
            if SHOW_ROBOT_LABELS:
                self.ax_video.text(x + 15, y, f"R{robot_idx+1}", color=color, 
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Right: Robot positions from data at this frame
        # X-axis pointing down (vertical), Y-axis pointing right (horizontal)
        for robot_idx in range(self.n_robots):
            color = self.colors[robot_idx % len(self.colors)]
            x = self.robot_data[robot_idx]['pos_x'][self.current_data_frame]
            y = self.robot_data[robot_idx]['pos_y'][self.current_data_frame]
            
            # Highlight the current robot to click
            if robot_idx == self.current_robot_idx:
                marker_size = 300
                linewidth = 3
                alpha = 1.0
            else:
                marker_size = 200
                linewidth = 2
                alpha = 0.6
            
            self.ax_positions.scatter(y, x, s=marker_size, color=color, 
                                    edgecolors='black', linewidths=linewidth, 
                                    zorder=3, alpha=alpha)
            self.ax_positions.text(y, x, str(robot_idx+1), fontsize=12, 
                                 fontweight='bold', ha='center', va='center', 
                                 color='white',
                                 bbox=dict(boxstyle='circle', facecolor=color, 
                                          edgecolor='black', linewidth=2))
        
        self.ax_positions.set_xlabel('Y Position (m)', fontsize=11)
        self.ax_positions.set_ylabel('X Position (m)', fontsize=11)
        self.ax_positions.set_title(f'Robot Positions (Data Frame {self.current_data_frame})\n(Click robots in this order)', 
                                   fontsize=12)
        self.ax_positions.invert_yaxis()  # Invert Y-axis so X increases downward
        self.ax_positions.grid(True, alpha=0.3)
        self.ax_positions.set_aspect('equal', adjustable='box')
        
        # Show progress and instructions
        total_clicks = sum(len(clicks) for clicks in self.clicks.values())
        total_needed = len(self.frames_to_click) * self.n_robots
        
        if self.current_robot_idx < self.n_robots:
            color = self.colors[self.current_robot_idx % len(self.colors)]
            instruction = f"Click on Robot {self.current_robot_idx + 1} in VIDEO ({total_clicks}/{total_needed} done)"
        else:
            instruction = f"Frame done! Press SPACE for next frame ({total_clicks}/{total_needed} done)"
        
        title = f"Video Frame {frame_num} - {instruction}\n(Press 'c' to clear last click, 'q' to finish early)"
        self.ax_video.set_title(title, fontsize=12, pad=10)
        
        self.fig.canvas.draw()
    
    def on_click(self, event):
        """Handle mouse clicks."""
        if event.inaxes != self.ax_video:
            return
        
        if event.button == 1:  # Left click
            if self.current_robot_idx < self.n_robots:
                frame_num = self.frames_to_click[self.current_frame_idx]
                x, y = event.xdata, event.ydata
                self.clicks[frame_num][self.current_robot_idx] = (x, y)
                print(f"  Clicked robot {self.current_robot_idx + 1} at frame {frame_num}: ({x:.1f}, {y:.1f})")
                self.current_robot_idx += 1
                self.update_display()
    
    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'c':  # Clear last click
            if self.current_robot_idx > 0:
                frame_num = self.frames_to_click[self.current_frame_idx]
                self.current_robot_idx -= 1
                if self.current_robot_idx in self.clicks[frame_num]:
                    del self.clicks[frame_num][self.current_robot_idx]
                print(f"  Cleared robot {self.current_robot_idx + 1}")
                self.update_display()
        
        elif event.key == ' ':  # Space - next frame
            if self.current_robot_idx >= self.n_robots:
                self.current_frame_idx += 1
                if self.current_frame_idx < len(self.frames_to_click):
                    self.load_next_frame()
                else:
                    print("\n✓ Clicking completed!")
                    plt.close(self.fig)
        
        elif event.key == 'q':  # Quit early
            print("\nExiting early...")
            plt.close(self.fig)
    
    def load_next_frame(self):
        """Load and display the next frame."""
        frame_num = self.frames_to_click[self.current_frame_idx]
        # Use frame_num as the data frame (will be adjusted by offset search)
        self.current_data_frame = frame_num
        
        print(f"\nFrame {frame_num} ({self.current_frame_idx + 1}/{len(self.frames_to_click)})")
        
        self.frame = self.get_frame(frame_num)
        if self.frame is None:
            print(f"  Warning: Could not read frame {frame_num}")
            self.current_frame_idx += 1
            if self.current_frame_idx < len(self.frames_to_click):
                self.load_next_frame()
            else:
                plt.close(self.fig)
            return
        
        # Check if data frame is valid
        if self.current_data_frame >= len(self.robot_data[0]['pos_x']):
            print(f"  Warning: Data frame {self.current_data_frame} out of range")
            self.current_frame_idx += 1
            if self.current_frame_idx < len(self.frames_to_click):
                self.load_next_frame()
            else:
                plt.close(self.fig)
            return
        
        self.current_robot_idx = 0
        self.update_display()
    
    def run(self):
        """Run the interactive clicking session."""
        print("\n" + "="*80)
        print("INTERACTIVE ROBOT CLICKING")
        print("="*80)
        print(f"Frames to process: {self.frames_to_click}")
        print(f"Robots per frame: {self.n_robots}")
        print("\nInstructions:")
        print("  - LEFT panel: Video frame - CLICK HERE on each robot")
        print("  - RIGHT panel: Data positions - shows robot order to click")
        print("  - Press SPACE to move to next frame after all robots clicked")
        print("  - Press 'c' to clear the last click")
        print("  - Press 'q' to finish early with current clicks")
        print("="*80 + "\n")
        
        # Create figure with two subplots side by side
        self.fig, (self.ax_video, self.ax_positions) = plt.subplots(1, 2, figsize=(18, 9))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Load first frame
        self.load_next_frame()
        
        plt.tight_layout()
        plt.show()
        
        return self.clicks


#=============================================================================
# Offset estimation
#=============================================================================

def compute_offset_error(clicks, robot_data, calib_data, offset, use_3d):
    """
    Compute reprojection error for a given offset.
    
    Args:
        clicks: Dict mapping frame_num -> robot_idx -> (x, y)
        robot_data: List of robot state dicts
        calib_data: Camera calibration data
        offset: Frame offset to test (data_frame = video_frame + offset)
        use_3d: Whether to use 3D projection or 2D transform
        
    Returns:
        mean_error: Mean pixel distance between clicks and projections
        errors_per_point: List of individual errors
    """
    errors = []
    
    if use_3d:
        camera_matrix = calib_data['camera_matrix']
        rvec = calib_data['rvec']
        tvec = calib_data['tvec']
        dist_coeffs = calib_data['dist_coeffs']
    else:
        M = calib_data['matrix']
        is_perspective = calib_data['type'] == 'perspective'
    
    for video_frame, robot_clicks in clicks.items():
        data_frame = video_frame + offset
        
        # Check if data frame is valid
        if data_frame < 0 or data_frame >= len(robot_data[0]['pos_x']):
            continue
        
        for robot_idx, (click_x, click_y) in robot_clicks.items():
            # Get robot position from data
            phys_x = robot_data[robot_idx]['pos_x'][data_frame]
            phys_y = robot_data[robot_idx]['pos_y'][data_frame]
            
            # Project to image
            if use_3d:
                phys_pos_3d = np.array([phys_x, phys_y, 0.0])
                pixel_pos = project_3d_to_2d(phys_pos_3d, camera_matrix, rvec, tvec, dist_coeffs)[0]
            else:
                phys_pos = np.array([phys_x, phys_y])
                pixel_pos = apply_transform_2d(M, phys_pos, is_perspective)[0]
            
            # Compute error
            error = np.sqrt((pixel_pos[0] - click_x)**2 + (pixel_pos[1] - click_y)**2)
            errors.append(error)
    
    if len(errors) == 0:
        return float('inf'), []
    
    return np.mean(errors), errors


def find_best_offset(clicks, robot_data, calib_data, offset_min, offset_max, use_3d):
    """
    Search for the offset that minimizes reprojection error.
    
    Returns:
        best_offset: Offset with minimum error
        offset_errors: List of (offset, mean_error) tuples
    """
    print("\n" + "="*80)
    print("SEARCHING FOR BEST OFFSET")
    print("="*80)
    print(f"Search range: {offset_min} to {offset_max} frames")
    
    offset_errors = []
    best_offset = None
    best_error = float('inf')
    
    for offset in range(offset_min, offset_max + 1):
        mean_error, _ = compute_offset_error(clicks, robot_data, calib_data, offset, use_3d)
        offset_errors.append((offset, mean_error))
        
        if mean_error < best_error:
            best_error = mean_error
            best_offset = offset
        
        if offset % 10 == 0:
            print(f"  Offset {offset:4d}: error = {mean_error:.2f} px")
    
    print("="*80)
    print(f"\n✓ Best offset: {best_offset} frames (error = {best_error:.2f} px)")
    print(f"  Use DATA_FRAME_OFFSET = {best_offset} in overlay script")
    
    return best_offset, offset_errors


#=============================================================================
# Visualization
#=============================================================================

def plot_offset_analysis(offset_errors, best_offset, clicks, robot_data, calib_data, use_3d):
    """Create visualization of offset search results."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Error vs offset
    offsets = [o for o, _ in offset_errors]
    errors = [e for _, e in offset_errors]
    
    ax1.plot(offsets, errors, 'b-', linewidth=2)
    ax1.axvline(best_offset, color='r', linestyle='--', linewidth=2, label=f'Best offset: {best_offset}')
    ax1.axhline(min(errors), color='g', linestyle=':', alpha=0.5, label=f'Min error: {min(errors):.2f} px')
    ax1.set_xlabel('Offset (frames)', fontsize=12)
    ax1.set_ylabel('Mean reprojection error (pixels)', fontsize=12)
    ax1.set_title('Offset Search Results', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Distribution of errors at best offset
    _, errors_per_point = compute_offset_error(clicks, robot_data, calib_data, best_offset, use_3d)
    
    ax2.hist(errors_per_point, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(errors_per_point), color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(errors_per_point):.2f} px')
    ax2.axvline(np.median(errors_per_point), color='g', linestyle='--', linewidth=2,
                label=f'Median: {np.median(errors_per_point):.2f} px')
    ax2.set_xlabel('Reprojection error (pixels)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Error Distribution at Best Offset ({best_offset} frames)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'offset_estimation_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Analysis plot saved to: {output_file}")
    
    plt.show()


#=============================================================================
# Main
#=============================================================================

def main():
    print("\n" + "="*80)
    print("VIDEO-DATA OFFSET ESTIMATION")
    print("="*80)
    print(f"Video: {VIDEO_FILE}")
    print(f"Data: {DATA_FILE}")
    print(f"Calibration: {CALIBRATION_FILE}")
    print("="*80 + "\n")
    
    # Load calibration
    print("Loading calibration...")
    calib_data = load_calibration(CALIBRATION_FILE)
    
    # Check calibration type
    if 'camera_matrix' in calib_data:
        use_3d = True
        print("  Using 3D camera pose calibration")
    else:
        use_3d = False
        print(f"  Using 2D {calib_data['type']} calibration")
    
    # Load robot data
    print("\nLoading robot data...")
    df = load_robot_data(DATA_FILE)
    robot_ids = extract_robot_ids(df)
    
    if ROBOT_IDS_TO_USE is not None:
        robot_ids = [robot_ids[i] for i in ROBOT_IDS_TO_USE]
    
    print(f"  Found {len(robot_ids)} robots: {robot_ids}")
    
    robot_data = []
    for robot_id in robot_ids:
        states = get_robot_states(df, robot_id, verbose=True)
        robot_data.append(states)
    
    n_data_frames = len(robot_data[0]['pos_x'])
    print(f"  Data frames: {n_data_frames}")
    
    # Interactive clicking
    clicker = RobotClicker(VIDEO_FILE, CALIBRATION_FRAMES, len(robot_ids), robot_data)
    clicks = clicker.run()
    
    # Count valid clicks
    total_clicks = sum(len(robot_clicks) for robot_clicks in clicks.values())
    print(f"\nTotal clicks collected: {total_clicks}")
    
    if total_clicks == 0:
        print("No clicks collected. Exiting.")
        return
    
    # Find best offset
    best_offset, offset_errors = find_best_offset(
        clicks, robot_data, calib_data,
        OFFSET_SEARCH_MIN, OFFSET_SEARCH_MAX, use_3d
    )
    
    # Visualize results
    plot_offset_analysis(offset_errors, best_offset, clicks, robot_data, calib_data, use_3d)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nRecommended setting for overlay_data_on_video.py:")
    print(f"  DATA_FRAME_OFFSET = {best_offset}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
