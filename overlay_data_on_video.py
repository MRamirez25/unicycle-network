#!/usr/bin/env python3
"""
Overlay robot data positions onto video using camera transformation.

This script:
1. Loads the camera transformation from calibration
2. Reads the video and robot data
3. Projects physical coordinates to pixel coordinates
4. Overlays robot positions on the video
5. Creates comparison video showing real robots with data overlay
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from plot_robot_data import load_robot_data, extract_robot_ids, get_robot_states

#=============================================================================
# CONFIGURATION
#=============================================================================

# Calibration file
CALIBRATION_FILE = "camera_transform_aruco.pkl"

# Video file (actual camera recording)
VIDEO_FILE = "video_edits/tune-period-80-retake4(1).mkv"

# Robot data file
DATA_FILE = "period80-retakes/time-period-80-retake4.csv.gz"

# Number of robots to display
N_ROBOTS = 10

# Time step in data (seconds)
DT = 0.05

# Video output
OUTPUT_VIDEO = "video_overlay_no_springs.mp4"
OUTPUT_FPS = 30  # Output video frame rate

# Visualization settings
SHOW_TRAILS = False
TRAIL_LENGTH = 400  # Number of past positions to show
SHOW_ARROWS = False  # Show orientation arrows
ARROW_LENGTH = 0.3  # meters

# Spring connections
SHOW_SPRINGS = False  # Draw springs connecting robots
SPRING_CENTER_ROBOTS = [0,1,2,3,4,5,6,7,8,9]  # Robot indices to connect from (0-indexed), can be a list or single int
SPRING_COLOR = (0, 0, 0)  # BGR color for springs (gray - use non-black for fade to work!)
SPRING_THICKNESS = 3
SPRING_COILS = 32  # Number of zigzags in the spring
SPRING_AMPLITUDE = 0.03  # Amplitude of spring oscillations (meters)

# Spring animation (cycle through centers)
SPRING_ANIMATE_CENTERS = True  # If True, cycle through centers; if False, show all simultaneously
SPRING_FRAMES_PER_CENTER = 100  # Number of frames to show each center's connections
SPRING_FADE_FRAMES = 50  # Number of frames for fade in/out transition

# Frame range to process (None = all frames)
START_FRAME = 1000
END_FRAME = 2000  # None means until end of video

# Frame alignment offsets
VIDEO_FRAME_OFFSET = 0  # Start video from this frame (positive = skip video frames)
DATA_FRAME_OFFSET = 0   # Start data from this frame (positive = skip data frames)
# Example: If data starts 10 frames later than video, set DATA_FRAME_OFFSET = 10

# Video cropping (remove pixels from edges)
CROP_TOP = 600        # Pixels to remove from top
CROP_BOTTOM = 500     # Pixels to remove from bottom
CROP_LEFT = 500       # Pixels to remove from left
CROP_RIGHT = 300      # Pixels to remove from right

#=============================================================================
# Load calibration
#=============================================================================

def load_calibration(calib_file):
    """Load camera transformation from calibration file."""
    with open(calib_file, 'rb') as f:
        calib_data = pickle.load(f)
    
    print("Calibration loaded:")
    print(f"  Transform type: {calib_data['type']}")
    print(f"  Mean error: {calib_data['mean_error']:.2f} pixels")
    print(f"  Max error: {calib_data['max_error']:.2f} pixels")
    print(f"  Video: {calib_data['video_file']}")
    print(f"  Data: {calib_data['data_file']}")
    
    return calib_data


def apply_transform(M, physical_points, is_perspective=False):
    """Apply 2D transformation to physical points to get pixel coordinates (legacy method)."""
    if len(physical_points.shape) == 1:
        physical_points = physical_points.reshape(1, -1)
    
    points_homo = np.column_stack([physical_points, np.ones(len(physical_points))])
    
    if is_perspective:
        # For perspective transform, need to normalize by w
        transformed = (M @ points_homo.T).T
        pixel_points = transformed[:, :2] / transformed[:, 2:3]
    else:
        # For affine transform
        pixel_points = (M @ points_homo.T).T[:, :2]
    
    return pixel_points


def project_3d_to_2d(points_3d, camera_matrix, rvec, tvec, dist_coeffs):
    """Project 3D world points to 2D image using camera pose (proper 3D method)."""
    if len(points_3d.shape) == 1:
        points_3d = points_3d.reshape(1, -1)
    
    # Ensure points are float32 for OpenCV
    points_3d = points_3d.astype(np.float32)
    
    # Project using camera parameters
    projected_2d, _ = cv2.projectPoints(
        points_3d, rvec, tvec, camera_matrix, dist_coeffs
    )
    
    # Reshape from (N, 1, 2) to (N, 2)
    pixel_points = projected_2d.reshape(-1, 2)
    
    return pixel_points


def draw_spring(overlay, start_pixel, end_pixel, color, thickness, n_coils, amplitude_pixels):
    """
    Draw a spring-like line between two points.
    
    Args:
        overlay: Image to draw on
        start_pixel: (x, y) start point
        end_pixel: (x, y) end point
        color: BGR color tuple
        thickness: Line thickness
        n_coils: Number of oscillations/coils
        amplitude_pixels: Amplitude of oscillations in pixels
    """
    start = np.array(start_pixel, dtype=float)
    end = np.array(end_pixel, dtype=float)
    
    # Vector from start to end
    vec = end - start
    length = np.linalg.norm(vec)
    
    if length < 1:  # Too close, just draw a line
        cv2.line(overlay, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness)
        return
    
    # Unit vector along the line
    unit_vec = vec / length
    
    # Perpendicular unit vector (for oscillations)
    perp_vec = np.array([-unit_vec[1], unit_vec[0]])
    
    # Generate spring points
    n_points = n_coils * 4 + 1  # 4 points per coil
    t = np.linspace(0, 1, n_points)
    
    points = []
    for i, ti in enumerate(t):
        # Position along the line
        base_pos = start + ti * vec
        
        # Oscillation amplitude (fade at ends)
        fade = np.sin(ti * np.pi)  # 0 at ends, 1 in middle
        
        # Oscillation (square wave pattern for spring look)
        phase = (i % 4) / 4.0  # 0, 0.25, 0.5, 0.75
        if phase < 0.25:
            offset = amplitude_pixels * fade
        elif phase < 0.5:
            offset = -amplitude_pixels * fade
        elif phase < 0.75:
            offset = -amplitude_pixels * fade
        else:
            offset = amplitude_pixels * fade
        
        # Add perpendicular offset
        point = base_pos + offset * perp_vec
        points.append(point.astype(int))
    
    # Draw the spring as connected line segments
    for i in range(len(points) - 1):
        cv2.line(overlay, tuple(points[i]), tuple(points[i + 1]), color, thickness, cv2.LINE_AA)


#=============================================================================
# Main processing
#=============================================================================

def main():
    """Main execution."""
    
    print("\n" + "="*80)
    print("VIDEO OVERLAY WITH ROBOT DATA")
    print("="*80)
    print(f"Video: {VIDEO_FILE}")
    print(f"Data: {DATA_FILE}")
    print(f"Output: {OUTPUT_VIDEO}")
    print("="*80 + "\n")
    
    # Load calibration
    print("Loading calibration...")
    calib_data = load_calibration(CALIBRATION_FILE)
    
    # Check calibration type and extract parameters
    if 'camera_matrix' in calib_data:
        # New 3D camera pose calibration
        use_3d = True
        camera_matrix = calib_data['camera_matrix']
        rvec = calib_data['rvec']
        tvec = calib_data['tvec']
        dist_coeffs = calib_data['dist_coeffs']
        print("Using 3D camera pose calibration (solvePnP)")
    else:
        # Legacy 2D transformation calibration
        use_3d = False
        M = calib_data['matrix']
        is_perspective = calib_data['type'] == 'perspective'
        print(f"Using 2D transformation calibration ({calib_data['type']})")
    
    # Load robot data
    print("\nLoading robot data...")
    df = load_robot_data(DATA_FILE)
    robot_ids = extract_robot_ids(df)
    robot_ids = robot_ids[:N_ROBOTS]
    print(f"Found {len(robot_ids)} robots")
    
    # Extract all robot data
    print("Extracting robot states...")
    robot_data = []
    for robot_id in robot_ids:
        states = get_robot_states(df, robot_id, verbose=False)
        robot_data.append({
            'pos_x': states['pos_x'],
            'pos_y': states['pos_y'],
            'qz': states['qz'],
            'qw': states['qw']
        })
    
    n_data_frames = len(robot_data[0]['pos_x'])
    print(f"Data frames: {n_data_frames}")
    
    # Open video
    print("\nOpening video...")
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {VIDEO_FILE}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video FPS: {video_fps}")
    print(f"Video resolution: {video_width}x{video_height}")
    print(f"Video frames: {total_video_frames}")
    
    # Apply crop adjustments to dimensions
    output_width = video_width - CROP_LEFT - CROP_RIGHT
    output_height = video_height - CROP_TOP - CROP_BOTTOM
    
    if output_width <= 0 or output_height <= 0:
        raise ValueError(f"Crop settings result in invalid dimensions: {output_width}x{output_height}")
    
    if CROP_TOP > 0 or CROP_BOTTOM > 0 or CROP_LEFT > 0 or CROP_RIGHT > 0:
        print(f"Crop: top={CROP_TOP}, bottom={CROP_BOTTOM}, left={CROP_LEFT}, right={CROP_RIGHT}")
        print(f"Output resolution: {output_width}x{output_height}")
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, OUTPUT_FPS, (output_width, output_height))
    
    # Generate colors for robots
    colors_mpl = plt.cm.tab10(np.linspace(0, 1, len(robot_ids)))
    colors_bgr = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in colors_mpl]
    
    # Process frames
    print(f"\nProcessing frames {START_FRAME} to {END_FRAME or 'end'}...")
    print(f"Video offset: {VIDEO_FRAME_OFFSET} frames")
    print(f"Data offset: {DATA_FRAME_OFFSET} frames")
    
    # Set start frame with offset
    video_start = START_FRAME + VIDEO_FRAME_OFFSET
    if video_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_start)
    
    frame_idx = START_FRAME
    trail_history = [[] for _ in range(len(robot_ids))]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if END_FRAME is not None and frame_idx >= END_FRAME:
            break
        
        # Crop frame if configured
        if CROP_TOP > 0 or CROP_BOTTOM > 0 or CROP_LEFT > 0 or CROP_RIGHT > 0:
            frame = frame[CROP_TOP:video_height-CROP_BOTTOM, CROP_LEFT:video_width-CROP_RIGHT]
        
        # Update effective video dimensions after cropping
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate corresponding data frame with offset
        data_frame_idx = frame_idx + DATA_FRAME_OFFSET
        
        # Check if we have data for this frame
        if data_frame_idx < 0 or data_frame_idx >= n_data_frames:
            # Skip this frame if data is out of range
            frame_idx += 1
            continue
        
        # Create overlay
        overlay = frame.copy()
        
        # First pass: collect all robot pixel positions for this frame
        robot_pixel_positions = []
        for robot_idx in range(len(robot_ids)):
            data = robot_data[robot_idx]
            
            # Current position (using offset data frame)
            phys_pos = np.array([data['pos_x'][data_frame_idx], data['pos_y'][data_frame_idx]])
            
            # Transform to pixel coordinates (add z=0 for 3D method)
            if use_3d:
                phys_pos_3d = np.array([phys_pos[0], phys_pos[1], 0.0])  # Robots move on ground (z=0)
                pixel_pos = project_3d_to_2d(phys_pos_3d, camera_matrix, rvec, tvec, dist_coeffs)[0]
            else:
                pixel_pos = apply_transform(M, phys_pos, is_perspective)[0]
            
            px, py = int(pixel_pos[0]), int(pixel_pos[1])
            
            # Adjust pixel coordinates if frame was cropped
            if CROP_LEFT > 0 or CROP_TOP > 0:
                px -= CROP_LEFT
                py -= CROP_TOP
            
            # Skip if outside frame
            if px < 0 or px >= frame_width or py < 0 or py >= frame_height:
                robot_pixel_positions.append(None)
            else:
                robot_pixel_positions.append((px, py))
        
        # Draw springs connecting center robot(s) to all others
        if SHOW_SPRINGS:
            # Convert to list if single integer
            center_robots = SPRING_CENTER_ROBOTS if isinstance(SPRING_CENTER_ROBOTS, list) else [SPRING_CENTER_ROBOTS]
            
            if SPRING_ANIMATE_CENTERS and len(center_robots) > 1:
                # Animate through centers with fade
                cycle_length = SPRING_FRAMES_PER_CENTER + SPRING_FADE_FRAMES
                total_cycle = cycle_length * len(center_robots)
                frame_in_cycle = frame_idx % total_cycle
                
                # Determine which center(s) to show and their alpha values
                current_center_idx = frame_in_cycle // cycle_length
                frame_in_current = frame_in_cycle % cycle_length
                
                # Calculate alpha for current center
                if frame_in_current < SPRING_FADE_FRAMES:
                    # Fading in
                    alpha = frame_in_current / SPRING_FADE_FRAMES
                elif frame_in_current < SPRING_FRAMES_PER_CENTER:
                    # Fully visible
                    alpha = 1.0
                else:
                    # Fading out
                    alpha = 1.0 - (frame_in_current - SPRING_FRAMES_PER_CENTER) / SPRING_FADE_FRAMES
                
                # Create a mask and spring overlay for proper alpha blending
                spring_layer = np.zeros_like(overlay)
                mask = np.zeros(overlay.shape[:2], dtype=np.uint8)
                
                # Draw springs for current center
                center_robot_idx = center_robots[current_center_idx]
                if 0 <= center_robot_idx < len(robot_pixel_positions):
                    center_pos = robot_pixel_positions[center_robot_idx]
                    if center_pos is not None:
                        for robot_idx, robot_pos in enumerate(robot_pixel_positions):
                            # When animating, connect current center to all OTHER robots
                            if robot_idx != center_robot_idx and robot_pos is not None:
                                distance = np.linalg.norm(np.array(robot_pos) - np.array(center_pos))
                                amplitude_px = SPRING_AMPLITUDE * distance / 2
                                
                                # Draw spring on the spring layer
                                draw_spring(spring_layer, center_pos, robot_pos, 
                                          SPRING_COLOR, SPRING_THICKNESS, 
                                          SPRING_COILS, amplitude_px)
                                
                                # Draw the same spring on the mask (white = where springs are)
                                draw_spring(mask, center_pos, robot_pos, 
                                          255, SPRING_THICKNESS, 
                                          SPRING_COILS, amplitude_px)
                
                # Apply alpha to the mask
                mask_alpha = (mask * alpha).astype(np.uint8)
                
                # Blend only where springs are drawn
                mask_alpha_3ch = cv2.cvtColor(mask_alpha, cv2.COLOR_GRAY2BGR) / 255.0
                overlay = (overlay * (1 - mask_alpha_3ch) + spring_layer * mask_alpha_3ch).astype(np.uint8)
            else:
                # Show all centers simultaneously (original behavior)
                for center_robot_idx in center_robots:
                    if 0 <= center_robot_idx < len(robot_pixel_positions):
                        center_pos = robot_pixel_positions[center_robot_idx]
                        if center_pos is not None:
                            for robot_idx, robot_pos in enumerate(robot_pixel_positions):
                                # Don't draw spring to itself
                                # Note: When showing all simultaneously, we draw all connections
                                # This means if you have overlapping centers, springs will be drawn multiple times
                                if robot_idx != center_robot_idx and robot_pos is not None:
                                    # Calculate amplitude in pixels (proportional to distance)
                                    distance = np.linalg.norm(np.array(robot_pos) - np.array(center_pos))
                                    amplitude_px = SPRING_AMPLITUDE * distance / 2  # Scale with distance
                                    
                                    draw_spring(overlay, center_pos, robot_pos, 
                                              SPRING_COLOR, SPRING_THICKNESS, 
                                              SPRING_COILS, amplitude_px)
        
        # Second pass: draw robots on top of springs
        for robot_idx in range(len(robot_ids)):
            if robot_pixel_positions[robot_idx] is None:
                continue
            
            px, py = robot_pixel_positions[robot_idx]
            data = robot_data[robot_idx]
            color = colors_bgr[robot_idx]
            
            # Recalculate physical position for arrows (needed for orientation)
            phys_pos = np.array([data['pos_x'][data_frame_idx], data['pos_y'][data_frame_idx]])
            
            # # Draw circle for robot
            # cv2.circle(overlay, (px, py), 15, color, 2)
            # cv2.circle(overlay, (px, py), 3, color, -1)
            
            # # Draw robot number
            # cv2.putText(overlay, str(robot_idx + 1), (px + 20, py),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw orientation arrow
            if SHOW_ARROWS:
                qz = data['qz'][data_frame_idx]
                qw = data['qw'][data_frame_idx]
                theta = 2 * np.arctan2(qz, qw)
                
                # Arrow end point in physical coordinates
                arrow_end_phys = phys_pos + ARROW_LENGTH * np.array([np.cos(theta), np.sin(theta)])
                
                # Transform to pixel coordinates
                if use_3d:
                    arrow_end_phys_3d = np.array([arrow_end_phys[0], arrow_end_phys[1], 0.0])
                    arrow_end_pixel = project_3d_to_2d(arrow_end_phys_3d, camera_matrix, rvec, tvec, dist_coeffs)[0]
                else:
                    arrow_end_pixel = apply_transform(M, arrow_end_phys, is_perspective)[0]
                
                ax, ay = int(arrow_end_pixel[0]), int(arrow_end_pixel[1])
                
                # Draw arrow
                cv2.arrowedLine(overlay, (px, py), (ax, ay), color, 2, tipLength=0.3)
            
            # Update trail history
            if SHOW_TRAILS:
                trail_history[robot_idx].append((px, py))
                if len(trail_history[robot_idx]) > TRAIL_LENGTH:
                    trail_history[robot_idx].pop(0)
                
                # Draw trail
                if len(trail_history[robot_idx]) > 1:
                    pts = np.array(trail_history[robot_idx], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(overlay, [pts], False, color, 2, lineType=cv2.LINE_AA)
        
        # Blend overlay with original frame
        result = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        
        # Add frame information after blending (for full opacity text)
        text_x = 550  # Approximate center, adjust based on text width
        # cv2.putText(result, f'Video Frame: {video_start + frame_idx}', (text_x, 30),
                #    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # cv2.putText(result, f'Data Frame: {data_frame_idx}', (text_x, 65),
                #    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # cv2.putText(result, f'Time: {data_frame_idx * DT:.2f}s', (text_x, 630),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
        # cv2.putText(result, '4x', (text_x, 660),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
        
        # Write frame
        out.write(result)
        
        # Progress update
        if frame_idx % 100 == 0:
            print(f"  Processed frame {frame_idx}...")
        
        frame_idx += 1
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"\n✓ Overlay video saved to: {OUTPUT_VIDEO}")
    print(f"  Total frames processed: {frame_idx - START_FRAME}")
    print("="*80)


if __name__ == '__main__':
    main()
