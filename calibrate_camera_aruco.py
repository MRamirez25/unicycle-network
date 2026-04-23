#!/usr/bin/env python3
"""
Camera calibration using ArUco marker detection with full 3D coordinates.

This script:
1. Detects ArUco markers in video frame
2. Extracts marker corner positions in pixels (2D)
3. Loads known 3D positions of marker corners from YAML
4. Uses solvePnP to estimate camera pose (extrinsic parameters)
5. Creates projection function from 3D world coordinates to 2D pixel coordinates
6. Saves calibration for use in overlay scripts

This properly handles the 3D-to-2D projection using camera intrinsics and extrinsics.
"""

import os
import sys
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import pickle

#=============================================================================
# CONFIGURATION
#=============================================================================

# Video file (actual camera recording)
VIDEO_FILE = "video_edits/tune-period-80-retake4(1).mkv"

# Which video frame to use for detection (0 = first frame)
VIDEO_FRAME_NUMBER = 0

# ArUco marker configuration file (YAML format with physical coordinates)
ARUCO_CONFIG_FILE = "mm_arena.yml"

# ArUco marker settings
ARUCO_DICT = cv2.aruco.DICT_ARUCO_MIP_36h12  # Must match the dict in YAML file

# Camera intrinsic parameters (if known, otherwise will estimate)
# These should be obtained from camera calibration
# Format: fx, fy (focal lengths), cx, cy (principal point)
CAMERA_MATRIX = None  # Will be estimated if None
DIST_COEFFS = None    # Distortion coefficients (will assume zero if None)

# If you have camera calibration, set these:
# CAMERA_MATRIX = np.array([[fx, 0, cx],
#                           [0, fy, cy],
#                           [0,  0,  1]], dtype=float)
# DIST_COEFFS = np.array([k1, k2, p1, p2, k3], dtype=float)

# Refinement settings
REFINE_INTRINSICS = True  # If True, optimize fx, fy, cx, cy to minimize reprojection error
                          # If False, use initial estimates without refinement

# Output file for transformation
OUTPUT_FILE = "camera_transform_aruco.pkl"

# Visualization settings
SHOW_DETECTION = True  # Show detected markers on frame
SAVE_DETECTION_IMAGE = True  # Save image with detected markers

#=============================================================================
# ArUco Detection
#=============================================================================

def load_aruco_markers_from_yaml(yaml_file):
    """
    Load ArUco marker physical 3D coordinates from YAML file.
    
    Returns:
        marker_coords: Dictionary mapping marker_id -> [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3), (x4,y4,z4)]
        aruco_dict_name: Name of ArUco dictionary from file
    """
    # Use OpenCV's FileStorage to read the YAML file (handles %YAML:1.0 format)
    fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
    
    aruco_dict_name = fs.getNode('aruco_bc_dict').string()
    n_markers = int(fs.getNode('aruco_bc_nmarkers').real())
    
    marker_coords = {}
    
    # Read markers node
    markers_node = fs.getNode('aruco_bc_markers')
    
    for i in range(n_markers):
        marker_node = markers_node.at(i)
        marker_id = int(marker_node.getNode('id').real())
        
        # Read corners - keep 3D coordinates
        corners_node = marker_node.getNode('corners')
        corners_3d = []
        
        for j in range(4):  # 4 corners per marker
            corner_node = corners_node.at(j)
            x = corner_node.at(0).real()
            y = corner_node.at(1).real()
            z = corner_node.at(2).real()
            corners_3d.append((x, y, z))
        
        marker_coords[marker_id] = corners_3d
    
    fs.release()
    
    print(f"Loaded {len(marker_coords)} markers from {yaml_file}")
    print(f"ArUco dictionary: {aruco_dict_name}")
    print(f"Marker IDs: {list(marker_coords.keys())}")
    
    return marker_coords, aruco_dict_name


def detect_aruco_markers(frame, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    """
    Detect ArUco markers in frame and return their corners.
    
    Returns:
        markers_dict: Dictionary mapping marker_id -> corners (4x2 array)
    """
    # Load ArUco dictionary and detector parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Create detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    markers_dict = {}
    
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            # corners[i] is a 1x4x2 array, reshape to 4x2
            marker_corners = corners[i].reshape(4, 2)
            markers_dict[marker_id] = marker_corners
    
    return markers_dict, corners, ids


def visualize_detected_markers(frame, markers_dict, ids, corners):
    """Draw detected markers on frame."""
    display_frame = frame.copy()
    
    # Draw detected markers
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
        
        # Add marker IDs as text
        for marker_id, marker_corners in markers_dict.items():
            # Get center of marker
            center = marker_corners.mean(axis=0).astype(int)
            cv2.putText(display_frame, f'ID: {marker_id}', 
                       tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 255, 0), 2)
    
    return display_frame


#=============================================================================
# Calibration using 3D-2D correspondences
#=============================================================================

def estimate_camera_pose(object_points_3d, image_points_2d, camera_matrix=None, dist_coeffs=None, image_size=None):
    """
    Estimate camera pose using solvePnP with 3D-2D point correspondences.
    
    Args:
        object_points_3d: Nx3 array of 3D points in world coordinates
        image_points_2d: Nx2 array of corresponding 2D points in image
        camera_matrix: 3x3 camera intrinsic matrix (if None, will estimate)
        dist_coeffs: Distortion coefficients (if None, assumes zero distortion)
        image_size: (width, height) for camera matrix estimation
        
    Returns:
        camera_matrix: 3x3 intrinsic matrix
        dist_coeffs: Distortion coefficients  
        rvec: Rotation vector (camera orientation)
        tvec: Translation vector (camera position)
    """
    object_points_3d = np.array(object_points_3d, dtype=np.float32)
    image_points_2d = np.array(image_points_2d, dtype=np.float32)
    
    # If no camera matrix provided, estimate a reasonable one
    if camera_matrix is None:
        if image_size is None:
            raise ValueError("Need image_size to estimate camera matrix")
        width, height = image_size
        # Assume focal length is approx image width
        fx = fy = width
        cx = width / 2.0
        cy = height / 2.0
        camera_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]], dtype=np.float32)
        print("Estimated camera matrix (no calibration provided):")
        print(camera_matrix)
    
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float32)
    
    # Solve PnP to get camera pose
    success, rvec, tvec = cv2.solvePnP(object_points_3d, image_points_2d,
                                        camera_matrix, dist_coeffs,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
    
    if not success:
        raise RuntimeError("solvePnP failed to find camera pose")
    
    return camera_matrix, dist_coeffs, rvec, tvec


def project_3d_to_2d(points_3d, camera_matrix, rvec, tvec, dist_coeffs=None):
    """
    Project 3D points to 2D image coordinates using camera parameters.
    
    Args:
        points_3d: Nx3 array of 3D points in world coordinates
        camera_matrix: 3x3 camera intrinsic matrix
        rvec: Rotation vector
        tvec: Translation vector
        dist_coeffs: Distortion coefficients
        
    Returns:
        points_2d: Nx2 array of 2D image points
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
    
    points_3d = np.array(points_3d, dtype=np.float32).reshape(-1, 3)
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)


def refine_camera_calibration(object_points_3d, image_points_2d, 
                               camera_matrix, dist_coeffs, rvec, tvec,
                               refine_intrinsics=True):
    """
    Refine camera calibration by optimizing both intrinsics and extrinsics.
    
    Uses cv2.calibrateCamera which optimizes fx, fy, cx, cy along with pose.
    
    Args:
        object_points_3d: Nx3 array of 3D points in world coordinates
        image_points_2d: Nx2 array of corresponding 2D points in image
        camera_matrix: Initial 3x3 camera intrinsic matrix
        dist_coeffs: Initial distortion coefficients
        rvec: Initial rotation vector
        tvec: Initial translation vector
        refine_intrinsics: If True, optimize fx, fy, cx, cy; if False, keep them fixed
        
    Returns:
        Refined camera_matrix, dist_coeffs, rvec, tvec
    """
    object_points_3d = np.array(object_points_3d, dtype=np.float32)
    image_points_2d = np.array(image_points_2d, dtype=np.float32)
    
    # calibrateCamera expects lists of arrays (for multiple images)
    # We only have one "view" with all our correspondences
    object_points_list = [object_points_3d]
    image_points_list = [image_points_2d]
    
    # Get image size from camera matrix
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    image_size = (int(cx * 2), int(cy * 2))  # Approximate from principal point
    
    # Set calibration flags
    flags = 0
    if not refine_intrinsics:
        # Fix the intrinsics
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    else:
        # Allow intrinsics to be optimized
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    
    # Fix distortion to zero (assuming pinhole model)
    flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    flags |= cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
    flags |= cv2.CALIB_ZERO_TANGENT_DIST
    
    print(f"  Refining calibration (intrinsics={'optimized' if refine_intrinsics else 'fixed'})...")
    ret, camera_matrix_refined, dist_coeffs_refined, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list, image_points_list, image_size,
        camera_matrix.copy(), dist_coeffs.copy(),
        flags=flags
    )
    
    print(f"  Refinement RMS error: {ret:.3f} pixels")
    
    return camera_matrix_refined, dist_coeffs_refined, rvecs[0], tvecs[0]


#=============================================================================
# Main
#=============================================================================

def main():
    """Main execution."""
    
    print("\n" + "="*80)
    print("ARUCO MARKER CAMERA CALIBRATION (3D-to-2D)")
    print("="*80)
    print(f"Video: {VIDEO_FILE}")
    print(f"Frame: {VIDEO_FRAME_NUMBER}")
    print(f"Config: {ARUCO_CONFIG_FILE}")
    print("Method: solvePnP (camera pose estimation)")
    print("="*80 + "\n")
    
    # Load marker coordinates from YAML
    print("Loading ArUco marker coordinates...")
    MARKER_PHYSICAL_COORDS, aruco_dict_name = load_aruco_markers_from_yaml(ARUCO_CONFIG_FILE)
    print()
    
    # Load video frame
    print("Loading video frame...")
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {VIDEO_FILE}")
    
    # Set frame position
    if VIDEO_FRAME_NUMBER > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_FRAME_NUMBER)
    
    ret, frame_bgr = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {VIDEO_FRAME_NUMBER}")
    
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    print(f"Frame shape: {frame.shape}")
    
    # Detect ArUco markers
    print("\nDetecting ArUco markers...")
    markers_dict, corners_list, ids = detect_aruco_markers(frame, ARUCO_DICT)
    
    if not markers_dict:
        print("ERROR: No ArUco markers detected!")
        print("Tips:")
        print("  - Make sure markers are visible and well-lit")
        print("  - Check that ARUCO_DICT matches your marker type")
        print("  - Try a different frame with VIDEO_FRAME_NUMBER")
        return
    
    print(f"Detected {len(markers_dict)} markers: {list(markers_dict.keys())}")
    
    # Match detected markers with known physical 3D coordinates
    print("\nMatching markers with physical 3D coordinates...")
    physical_points_3d = []
    pixel_points_2d = []
    
    for marker_id, pixel_corners in markers_dict.items():
        if marker_id in MARKER_PHYSICAL_COORDS:
            phys_corners_3d = MARKER_PHYSICAL_COORDS[marker_id]
            
            # Add all 4 corners (3D world coords -> 2D image coords)
            for phys_corner_3d, pixel_corner in zip(phys_corners_3d, pixel_corners):
                physical_points_3d.append(phys_corner_3d)
                pixel_points_2d.append(pixel_corner)
            
            print(f"  Marker {marker_id}: matched 4 corners")
        else:
            print(f"  Marker {marker_id}: WARNING - no physical coordinates defined (skipped)")
    
    if len(physical_points_3d) < 4:
        print(f"\nERROR: Only found {len(physical_points_3d)} point correspondences")
        print("Need at least 4 for solvePnP")
        print("\nDetected markers without coordinates:")
        for marker_id in markers_dict.keys():
            if marker_id not in MARKER_PHYSICAL_COORDS:
                print(f"  Marker {marker_id}")
        return
    
    physical_points_3d = np.array(physical_points_3d)
    pixel_points_2d = np.array(pixel_points_2d)
    
    print(f"\nTotal correspondences: {len(physical_points_3d)} points (3D -> 2D)")
    
    # Estimate camera pose using 3D-2D correspondences
    print("\nEstimating camera pose with solvePnP...")
    image_size = (frame.shape[1], frame.shape[0])  # width, height
    camera_matrix, dist_coeffs, rvec, tvec = estimate_camera_pose(
        physical_points_3d, pixel_points_2d,
        CAMERA_MATRIX, DIST_COEFFS, image_size
    )
    
    print("\nCamera Matrix (intrinsics):")
    print(camera_matrix)
    print("\nRotation vector (rvec):")
    print(rvec.T)
    print("\nTranslation vector (tvec):")
    print(tvec.T)
    
    # Optionally refine the calibration by optimizing intrinsics
    if REFINE_INTRINSICS:
        print("\n" + "="*80)
        print("REFINING CAMERA CALIBRATION")
        print("="*80)
        
        camera_matrix_refined, dist_coeffs_refined, rvec_refined, tvec_refined = refine_camera_calibration(
            physical_points_3d, pixel_points_2d,
            camera_matrix, dist_coeffs, rvec, tvec,
            refine_intrinsics=True
        )
        
        print("\nRefined Camera Matrix:")
        print(camera_matrix_refined)
        print("\nChange in intrinsics:")
        print(f"  fx: {camera_matrix[0,0]:.2f} -> {camera_matrix_refined[0,0]:.2f} (Δ={camera_matrix_refined[0,0]-camera_matrix[0,0]:.2f})")
        print(f"  fy: {camera_matrix[1,1]:.2f} -> {camera_matrix_refined[1,1]:.2f} (Δ={camera_matrix_refined[1,1]-camera_matrix[1,1]:.2f})")
        print(f"  cx: {camera_matrix[0,2]:.2f} -> {camera_matrix_refined[0,2]:.2f} (Δ={camera_matrix_refined[0,2]-camera_matrix[0,2]:.2f})")
        print(f"  cy: {camera_matrix[1,2]:.2f} -> {camera_matrix_refined[1,2]:.2f} (Δ={camera_matrix_refined[1,2]-camera_matrix[1,2]:.2f})")
        
        # Use refined parameters
        camera_matrix = camera_matrix_refined
        dist_coeffs = dist_coeffs_refined
        rvec = rvec_refined
        tvec = tvec_refined
        print("="*80)
    
    # Validate by reprojecting 3D points
    print("\nValidation (3D world -> predicted 2D image):")
    predicted_pixels = project_3d_to_2d(physical_points_3d, camera_matrix, rvec, tvec, dist_coeffs)
    
    errors = []
    for i, (phys_3d, actual_pix, pred_pix) in enumerate(zip(physical_points_3d, pixel_points_2d, predicted_pixels), 1):
        error = np.linalg.norm(actual_pix - pred_pix)
        errors.append(error)
        print(f"  Point {i}: 3D ({phys_3d[0]:.3f}, {phys_3d[1]:.3f}, {phys_3d[2]:.3f}) -> "
              f"Detected ({actual_pix[0]:.1f}, {actual_pix[1]:.1f}) "
              f"Projected ({pred_pix[0]:.1f}, {pred_pix[1]:.1f}) "
              f"Error: {error:.2f} px")
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    print(f"\nMean reprojection error: {mean_error:.2f} pixels")
    print(f"Max reprojection error: {max_error:.2f} pixels")
    
    # Save calibration with camera parameters
    transform_data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvec': rvec,
        'tvec': tvec,
        'physical_points_3d': physical_points_3d,
        'pixel_points_2d': pixel_points_2d,
        'mean_error': mean_error,
        'max_error': max_error,
        'video_file': VIDEO_FILE,
        'data_file': 'aruco_markers',
        'frame_shape': frame.shape,
        'marker_ids': list(markers_dict.keys()),
        'method': 'aruco_3d',
        'type': 'camera_pose'
    }
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(transform_data, f)
    
    print(f"\n✓ Transformation saved to: {OUTPUT_FILE}")
    print("="*80)
    
    # Visualize results
    if SHOW_DETECTION:
        print("\nGenerating visualization...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Left: detected markers
        display_frame = visualize_detected_markers(frame, markers_dict, ids, corners_list)
        ax1.imshow(display_frame)
        ax1.set_title(f'Detected ArUco Markers (Frame {VIDEO_FRAME_NUMBER})', fontsize=14)
        ax1.axis('off')
        
        # Right: calibration validation
        ax2.imshow(frame)
        ax2.set_title(f'Calibration Validation (Mean error: {mean_error:.2f} px)', fontsize=14)
        ax2.axis('off')
        
        # Draw actual and predicted points
        for actual, pred in zip(pixel_points_2d, predicted_pixels):
            # Actual (red)
            ax2.plot(actual[0], actual[1], 'ro', markersize=8)
            # Predicted (green)
            ax2.plot(pred[0], pred[1], 'gx', markersize=10, markeredgewidth=2)
            # Error line
            ax2.plot([actual[0], pred[0]], [actual[1], pred[1]], 
                    'y-', linewidth=1, alpha=0.5)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='r', 
                   markersize=8, label='Detected'),
            Line2D([0], [0], marker='x', color='g', markersize=10, 
                   markeredgewidth=2, label='Predicted'),
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        
        if SAVE_DETECTION_IMAGE:
            plt.savefig('aruco_calibration_result.png', dpi=150, bbox_inches='tight')
            print("✓ Visualization saved to: aruco_calibration_result.png")
        
        plt.show()


if __name__ == '__main__':
    main()
