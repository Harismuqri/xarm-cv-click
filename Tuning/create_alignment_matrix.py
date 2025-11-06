"""
Create homography matrix for detection to robot coordinate transformation
Based on calibration data with corrected corner mappings
"""
import numpy as np
import cv2
import pickle

# Detection coordinates (300x300mm workspace)
detection_corners = np.array([
    [0, 0],      # BL - Bottom Left
    [300, 0],    # BR - Bottom Right  
    [300, 300],  # TR - Top Right
    [0, 300]     # TL - Top Left
], dtype=np.float32)

# Robot coordinates (actual measured positions in mm)
robot_corners = np.array([
    [88.9, 312],   # BL - Bottom Left
    [88.9, 14.7],  # BR - Bottom Right
    [382, 14.7],   # TR - Top Right
    [382, 312]     # TL - Top Left
], dtype=np.float32)

# Calculate homography matrix (detection → robot)
H_det_to_robot, status = cv2.findHomography(detection_corners, robot_corners, cv2.RANSAC, 5.0)

# Save the homography matrix
with open("homography_det_to_robot.pkl", "wb") as f:
    pickle.dump(H_det_to_robot, f)

print("="*60)
print("HOMOGRAPHY MATRIX CREATED")
print("="*60)
print("\nDetection → Robot transformation matrix:")
print(H_det_to_robot)
print("\n" + "="*60)
print("VERIFICATION")
print("="*60)

# Verify the transformation
for i, (det_pt, robot_pt) in enumerate(zip(detection_corners, robot_corners)):
    det_pt_homog = np.array([[det_pt]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(det_pt_homog, H_det_to_robot)
    result = transformed[0][0]
    
    corner_names = ["BL (0,0)", "BR (300,0)", "TR (300,300)", "TL (0,300)"]
    print(f"\n{corner_names[i]}:")
    print(f"  Detection: ({det_pt[0]:.1f}, {det_pt[1]:.1f})")
    print(f"  Expected:  ({robot_pt[0]:.1f}, {robot_pt[1]:.1f})")
    print(f"  Result:    ({result[0]:.1f}, {result[1]:.1f})")
    
    error = np.linalg.norm(result - robot_pt)
    print(f"  Error:     {error:.3f} mm")

print("\n" + "="*60)
print("✅ File saved: homography_det_to_robot.pkl")
print("   Copy this file to the same directory as your robot control script")
print("="*60)