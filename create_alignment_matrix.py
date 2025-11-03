"""
Create alignment matrix for detection to robot coordinate transformation.
Run this to generate the homography matrix for proper coordinate alignment.
"""
import cv2
import numpy as np
import pickle
import json

print("="*80)
print("DETECTION TO ROBOT COORDINATE ALIGNMENT")
print("="*80)

# Calibration points: Detection (X,Y) -> Robot (X,Y)
detection_points = np.array([
    [0, 0],
    [300, 0],
    [300, 300],
    [0, 300]
], dtype=np.float32)

# First, calculate the missing point (0,300)
# Using rectangular workspace assumption
# Vector from (0,0) to (300,0): (-3.8, -299.8)
# Vector from (0,300) to (300,300) should be the same
# (382 - X, 14.7 - Y) = (-3.8, -299.8)
# X = 385.8, Y = 314.5

robot_points = np.array([
    [88.9, 312.0],      # (0,0)
    [85.1, 12.2],       # (300,0)
    [382.0, 14.7],      # (300,300)
    [385.8, 314.5]      # (0,300) - calculated
], dtype=np.float32)

print("\nCalibration Points:")
print("-" * 80)
for i, (det, rob) in enumerate(zip(detection_points, robot_points)):
    print(f"  Detection ({det[0]:>5.0f}, {det[1]:>5.0f}) mm  ->  "
          f"Robot ({rob[0]:>6.1f}, {rob[1]:>6.1f}) mm")

# Calculate homography: Detection coordinates -> Robot coordinates
H_det_to_robot, status = cv2.findHomography(detection_points, robot_points)

print("\n" + "="*80)
print("HOMOGRAPHY MATRIX (Detection -> Robot):")
print("="*80)
print(H_det_to_robot)
print()

# Verify the transformation
print("="*80)
print("VERIFICATION (Testing the transformation):")
print("="*80)
max_error = 0
for i, det_pt in enumerate(detection_points):
    det_homogeneous = np.array([[det_pt[0], det_pt[1]]], dtype=np.float32).reshape(-1, 1, 2)
    robot_transformed = cv2.perspectiveTransform(det_homogeneous, H_det_to_robot).reshape(-1, 2)

    expected = robot_points[i]
    actual = robot_transformed[0]
    error = np.linalg.norm(expected - actual)
    max_error = max(max_error, error)

    print(f"  Input:    Detection ({det_pt[0]:>5.0f}, {det_pt[1]:>5.0f}) mm")
    print(f"  Output:   Robot     ({actual[0]:>6.2f}, {actual[1]:>6.2f}) mm")
    print(f"  Expected: Robot     ({expected[0]:>6.2f}, {expected[1]:>6.2f}) mm")
    print(f"  Error:    {error:.3f} mm")
    print()

print(f"✅ Maximum transformation error: {max_error:.3f} mm")

# Test some intermediate points
print("\n" + "="*80)
print("SAMPLE TRANSFORMATIONS (Testing intermediate points):")
print("="*80)

test_points = [
    ("Center", [150, 150]),
    ("Quarter 1", [75, 75]),
    ("Quarter 2", [225, 75]),
    ("Quarter 3", [225, 225]),
    ("Quarter 4", [75, 225]),
]

for name, test_pt in test_points:
    test_pt = np.array(test_pt, dtype=np.float32)
    test_homogeneous = np.array([[test_pt[0], test_pt[1]]], dtype=np.float32).reshape(-1, 1, 2)
    robot_result = cv2.perspectiveTransform(test_homogeneous, H_det_to_robot).reshape(-1, 2)

    print(f"  {name:12s}: Detection ({test_pt[0]:>5.0f}, {test_pt[1]:>5.0f}) mm  ->  "
          f"Robot ({robot_result[0][0]:>6.1f}, {robot_result[0][1]:>6.1f}) mm")

# Save the homography matrix
with open("homography_det_to_robot.pkl", "wb") as f:
    pickle.dump(H_det_to_robot, f)

print("\n" + "="*80)
print("✅ Homography matrix saved to: homography_det_to_robot.pkl")
print("="*80)

# Analyze the transformation
print("\n" + "="*80)
print("TRANSFORMATION ANALYSIS:")
print("="*80)

# Decompose to understand the transformation
# Calculate approximate rotation
U, S, Vt = np.linalg.svd(H_det_to_robot[:2, :2])
rotation_matrix = U @ Vt
angle_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
angle_deg = np.degrees(angle_rad)

print(f"  Approximate rotation:  {angle_deg:.1f} degrees")
print(f"  Scale factors:         X={S[0]:.3f}, Y={S[1]:.3f}")
print(f"  Translation at origin: ({H_det_to_robot[0,2]:.1f}, {H_det_to_robot[1,2]:.1f}) mm")
print()
print("  ⚠️  This transformation includes rotation, scaling, AND translation!")
print("  ⚠️  Simple X,Y offsets cannot handle this - that's why alignment was wrong!")

# Create instructions
print("\n" + "="*80)
print("HOW TO USE THIS IN YOUR CODE:")
print("="*80)
print("""
In xarm-motion-v1.2.py, replace the simple offset code:

    # OLD CODE (wrong):
    robot_x = click_x + self.offset_x
    robot_y = click_y + self.offset_y

    # NEW CODE (correct):
    import pickle
    import cv2
    import numpy as np

    # Load once during initialization:
    with open("homography_det_to_robot.pkl", "rb") as f:
        self.H_det_to_robot = pickle.load(f)

    # Use for each click:
    det_pt = np.array([[click_x, click_y]], dtype=np.float32).reshape(-1, 1, 2)
    robot_pt = cv2.perspectiveTransform(det_pt, self.H_det_to_robot).reshape(-1, 2)
    robot_x = robot_pt[0][0]
    robot_y = robot_pt[0][1]
""")

print("="*80)
print("✅ SETUP COMPLETE!")
print("="*80)
print("\nNext steps:")
print("  1. Review the analysis above")
print("  2. Update xarm-motion-v1.2.py to use the homography matrix")
print("  3. Test with clicks to verify proper alignment")
print("="*80)
