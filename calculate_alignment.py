"""
Calculate coordinate transformation between detection and robot systems.
"""
import numpy as np
import cv2
import json

# Given calibration points: Detection (X,Y) -> Robot (X,Y)
detection_points = np.array([
    [0, 0],
    [300, 0],
    [300, 300],
    [0, 300]
], dtype=np.float32)

robot_points = np.array([
    [88.9, 312],
    [85.1, 12.2],
    [382, 14.7],
    [0, 0]  # This is what we need to calculate
], dtype=np.float32)

# First, let's calculate the missing point (0,300)
# For a rectangular workspace, opposite sides should be parallel
# Vector from (0,0) to (300,0)
v1 = robot_points[1] - robot_points[0]  # (85.1-88.9, 12.2-312) = (-3.8, -299.8)
print(f"Vector (0,0) to (300,0): {v1}")

# Vector from (0,300) to (300,300) should be the same
# (382 - X, 14.7 - Y) = (-3.8, -299.8)
# So: X = 382 + 3.8 = 385.8
#     Y = 14.7 + 299.8 = 314.5

# Verify with the other pair of sides:
# Vector from (0,0) to (0,300)
# Vector from (300,0) to (300,300)
v2 = robot_points[2] - robot_points[1]  # (382-85.1, 14.7-12.2) = (296.9, 2.5)
print(f"Vector (300,0) to (300,300): {v2}")

# (X - 88.9, Y - 312) = (296.9, 2.5)
# So: X = 88.9 + 296.9 = 385.8
#     Y = 312 + 2.5 = 314.5

calculated_point = np.array([385.8, 314.5])
print(f"\nCalculated point for Detection (0,300) -> Robot ({calculated_point[0]}, {calculated_point[1]})")

# Update robot_points with calculated value
robot_points[3] = calculated_point

print("\n" + "="*70)
print("COMPLETE COORDINATE MAPPING:")
print("="*70)
for i, (det, rob) in enumerate(zip(detection_points, robot_points)):
    print(f"Detection ({det[0]:>5.1f}, {det[1]:>5.1f}) -> Robot ({rob[0]:>6.1f}, {rob[1]:>6.1f})")

# Calculate homography matrix: Detection -> Robot
H_det_to_robot, _ = cv2.findHomography(detection_points, robot_points)

print("\n" + "="*70)
print("HOMOGRAPHY MATRIX (Detection -> Robot):")
print("="*70)
print(H_det_to_robot)

# Verify the transformation
print("\n" + "="*70)
print("VERIFICATION (applying transformation to detection points):")
print("="*70)
for det_pt in detection_points:
    # Apply homography
    det_homogeneous = np.array([[det_pt[0], det_pt[1]]], dtype=np.float32).reshape(-1, 1, 2)
    robot_transformed = cv2.perspectiveTransform(det_homogeneous, H_det_to_robot).reshape(-1, 2)

    print(f"Detection ({det_pt[0]:>5.1f}, {det_pt[1]:>5.1f}) -> "
          f"Robot ({robot_transformed[0][0]:>6.1f}, {robot_transformed[0][1]:>6.1f})")

# Test some intermediate points
print("\n" + "="*70)
print("TEST POINTS:")
print("="*70)
test_points = np.array([
    [150, 150],  # Center
    [100, 100],
    [200, 200],
    [0, 150],
    [150, 0]
], dtype=np.float32)

for test_pt in test_points:
    test_homogeneous = np.array([[test_pt[0], test_pt[1]]], dtype=np.float32).reshape(-1, 1, 2)
    robot_result = cv2.perspectiveTransform(test_homogeneous, H_det_to_robot).reshape(-1, 2)

    print(f"Detection ({test_pt[0]:>5.1f}, {test_pt[1]:>5.1f}) -> "
          f"Robot ({robot_result[0][0]:>6.1f}, {robot_result[0][1]:>6.1f})")

# Save the homography matrix
np.save("homography_det_to_robot.npy", H_det_to_robot)
print("\n✅ Homography matrix saved to 'homography_det_to_robot.npy'")

# Create updated config with transformation matrix
print("\n" + "="*70)
print("CONFIGURATION UPDATE:")
print("="*70)
print("The homography matrix has been calculated and saved.")
print("To use this in your code, replace the simple offset calculation with:")
print("""
# Load homography
H_det_to_robot = np.load("homography_det_to_robot.npy")

# Transform detection coordinates to robot coordinates
det_pt = np.array([[click_x, click_y]], dtype=np.float32).reshape(-1, 1, 2)
robot_pt = cv2.perspectiveTransform(det_pt, H_det_to_robot).reshape(-1, 2)
robot_x, robot_y = robot_pt[0][0], robot_pt[0][1]
""")

# Analyze the transformation
print("\n" + "="*70)
print("TRANSFORMATION ANALYSIS:")
print("="*70)

# Check rotation angle
# Average the rotation from the homography
angle_rad = np.arctan2(H_det_to_robot[1,0], H_det_to_robot[0,0])
angle_deg = np.degrees(angle_rad)
print(f"Approximate rotation: {angle_deg:.1f} degrees")

# Check scale
scale_x = np.sqrt(H_det_to_robot[0,0]**2 + H_det_to_robot[1,0]**2)
scale_y = np.sqrt(H_det_to_robot[0,1]**2 + H_det_to_robot[1,1]**2)
print(f"Scale factors: X={scale_x:.3f}, Y={scale_y:.3f}")

# Translation
print(f"Translation: ({H_det_to_robot[0,2]:.1f}, {H_det_to_robot[1,2]:.1f})")

print("\n" + "="*70)
print("The transformation includes rotation, scaling, and translation.")
print("This explains why simple X,Y offsets weren't working correctly!")
print("="*70)
