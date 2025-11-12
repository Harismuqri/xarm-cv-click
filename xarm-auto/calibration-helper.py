"""
Live Camera Calibration Helper
Shows real-time camera view with circle detection overlay
Press 'c' to attempt calibration, 'q' to quit
"""

import cv2
import numpy as np
import PySpin
import pickle
import json
import os

# Load config
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.json")

with open(config_path, 'r') as f:
    CONFIG = json.load(f)

camera_config = CONFIG.get("camera_config", {})
DETECTION_CAMERA_INDEX = camera_config.get("detection_camera_index", 0)

workspace_config = CONFIG.get("workspace", {})
WORKSPACE_WIDTH = workspace_config.get("width", 300)
WORKSPACE_HEIGHT = workspace_config.get("height", 300)

def convert_pyspin_image_to_cv2(image):
    """Convert PySpin image to OpenCV format."""
    if not image.IsValid():
        return None
    img_array = image.GetNDArray()
    if len(img_array.shape) == 2:
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif len(img_array.shape) == 3:
        return img_array
    return None

def detect_circles(frame):
    """Detect circles in frame and return filtered results."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=30
    )

    filtered = []
    if circles is not None:
        for (x, y, r) in np.round(circles[0]).astype("int"):
            if 10 <= r <= 50:
                filtered.append((x, y, r))

    return filtered

def save_calibration(filtered, frame_shape):
    """Create and save homography calibration files."""
    if len(filtered) < 4:
        return False

    # Get 4 circle positions
    image_points = np.array([[x, y] for (x, y, _) in filtered[:4]], dtype=np.float32)
    image_points = sorted(image_points, key=lambda pt: (pt[1], pt[0]))
    top = sorted(image_points[:2], key=lambda pt: pt[0])
    bottom = sorted(image_points[2:], key=lambda pt: pt[0])
    sorted_img_pts = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)

    # Real world points (workspace corners)
    real_pts = np.array([
        [0, WORKSPACE_HEIGHT],
        [WORKSPACE_WIDTH, WORKSPACE_HEIGHT],
        [WORKSPACE_WIDTH, 0],
        [0, 0]
    ], dtype=np.float32)

    # Create first homography (camera → workspace)
    H, _ = cv2.findHomography(sorted_img_pts, real_pts)
    if H is None:
        return False

    # Save homography_auto.pkl
    homography_auto_file = os.path.join(script_dir, "homography_auto.pkl")
    with open(homography_auto_file, "wb") as f:
        pickle.dump(H, f)
    print(f"\n✅ Saved: {homography_auto_file}")

    # Create second homography (workspace → robot)
    detection_corners = np.array([
        [0, 0], [WORKSPACE_WIDTH, 0],
        [WORKSPACE_WIDTH, WORKSPACE_HEIGHT], [0, WORKSPACE_HEIGHT]
    ], dtype=np.float32)

    robot_corners = np.array([
        [88.9, 312], [88.9, 14.7],
        [382, 14.7], [382, 312]
    ], dtype=np.float32)

    H_det_to_robot, _ = cv2.findHomography(detection_corners, robot_corners)
    homography_det_file = os.path.join(script_dir, "homography_det_to_robot.pkl")
    with open(homography_det_file, "wb") as f:
        pickle.dump(H_det_to_robot, f)
    print(f"✅ Saved: {homography_det_file}")

    return True

def main():
    print("\n" + "="*60)
    print("CALIBRATION HELPER - Live Camera View")
    print("="*60)
    print("This tool shows live camera view with circle detection")
    print("\nInstructions:")
    print("1. Place 4 WHITE circles (20-30mm diameter) at workspace corners")
    print("2. Press 'c' to capture and calibrate")
    print("3. Press 'q' to quit")
    print("="*60 + "\n")

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()
    if num_cameras == 0:
        print("[ERROR] No cameras found!")
        system.ReleaseInstance()
        return

    # Use detection camera
    camera_index = 0 if DETECTION_CAMERA_INDEX >= num_cameras else DETECTION_CAMERA_INDEX
    print(f"[INFO] Using camera index {camera_index}")

    cam = cam_list.GetByIndex(camera_index)
    cam.Init()
    serial = cam.TLDevice.DeviceSerialNumber.GetValue()
    print(f"[INFO] Camera Serial: {serial}")

    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    cam.BeginAcquisition()

    window_name = "Calibration Helper - Press 'c' to calibrate, 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 1200, 900)

    print("\n[LIVE] Camera feed started. Position your circles...")

    try:
        while True:
            image = cam.GetNextImage()
            if image.IsIncomplete():
                image.Release()
                continue

            frame = convert_pyspin_image_to_cv2(image)
            image.Release()

            if frame is None:
                continue

            # Detect circles
            filtered = detect_circles(frame)

            # Draw detected circles
            display_frame = frame.copy()
            for i, (x, y, r) in enumerate(filtered):
                # Draw circle outline
                cv2.circle(display_frame, (x, y), r, (0, 255, 0), 2)
                # Draw center point
                cv2.circle(display_frame, (x, y), 3, (0, 0, 255), -1)
                # Draw label
                cv2.putText(display_frame, f"#{i+1}", (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Show count and status
            status_color = (0, 255, 0) if len(filtered) >= 4 else (0, 0, 255)
            status_text = f"Circles detected: {len(filtered)}/4"
            cv2.putText(display_frame, status_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

            if len(filtered) >= 4:
                cv2.putText(display_frame, "Ready! Press 'c' to calibrate", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Need 4 circles for calibration", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Instructions
            cv2.putText(display_frame, "Place WHITE circles at corners | Press 'c' to calibrate | 'q' to quit",
                       (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('c') or key == ord('C'):
                if len(filtered) >= 4:
                    print("\n[INFO] Attempting calibration...")
                    if save_calibration(filtered, frame.shape):
                        print("\n" + "="*60)
                        print("✅ CALIBRATION SUCCESSFUL!")
                        print("="*60)
                        print("Files created:")
                        print("  - homography_auto.pkl")
                        print("  - homography_det_to_robot.pkl")
                        print("\nYou can now run yolo-mouse-v2.py")
                        print("="*60 + "\n")
                        break
                    else:
                        print("\n❌ Calibration failed. Try again.")
                else:
                    print(f"\n❌ Need 4 circles, only found {len(filtered)}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        cam.EndAcquisition()
        cam.DeInit()
        del cam
        cam_list.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")

if __name__ == "__main__":
    main()
