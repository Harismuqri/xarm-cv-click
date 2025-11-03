"""
Robot Keyboard Control with YOLO Detection Visual
Combines keyboard control with object detection visualization
"""

import cv2
import numpy as np
import pickle
from ultralytics import YOLO
import PySpin
import time
import json
import os
from xarm.wrapper import XArmAPI
import sys

# Load YOLO model
model = YOLO("D:\\2. yolo\\train30\\weights\\best.pt")
model.overrides['verbose'] = False

class RobotKeyboardDetection:
    """Robot control with keyboard while showing YOLO detection."""

    def __init__(self, config_path="config.json"):
        """Initialize robot and camera."""
        self.config = self.load_config(config_path)
        self.robot_ip = self.config.get("robot_ip", "192.168.1.151")
        self._arm = None

        # Movement parameters
        self.tcp_speed = self.config.get("tcp_speed", 200)
        self.tcp_acc = self.config.get("tcp_acc", 2000)
        self.angle_speed = self.config.get("angle_speed", 20)
        self.angle_acc = self.config.get("angle_acc", 500)

        # Load offsets from config
        click_config = self.config.get("click_control", {})
        self.offset_x = click_config.get("coordinate_offset_x", 0)
        self.offset_y = click_config.get("coordinate_offset_y", -150)

        # Movement step sizes (mm)
        self.step_size = 10
        self.fine_step = 1
        self.large_step = 50

        # Current position
        self.current_x = None
        self.current_y = None
        self.current_z = None

        # Home position
        self.home_position = [1.5, 6.3, 45.5, -0.4, 41.0, -7.0]

        # Camera
        self.camera = None
        self.camera_system = None
        self.H = None
        self.H_inv = None

        self.connect_robot()
        self.initialize_robot()
        self.initialize_camera()

    def load_config(self, path="config.json"):
        """Load configuration."""
        try:
            if not os.path.isabs(path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(base_dir, path)
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Config] Failed to load: {e}")
            return {
                "robot_ip": "192.168.1.151",
                "tcp_speed": 200,
                "tcp_acc": 2000,
                "click_control": {
                    "coordinate_offset_x": 0,
                    "coordinate_offset_y": -150
                }
            }

    def connect_robot(self):
        """Connect to robot."""
        while True:
            try:
                print(f"[Robot] Connecting to {self.robot_ip}...")
                self._arm = XArmAPI(self.robot_ip, baud_checkset=False)
                print("[Robot] ✅ Connected")
                break
            except Exception as e:
                print(f"[Robot] Connection failed: {e}. Retrying in 5s...")
                time.sleep(5)

    def initialize_robot(self):
        """Initialize robot."""
        try:
            print("[Robot] Initializing...")
            self._arm.clean_warn()
            self._arm.clean_error()
            self._arm.motion_enable(True)
            self._arm.set_mode(0)
            self._arm.set_state(0)
            time.sleep(0.5)
            self.update_current_position()
            print("[Robot] ✅ Ready")
        except Exception as e:
            print(f"[Robot] Initialization error: {e}")

    def initialize_camera(self):
        """Initialize camera and load homography."""
        try:
            print("[Camera] Initializing...")
            self.camera_system = PySpin.System.GetInstance()
            cam_list = self.camera_system.GetCameras()

            if cam_list.GetSize() == 0:
                print("[Camera] ❌ No camera found")
                return False

            self.camera = cam_list.GetByIndex(0)
            self.camera.Init()
            self.camera.BeginAcquisition()

            # Load homography
            for filename in ["homography_auto.pkl", "homography_calibration.pkl"]:
                if os.path.exists(filename):
                    with open(filename, "rb") as f:
                        self.H = pickle.load(f)
                    self.H_inv = np.linalg.inv(self.H)
                    print(f"[Camera] ✅ Loaded {filename}")
                    return True

            print("[Camera] ⚠️  No homography found")
            return False
        except Exception as e:
            print(f"[Camera] Error: {e}")
            return False

    def update_current_position(self):
        """Get current robot position."""
        try:
            code, pos = self._arm.get_position()
            if code == 0:
                self.current_x = pos[0]
                self.current_y = pos[1]
                self.current_z = pos[2]
        except Exception as e:
            print(f"[ERROR] Failed to get position: {e}")

    def move_relative(self, dx=0, dy=0, dz=0):
        """Move robot relative to current position."""
        if self.current_x is None:
            self.update_current_position()

        new_x = self.current_x + dx
        new_y = self.current_y + dy
        new_z = self.current_z + dz

        return self.move_to_position(new_x, new_y, new_z)

    def move_to_position(self, x, y, z):
        """Move robot to position."""
        try:
            code, current_pos = self._arm.get_position()
            if code == 0:
                roll, pitch, yaw = current_pos[3], current_pos[4], current_pos[5]
            else:
                roll, pitch, yaw = 180, 0, 0

            code = self._arm.set_position(
                x, y, z, roll, pitch, yaw,
                speed=self.tcp_speed,
                mvacc=self.tcp_acc,
                radius=0,
                wait=True
            )

            if code == 0:
                self.current_x = x
                self.current_y = y
                self.current_z = z
                return True
            return False
        except Exception as e:
            print(f"[MOVE] Error: {e}")
            return False

    def go_home(self):
        """Go to home position."""
        print("[Robot] Going home...")
        try:
            code = self._arm.set_servo_angle(
                angle=self.home_position,
                speed=self.angle_speed,
                mvacc=self.angle_acc,
                wait=True
            )
            if code == 0:
                self.update_current_position()
                print("[Robot] ✅ Home")
                return True
        except Exception as e:
            print(f"[Robot] Error: {e}")
        return False

    def get_camera_frame(self):
        """Get camera frame."""
        if self.camera is None:
            return None
        try:
            image = self.camera.GetNextImage()
            if image.IsIncomplete():
                image.Release()
                return None

            img_array = image.GetNDArray()
            image.Release()

            if len(img_array.shape) == 2:
                return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            return img_array
        except:
            return None

    def transform_points(self, points, H):
        """Transform points using homography."""
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, H)
        return warped.reshape(-1, 2)

    def is_inside_box(self, pts, width=300, height=300):
        """Check if points inside workspace."""
        x, y = pts[:, 0], pts[:, 1]
        return np.all((x >= 0) & (x <= width) & (y >= 0) & (y <= height))

    def get_angle(self, obb_pts):
        """Get angle from OBB points."""
        v1 = obb_pts[1] - obb_pts[0]
        v2 = obb_pts[2] - obb_pts[1]
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        long_vec = v1 if len1 >= len2 else v2
        angle_rad = np.arctan2(long_vec[1], long_vec[0])
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 180
        return angle_deg

    def draw_workspace_box(self, frame):
        """Draw workspace boundary."""
        if self.H_inv is None:
            return

        box_real = np.array([
            [0, 0],
            [300, 0],
            [300, 300],
            [0, 300]
        ], dtype=np.float32).reshape(-1, 1, 2)
        box_img = cv2.perspectiveTransform(box_real, self.H_inv).reshape(-1, 2).astype(int)
        cv2.polylines(frame, [box_img], isClosed=True, color=(0, 0, 0), thickness=2)  # Thinner border

    def draw_robot_position(self, frame):
        """Draw robot position on frame."""
        if self.H_inv is None or self.current_x is None:
            return

        # Transform robot position to camera coordinates
        camera_x = self.current_x - self.offset_x
        camera_y = self.current_y - self.offset_y

        # Transform to pixel coordinates
        pos_real = np.array([[camera_x, camera_y]], dtype=np.float32).reshape(-1, 1, 2)
        pos_img = cv2.perspectiveTransform(pos_real, self.H_inv).reshape(-1, 2).astype(int)
        pos_pt = tuple(pos_img[0])

        # Draw robot position - styled like center marker but in green
        cv2.circle(frame, pos_pt, 12, (0, 255, 0), 3)  # Green outer circle
        cv2.circle(frame, pos_pt, 6, (0, 255, 0), -1)  # Green filled inner
        cv2.drawMarker(frame, pos_pt, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)  # White crosshair

        # Label
        cv2.putText(frame, "ROBOT", (pos_pt[0] + 18, pos_pt[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def run(self):
        """Main loop with keyboard control and YOLO detection."""
        print("\n" + "="*70)
        print("ROBOT KEYBOARD CONTROL WITH YOLO DETECTION")
        print("="*70)
        print("KEYBOARD CONTROLS:")
        print("  ↑↓←→   : Move robot (arrow keys)")
        print("  [ ]    : Move Z up/down")
        print("  1/2/3  : Step size (1mm/10mm/50mm)")
        print("  h      : Home position")
        print("  p      : Print position")
        print("  s      : STOP robot (emergency stop)")
        print("  q/ESC  : Quit (robot stays powered)")
        print("="*70 + "\n")

        window_name = "Robot Control with Detection"
        cv2.namedWindow(window_name)

        # Try to use keyboard input
        try:
            import msvcrt  # Windows
            def get_key():
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\xe0':  # Arrow key prefix
                        key = msvcrt.getch()
                        arrow_map = {b'H': 'up', b'P': 'down', b'K': 'left', b'M': 'right'}
                        return arrow_map.get(key, '')
                    elif key == b'\x1b':
                        return 'esc'
                    return key.decode('utf-8', errors='ignore')
                return None
        except ImportError:
            import tty
            import termios
            def get_key():
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':
                        ch2 = sys.stdin.read(1)
                        if ch2 == '[':
                            ch3 = sys.stdin.read(1)
                            arrow_map = {'A': 'up', 'B': 'down', 'C': 'right', 'D': 'left'}
                            return arrow_map.get(ch3, '')
                        return 'esc'
                    return ch
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return None

        print("[INFO] Keyboard control active")
        print(f"[INFO] Step size: {self.step_size}mm\n")

        try:
            while True:
                # Get camera frame
                frame = self.get_camera_frame()
                if frame is None:
                    continue

                # Run YOLO detection
                results = model(frame, conf=0.7)
                obb_preds = results[0].obb

                annotated_frame = frame.copy()

                # Draw detected objects
                for i, obb in enumerate(obb_preds, 1):
                    if hasattr(obb, "xyxyxyxy"):
                        corners = obb.xyxyxyxy.cpu().numpy().reshape(-1, 2)
                    elif hasattr(obb, "xyxy"):
                        corners = obb.xyxy.cpu().numpy().reshape(-1, 2)
                    else:
                        continue

                    # Check if inside workspace
                    transformed = self.transform_points(corners, self.H)
                    is_inside = self.is_inside_box(transformed)

                    if is_inside:
                        color = (0, 255, 0)  # Green

                        # Draw center point
                        center_img = np.mean(corners, axis=0).astype(int)
                        cv2.circle(annotated_frame, tuple(center_img), 5, (0, 0, 255), -1)  # Red
                        cv2.circle(annotated_frame, tuple(center_img), 5, (255, 255, 255), 1)
                        cv2.drawMarker(annotated_frame, tuple(center_img), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
                    else:
                        color = (0, 0, 255)  # Red

                    # Draw border
                    corners_int = corners.astype(int)
                    cv2.polylines(annotated_frame, [corners_int], isClosed=True, color=color, thickness=2)

                    # Label
                    label_pos = tuple(corners_int[0] - [0, 10])
                    cv2.putText(annotated_frame, f"Obj {i}", label_pos,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw workspace boundary
                self.draw_workspace_box(annotated_frame)

                # Draw robot position
                self.draw_robot_position(annotated_frame)

                # Draw info overlay
                cv2.putText(annotated_frame, "Robot Keyboard Control + Detection", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                if self.current_x is not None:
                    camera_x = self.current_x - self.offset_x
                    camera_y = self.current_y - self.offset_y
                    cv2.putText(annotated_frame, f"Robot: ({camera_x:.1f}, {camera_y:.1f}) mm", (10, 65),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                cv2.putText(annotated_frame, f"Step: {self.step_size}mm", (10, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Show frame
                cv2.imshow(window_name, annotated_frame)

                # Handle keyboard input
                key = get_key()

                if key == 'up':
                    print(f"↑ Moving UP ({self.step_size}mm)")
                    self.move_relative(dx=self.step_size)
                    self.update_current_position()

                elif key == 'down':
                    print(f"↓ Moving DOWN ({self.step_size}mm)")
                    self.move_relative(dx=-self.step_size)
                    self.update_current_position()

                elif key == 'left':
                    print(f"← Moving LEFT ({self.step_size}mm)")
                    self.move_relative(dy=self.step_size)
                    self.update_current_position()

                elif key == 'right':
                    print(f"→ Moving RIGHT ({self.step_size}mm)")
                    self.move_relative(dy=-self.step_size)
                    self.update_current_position()

                elif key in ['[', '5']:
                    print(f"↑ Moving Z+ ({self.step_size}mm)")
                    self.move_relative(dz=self.step_size)
                    self.update_current_position()

                elif key in [']', '6']:
                    print(f"↓ Moving Z- ({self.step_size}mm)")
                    self.move_relative(dz=-self.step_size)
                    self.update_current_position()

                elif key == '1':
                    self.step_size = self.fine_step
                    print(f"[STEP] Fine: {self.step_size}mm")

                elif key == '2':
                    self.step_size = 10
                    print(f"[STEP] Normal: {self.step_size}mm")

                elif key == '3':
                    self.step_size = self.large_step
                    print(f"[STEP] Large: {self.step_size}mm")

                elif key and key.lower() == 'h':
                    self.go_home()
                    self.update_current_position()

                elif key and key.lower() == 'p':
                    self.update_current_position()
                    if self.current_x is not None:
                        camera_x = self.current_x - self.offset_x
                        camera_y = self.current_y - self.offset_y
                        print(f"\n[POSITION]")
                        print(f"  Robot:  ({self.current_x:.1f}, {self.current_y:.1f}, {self.current_z:.1f}) mm")
                        print(f"  Camera: ({camera_x:.1f}, {camera_y:.1f}) mm\n")

                elif key and key.lower() == 's':
                    self.stop_robot()
                    self.update_current_position()

                elif key in ['q', 'esc'] or (cv2.waitKey(1) & 0xFF == ord('q')):
                    print("\n[INFO] Quitting... (Robot will remain powered)")
                    break

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n\n[INFO] Interrupted")

        finally:
            self.shutdown()

    def stop_robot(self):
        """Emergency stop - halt all robot motion."""
        try:
            print("\n[STOP] Emergency stop activated!")
            self._arm.set_state(4)  # Stop state
            time.sleep(0.1)
            self._arm.set_state(0)  # Back to ready
            print("[STOP] ✅ Robot stopped and ready")
            return True
        except Exception as e:
            print(f"[STOP] Error: {e}")
            return False

    def shutdown(self):
        """Cleanup camera and windows only. Does NOT stop or kill robot processes."""
        print("\n[Shutdown] Cleaning up camera and windows...")
        print("[Shutdown] Note: Robot remains active and powered")

        if self.camera is not None:
            try:
                self.camera.EndAcquisition()
                self.camera.DeInit()
            except:
                pass

        if self.camera_system is not None:
            try:
                cam_list = self.camera_system.GetCameras()
                cam_list.Clear()
                self.camera_system.ReleaseInstance()
            except:
                pass

        cv2.destroyAllWindows()
        print("[Shutdown] ✅ Complete\n")


if __name__ == "__main__":
    app = RobotKeyboardDetection()
    app.run()
