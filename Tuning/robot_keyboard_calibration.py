"""
Robot Area Calibration Tool - Keyboard Control with Camera View
Manually move the robot to calibrate workspace boundaries and test movements
Shows live camera feed with robot position overlay
"""

import json
import time
import os
from xarm.wrapper import XArmAPI
import sys
import cv2
import numpy as np
import pickle
import PySpin
import threading

class RobotAreaCalibrator:
    """Interactive robot calibration using keyboard controls with camera view."""
    
    def __init__(self, config_path="config.json"):
        """Initialize robot calibrator."""
        self.config = self.load_config(config_path)
        self.robot_ip = self.config.get("robot_ip", "192.168.1.151")
        self._arm = None
        
        # Movement parameters
        self.tcp_speed = self.config.get("tcp_speed", 200)
        self.tcp_acc = self.config.get("tcp_acc", 2000)
        self.angle_speed = self.config.get("angle_speed", 20)
        self.angle_acc = self.config.get("angle_acc", 500)
        
        # Load heights and offsets from config
        click_config = self.config.get("click_control", {})
        self.safe_height = click_config.get("safe_height", 200)
        self.pick_height = click_config.get("pick_height", 50)
        self.offset_x = click_config.get("coordinate_offset_x", 0)
        self.offset_y = click_config.get("coordinate_offset_y", -150)
        
        # Workspace limits
        self.workspace_min_x = click_config.get("workspace_min_x", 0)
        self.workspace_max_x = click_config.get("workspace_max_x", 300)
        self.workspace_min_y = click_config.get("workspace_min_y", 0)
        self.workspace_max_y = click_config.get("workspace_max_y", 300)
        
        # Movement step sizes (mm)
        self.step_size = 10  # Default step
        self.fine_step = 1   # Fine adjustment
        self.large_step = 50 # Large movement
        
        # Current position tracking
        self.current_x = None
        self.current_y = None
        self.current_z = None
        
        # Previous position for change detection
        self.prev_x = None
        self.prev_y = None
        self.prev_z = None
        
        # Calibration points
        self.calibration_points = {
            'top_left': None,      # (0, 300)
            'top_right': None,     # (300, 300)
            'bottom_right': None,  # (300, 0)
            'bottom_left': None,   # (0, 0)
            'center': None         # (150, 150)
        }

        # Coordinate transformation mode (for testing different formulas)
        self.transform_mode = 0  # Cycle through different transformations
        self.transform_modes = {
            0: "Mode 0: camera_x = -robot_y - offset_y, camera_y = robot_x - offset_x",
            1: "Mode 1: camera_x = robot_y + offset_y, camera_y = robot_x - offset_x",
            2: "Mode 2: camera_x = robot_x - offset_x, camera_y = robot_y - offset_y (no rotation)",
            3: "Mode 3: camera_x = robot_x - offset_x, camera_y = -robot_y - offset_y",
            4: "Mode 4: camera_x = -robot_y - offset_y, camera_y = -robot_x + offset_x"
        }

        # Home position
        self.home_position = [1.5, 6.3, 45.5, -0.4, 41.0, -7.0]
        
        # Camera setup
        self.camera = None
        self.camera_system = None
        self.H = None  # Homography matrix
        self.H_inv = None  # Inverse homography
        self.camera_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        self.connect_robot()
        self.initialize_robot()
        self.initialize_camera()
    
    def load_config(self, path="config.json"):
        """Load configuration from JSON file."""
        try:
            if not os.path.isabs(path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(base_dir, path)
            
            with open(path, "r") as f:
                config = json.load(f)
                print(f"[Config] ✅ Loaded from {path}")
                return config
        except Exception as e:
            print(f"[Config] ⚠️  Failed to load config: {e}")
            return {
                "robot_ip": "192.168.1.151",
                "tcp_speed": 200,
                "tcp_acc": 2000,
                "angle_speed": 20,
                "angle_acc": 500,
                "click_control": {
                    "safe_height": 200,
                    "pick_height": 50,
                    "workspace_min_x": 0,
                    "workspace_max_x": 300,
                    "workspace_min_y": 0,
                    "workspace_max_y": 300,
                    "coordinate_offset_x": 0,
                    "coordinate_offset_y": -150
                }
            }
    
    def connect_robot(self):
        """Connect to xArm robot."""
        while True:
            try:
                print(f"[Robot] Connecting to xArm at {self.robot_ip}...")
                self._arm = XArmAPI(self.robot_ip, baud_checkset=False)
                print("[Robot] ✅ Connection established.")
                break
            except Exception as e:
                print(f"[Robot] Connection failed: {e}. Retrying in 5s...")
                time.sleep(5)
    
    def initialize_robot(self):
        """Initialize robot state."""
        try:
            print("[Robot] Initializing robot...")
            self._arm.clean_warn()
            self._arm.clean_error()
            self._arm.motion_enable(True)
            self._arm.set_mode(0)
            self._arm.set_state(0)
            time.sleep(0.5)
            
            # Get current position
            self.update_current_position()
            
            print("[Robot] ✅ Robot initialized and ready.")
        except Exception as e:
            print(f"[Robot] ⚠️  Initialization warning: {e}")
    
    def initialize_camera(self):
        """Initialize FLIR camera and load homography."""
        try:
            print("[Camera] Initializing FLIR camera...")
            self.camera_system = PySpin.System.GetInstance()
            cam_list = self.camera_system.GetCameras()
            
            if cam_list.GetSize() == 0:
                print("[Camera] ⚠️  No FLIR cameras found. Running without camera view.")
                return False
            
            self.camera = cam_list.GetByIndex(0)
            self.camera.Init()
            self.camera.BeginAcquisition()
            print("[Camera] ✅ Camera initialized successfully.")
            
            # Try to load homography
            homography_files = ["homography_auto.pkl", "homography_calibration.pkl"]
            for filename in homography_files:
                if os.path.exists(filename):
                    try:
                        with open(filename, "rb") as f:
                            self.H = pickle.load(f)
                        self.H_inv = np.linalg.inv(self.H)
                        print(f"[Camera] ✅ Loaded homography from: {filename}")
                        return True
                    except Exception as e:
                        print(f"[Camera] ⚠️  Failed to load {filename}: {e}")
            
            print("[Camera] ⚠️  No homography calibration found. Camera view available but no coordinate overlay.")
            return True
            
        except Exception as e:
            print(f"[Camera] ⚠️  Camera initialization failed: {e}")
            return False
    
    def convert_pyspin_to_cv2(self, image):
        """Convert PySpin image to OpenCV format."""
        if not image.IsValid():
            return None
        img_array = image.GetNDArray()
        if len(img_array.shape) == 2:
            return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif len(img_array.shape) == 3:
            return img_array
        return None
    
    def get_camera_frame(self):
        """Get current camera frame."""
        if self.camera is None:
            return None
        
        try:
            image = self.camera.GetNextImage()
            if image.IsIncomplete():
                image.Release()
                return None
            
            frame = self.convert_pyspin_to_cv2(image)
            image.Release()
            return frame
        except Exception as e:
            return None
    
    def robot_to_camera_coords(self, robot_x, robot_y):
        """
        Convert robot coordinates to camera/workspace coordinates.
        Supports multiple transformation modes for testing.
        Press 'm' to cycle through modes to find correct transformation.
        """
        if self.transform_mode == 0:
            # Original fix: 90-degree rotation
            camera_x = -robot_y - self.offset_y
            camera_y = robot_x - self.offset_x
        elif self.transform_mode == 1:
            # Alternative: different rotation direction
            camera_x = robot_y + self.offset_y
            camera_y = robot_x - self.offset_x
        elif self.transform_mode == 2:
            # No rotation - simple offset
            camera_x = robot_x - self.offset_x
            camera_y = robot_y - self.offset_y
        elif self.transform_mode == 3:
            # Rotation variant 3
            camera_x = robot_x - self.offset_x
            camera_y = -robot_y - self.offset_y
        elif self.transform_mode == 4:
            # Rotation variant 4
            camera_x = -robot_y - self.offset_y
            camera_y = -robot_x + self.offset_x
        else:
            camera_x = robot_x - self.offset_x
            camera_y = robot_y - self.offset_y

        return camera_x, camera_y

    def draw_workspace_boundary(self, frame):
        """Draw the workspace boundary on the frame."""
        if self.H_inv is None:
            return
        
        # Create workspace corners in real coordinates (mm)
        box_real = np.array([
            [self.workspace_min_x, self.workspace_max_y],  # Top-left
            [self.workspace_max_x, self.workspace_max_y],  # Top-right
            [self.workspace_max_x, self.workspace_min_y],  # Bottom-right
            [self.workspace_min_x, self.workspace_min_y]   # Bottom-left
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # Transform to image coordinates
        box_img = cv2.perspectiveTransform(box_real, self.H_inv).reshape(-1, 2).astype(int)
        
        # Draw boundary - white line like original
        cv2.polylines(frame, [box_img], isClosed=True, color=(255, 255, 255), thickness=3)  # White, thick for visibility
        
        # Label corners with better visibility
        labels = ["TL", "TR", "BR", "BL"]
        for i, (pt, label) in enumerate(zip(box_img, labels)):
            cv2.putText(frame, label, tuple(pt + [5, -5]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Larger, white, bold
        
        # Draw center point of workspace - LARGE AND VISIBLE
        center_real = np.array([[(self.workspace_max_x + self.workspace_min_x) / 2,
                                  (self.workspace_max_y + self.workspace_min_y) / 2]], 
                                dtype=np.float32).reshape(-1, 1, 2)
        center_img = cv2.perspectiveTransform(center_real, self.H_inv).reshape(-1, 2).astype(int)
        
        # Draw center reference marker - visible but not huge
        center_pt = tuple(center_img[0])

        # Outer circle (bright yellow for high visibility)
        cv2.circle(frame, center_pt, 12, (0, 255, 255), 3)  # Cyan, thicker

        # Inner filled circle for contrast
        cv2.circle(frame, center_pt, 6, (0, 255, 255), -1)  # Filled cyan

        # Crosshair
        cv2.drawMarker(frame, center_pt, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)  # White crosshair

        # Label
        cv2.putText(frame, "CENTER", (center_pt[0] + 18, center_pt[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    
    def draw_robot_position(self, frame):
        """Draw current robot position on the frame."""
        if self.H_inv is None or self.current_x is None:
            return

        # Convert robot coordinates to camera coordinates (accounts for 90° rotation)
        camera_x, camera_y = self.robot_to_camera_coords(self.current_x, self.current_y)
        
        # Check if position changed (threshold of 0.1mm to avoid floating point noise)
        position_changed = False
        if self.prev_x is None or self.prev_y is None or self.prev_z is None:
            position_changed = True
        elif (abs(self.current_x - self.prev_x) > 0.1 or 
              abs(self.current_y - self.prev_y) > 0.1 or 
              abs(self.current_z - self.prev_z) > 0.1):
            position_changed = True
        
        # Transform to image coordinates
        pos_real = np.array([[camera_x, camera_y]], dtype=np.float32).reshape(-1, 1, 2)
        pos_img = cv2.perspectiveTransform(pos_real, self.H_inv).reshape(-1, 2).astype(int)
        
        pos_pt = tuple(pos_img[0])
        pixel_x, pixel_y = pos_img[0]
        
        # Only print debug when position changes
        if position_changed:
            frame_height, frame_width = frame.shape[:2]
            in_frame = (0 <= pixel_x < frame_width and 0 <= pixel_y < frame_height)
            print(f"\n[DEBUG - Mode {self.transform_mode}]")
            print(f"  Robot:  X={self.current_x:7.1f}, Y={self.current_y:7.1f}, Z={self.current_z:7.1f} mm")
            print(f"  Camera: X={camera_x:7.1f}, Y={camera_y:7.1f} mm")
            print(f"  Pixel:  X={pixel_x:4d}, Y={pixel_y:4d} {'✓ IN FRAME' if in_frame else '✗ OFF-SCREEN'}")
            print(f"  Offset: X={self.offset_x:7.1f}, Y={self.offset_y:7.1f} mm")

            # Update previous position
            self.prev_x = self.current_x
            self.prev_y = self.current_y
            self.prev_z = self.current_z
        
<<<<<<< HEAD
        # Draw robot position - visible dot with high contrast
        cv2.circle(frame, pos_pt, 30, (0, 0, 255), -1)  # Red filled circle (larger)
        cv2.circle(frame, pos_pt, 50, (255, 255, 255), 2)  # Thick white border
        cv2.circle(frame, pos_pt, 20, (255, 255, 0), -1)  # Small cyan center dot for precision
=======
        # Draw robot position - styled like center marker
        cv2.circle(frame, pos_pt, 12, (0, 0, 255), 3)  # Red outer circle (thick border)
        cv2.circle(frame, pos_pt, 6, (0, 0, 255), -1)  # Red filled inner circle
        cv2.drawMarker(frame, pos_pt, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)  # White crosshair
>>>>>>> b80d9c9d4fa0f81317ec0a2204dcfe2a6608ccff
        
        # If off-screen, draw arrow pointing to it
        frame_height, frame_width = frame.shape[:2]
        if not (0 <= pixel_x < frame_width and 0 <= pixel_y < frame_height):
            center_x, center_y = frame_width // 2, frame_height // 2
            # Clamp to edge with margin
            arrow_end_x = max(50, min(frame_width - 50, pixel_x))
            arrow_end_y = max(50, min(frame_height - 50, pixel_y))
            cv2.arrowedLine(frame, (center_x, center_y), (arrow_end_x, arrow_end_y),
                           (0, 0, 255), 3, tipLength=0.2)
            cv2.putText(frame, "ROBOT OFF-SCREEN", (center_x - 80, center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    def draw_saved_points(self, frame):
        """Draw saved calibration points on the frame."""
        if self.H_inv is None:
            return
        
        point_info = {
            'top_left': ('TL', (0, 255, 0)),
            'top_right': ('TR', (0, 255, 0)),
            'bottom_right': ('BR', (0, 255, 0)),
            'bottom_left': ('BL', (0, 255, 0)),
            'center': ('C', (255, 0, 255))
        }
        
        for key, (label, color) in point_info.items():
            point = self.calibration_points[key]
            if point is not None:
                # Convert robot coordinates to camera coordinates (accounts for 90° rotation)
                camera_x, camera_y = self.robot_to_camera_coords(point['robot_x'], point['robot_y'])
                
                # Transform to image coordinates
                pos_real = np.array([[camera_x, camera_y]], dtype=np.float32).reshape(-1, 1, 2)
                pos_img = cv2.perspectiveTransform(pos_real, self.H_inv).reshape(-1, 2).astype(int)
                
                pos_pt = tuple(pos_img[0])

                # Draw saved point with better visibility
                cv2.circle(frame, pos_pt, 10, color, -1)  # Larger filled circle
                cv2.circle(frame, pos_pt, 12, (255, 255, 255), 3)  # Thick white border
                cv2.putText(frame, label, (pos_pt[0] + 15, pos_pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def camera_display_thread(self):
        """Thread to continuously display camera feed."""
        window_name = "Robot Calibration - Camera View"
        cv2.namedWindow(window_name)
        
        print("[Camera] Starting camera display thread...")
        
        while self.camera_running:
            frame = self.get_camera_frame()
            
            if frame is not None:
                display_frame = frame.copy()
                
                # Draw workspace boundary
                self.draw_workspace_boundary(display_frame)
                
                # Draw robot position
                self.draw_robot_position(display_frame)
                
                # Draw saved calibration points
                self.draw_saved_points(display_frame)
                
                # Add info overlay with better visibility
                info_y = 30
                cv2.putText(display_frame, "Robot Calibration - Live View", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Show transformation mode (highlighted)
                info_y += 35
                mode_text = f"Transform Mode: {self.transform_mode} (press 'm' to cycle)"
                cv2.putText(display_frame, mode_text, (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if self.current_x is not None:
                    camera_x, camera_y = self.robot_to_camera_coords(self.current_x, self.current_y)
                    info_y += 30
                    cv2.putText(display_frame, f"Robot: ({self.current_x:.1f}, {self.current_y:.1f}) mm",
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    info_y += 25
                    cv2.putText(display_frame, f"Camera: ({camera_x:.1f}, {camera_y:.1f}) mm",
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                info_y += 30
                cv2.putText(display_frame, f"Step: {self.step_size}mm",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                with self.frame_lock:
                    self.current_frame = display_frame
                
                cv2.imshow(window_name, display_frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.03)  # ~30 fps
        
        cv2.destroyWindow(window_name)
        print("[Camera] Camera display thread stopped.")
    
    def update_current_position(self):
        """Update current position from robot."""
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
        """Move robot to absolute position."""
        print(f"\n[MOVE] Target: X={x:.1f}, Y={y:.1f}, Z={z:.1f} mm")
        
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
                print("[MOVE] ✅ Movement complete")
                return True
            else:
                print(f"[MOVE] ❌ Movement failed (code={code})")
                return False
                
        except Exception as e:
            print(f"[MOVE] ❌ Exception: {e}")
            return False
    
    def go_home(self):
        """Move robot to home position."""
        print("[Robot] 🏠 Moving to home position...")
        try:
            code = self._arm.set_servo_angle(
                angle=self.home_position,
                speed=self.angle_speed,
                mvacc=self.angle_acc,
                wait=True
            )
            if code == 0:
                self.update_current_position()
                print("[Robot] ✅ Home position reached.")
                return True
        except Exception as e:
            print(f"[Robot Error] Failed to go home: {e}")
        return False
    
    def save_calibration_point(self, point_name):
        """Save current position as a calibration point."""
        self.update_current_position()
        
        if self.current_x is not None:
            # Store raw robot coordinates
            self.calibration_points[point_name] = {
                'robot_x': self.current_x,
                'robot_y': self.current_y,
                'robot_z': self.current_z
            }
            
            # Convert to camera/workspace coordinates
            camera_x, camera_y = self.robot_to_camera_coords(self.current_x, self.current_y)

            print(f"\n[SAVED] {point_name.upper().replace('_', ' ')}")
            print(f"  Robot coords: ({self.current_x:.1f}, {self.current_y:.1f}, {self.current_z:.1f})")
            print(f"  Camera coords: ({camera_x:.1f}, {camera_y:.1f})")
            return True
        return False
    
    def test_gripper(self, action):
        """Test gripper open/close."""
        try:
            if action == 'open':
                print("[GRIPPER] Opening...")
                self._arm.open_lite6_gripper()
            elif action == 'close':
                print("[GRIPPER] Closing...")
                self._arm.close_lite6_gripper()
            elif action == 'stop':
                print("[GRIPPER] Stopping...")
                self._arm.stop_lite6_gripper()
            time.sleep(1)
            return True
        except Exception as e:
            print(f"[GRIPPER] Error: {e}")
            return False
    
    def display_help(self):
        """Display keyboard controls."""
        print("\n" + "="*80)
        print("KEYBOARD CONTROLS (Movements match visual camera view)")
        print("="*80)
        print("\n--- MOVEMENT (Visual Direction) ---")
        print("  Arrow Keys:")
        print("    ↑         : Move UP (toward top of camera view)")
        print("    ↓         : Move DOWN (toward bottom of camera view)")
        print("    ←         : Move LEFT (toward left of camera view)")
        print("    →         : Move RIGHT (toward right of camera view)")
        print("  Page Up/Down:")
        print("    PgUp / [  : Move Z+ (Up - away from table)")
        print("    PgDn / ]  : Move Z- (Down - toward table)")
        print("\n--- STEP SIZE ---")
        print("    1         : Fine step (1mm)")
        print("    2         : Normal step (10mm)")
        print("    3         : Large step (50mm)")
        print("\n--- COORDINATE TRANSFORM ---")
        print("    m         : Cycle transform mode (if dot doesn't sync with robot)")
        print("\n--- CALIBRATION POINTS ---")
        print("    q         : Save current position as TOP-LEFT (0, 300)")
        print("    w         : Save current position as TOP-RIGHT (300, 300)")
        print("    e         : Save current position as BOTTOM-RIGHT (300, 0)")
        print("    r         : Save current position as BOTTOM-LEFT (0, 0)")
        print("    t         : Save current position as CENTER (150, 150)")
        print("\n--- GRIPPER TEST ---")
        print("    o         : Open gripper")
        print("    c         : Close gripper")
        print("    s         : Stop gripper")
        print("\n--- SYSTEM ---")
        print("    h         : Go to home position")
        print("    p         : Show current position")
        print("    v         : View saved calibration points")
        print("    x         : Export calibration to file")
        print("    ?         : Show this help")
        print("    ESC / z   : Exit calibration")
        print("="*80 + "\n")
    
    def display_current_position(self):
        """Display current robot position."""
        self.update_current_position()
        
        if self.current_x is not None:
            # Convert robot coordinates to camera coordinates (accounts for 90° rotation)
            camera_x, camera_y = self.robot_to_camera_coords(self.current_x, self.current_y)

            print("\n" + "-"*60)
            print("CURRENT POSITION")
            print("-"*60)
            print(f"Robot coords:  X={self.current_x:7.1f} mm, Y={self.current_y:7.1f} mm, Z={self.current_z:7.1f} mm")
            print(f"Camera coords: X={camera_x:7.1f} mm, Y={camera_y:7.1f} mm")
            
            # Check if in workspace
            workspace_min_x = 0
            workspace_max_x = 300
            workspace_min_y = 0
            workspace_max_y = 300
            
            in_workspace = (workspace_min_x <= camera_x <= workspace_max_x and
                          workspace_min_y <= camera_y <= workspace_max_y)
            status = "✅ INSIDE" if in_workspace else "⚠️  OUTSIDE"
            print(f"Workspace: {status}")
            print("-"*60 + "\n")
    
    def view_calibration_points(self):
        """Display all saved calibration points."""
        print("\n" + "="*80)
        print("SAVED CALIBRATION POINTS")
        print("="*80)
        
        point_labels = {
            'top_left': 'TOP-LEFT (0, 300)',
            'top_right': 'TOP-RIGHT (300, 300)',
            'bottom_right': 'BOTTOM-RIGHT (300, 0)',
            'bottom_left': 'BOTTOM-LEFT (0, 0)',
            'center': 'CENTER (150, 150)'
        }
        
        any_saved = False
        for key, label in point_labels.items():
            point = self.calibration_points[key]
            if point is not None:
                any_saved = True
                camera_x, camera_y = self.robot_to_camera_coords(point['robot_x'], point['robot_y'])
                print(f"\n{label}:")
                print(f"  Robot:  X={point['robot_x']:7.1f}, Y={point['robot_y']:7.1f}, Z={point['robot_z']:7.1f}")
                print(f"  Camera: X={camera_x:7.1f}, Y={camera_y:7.1f}")
            else:
                print(f"\n{label}: ❌ Not saved")
        
        if not any_saved:
            print("\n⚠️  No calibration points saved yet!")
        
        print("="*80 + "\n")
    
    def export_calibration(self, filename="robot_area_calibration.json"):
        """Export calibration points to JSON file."""
        if all(v is None for v in self.calibration_points.values()):
            print("\n[ERROR] No calibration points to export!")
            return False
        
        try:
            export_data = {
                "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "robot_ip": self.robot_ip,
                "offsets": {
                    "x": self.offset_x,
                    "y": self.offset_y
                },
                "workspace_limits": {
                    "min_x": self.workspace_min_x,
                    "max_x": self.workspace_max_x,
                    "min_y": self.workspace_min_y,
                    "max_y": self.workspace_max_y
                },
                "calibration_points": {}
            }
            
            for key, point in self.calibration_points.items():
                if point is not None:
                    camera_x, camera_y = self.robot_to_camera_coords(point['robot_x'], point['robot_y'])

                    export_data["calibration_points"][key] = {
                        "robot_coords": {
                            "x": round(point['robot_x'], 2),
                            "y": round(point['robot_y'], 2),
                            "z": round(point['robot_z'], 2)
                        },
                        "camera_coords": {
                            "x": round(camera_x, 2),
                            "y": round(camera_y, 2)
                        }
                    }
            
            with open(filename, "w") as f:
                json.dump(export_data, f, indent=4)
            
            print(f"\n[SUCCESS] ✅ Calibration exported to: {filename}")
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Failed to export calibration: {e}")
            return False
    
    def run_interactive_calibration(self):
        """Main interactive calibration loop with keyboard controls."""
        print("\n" + "="*80)
        print("ROBOT AREA CALIBRATION - INTERACTIVE MODE WITH CAMERA VIEW")
        print("="*80)
        print("\nUse keyboard to move the robot and calibrate workspace boundaries.")
        print("Camera view shows live robot position and workspace overlay.")
        print("Press '?' for help at any time.")
        print("="*80 + "\n")
        
        # Start camera display thread if camera is available
        if self.camera is not None:
            self.camera_running = True
            camera_thread = threading.Thread(target=self.camera_display_thread, daemon=True)
            camera_thread.start()
            print("[Camera] ✅ Camera view started in separate window.")
            time.sleep(1)  # Give camera thread time to start
        else:
            print("[Camera] ⚠️  Running without camera view.")
        
        # Show help initially
        self.display_help()
        
        # Display current position
        self.display_current_position()
        
        try:
            # Try to use getch for single keypress (Windows/Unix compatible)
            try:
                import msvcrt  # Windows
                def get_key():
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b'\xe0':  # Arrow key prefix on Windows
                            key = msvcrt.getch()
                            arrow_map = {b'H': 'up', b'P': 'down', b'K': 'left', b'M': 'right'}
                            return arrow_map.get(key, '')
                        elif key == b'\x1b':  # ESC
                            return 'esc'
                        return key.decode('utf-8', errors='ignore')
                    return None
            except ImportError:
                # Unix/Linux
                import tty
                import termios
                def get_key():
                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    try:
                        tty.setraw(fd)
                        ch = sys.stdin.read(1)
                        if ch == '\x1b':  # ESC or arrow key
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
            
            print("[INFO] Keyboard control active. Press keys to move robot...")
            print("[INFO] Current step size: 10mm (press 1/2/3 to change)")
            if self.camera is not None:
                print("[INFO] Watch the camera window to see robot position!")
            print()
            
            while True:
                key = get_key()
                
                if key is None:
                    time.sleep(0.05)
                    continue
                
                # Movement controls (CORRECT MAPPING from diagnostic test)
                # Diagnostic results:
                # - Robot X+ moves visually UP
                # - Robot Y+ moves visually LEFT
                # Therefore:
                # - To move visually UP: dx=+step
                # - To move visually DOWN: dx=-step
                # - To move visually LEFT: dy=+step
                # - To move visually RIGHT: dy=-step
                
                if key == 'up':
                    print(f"↑ Moving visually UP ({self.step_size}mm)")
                    self.move_relative(dx=self.step_size)
                    self.display_current_position()
                
                elif key == 'down':
                    print(f"↓ Moving visually DOWN ({self.step_size}mm)")
                    self.move_relative(dx=-self.step_size)
                    self.display_current_position()
                
                elif key == 'left':
                    print(f"← Moving visually LEFT ({self.step_size}mm)")
                    self.move_relative(dy=self.step_size)
                    self.display_current_position()
                
                elif key == 'right':
                    print(f"→ Moving visually RIGHT ({self.step_size}mm)")
                    self.move_relative(dy=-self.step_size)
                    self.display_current_position()
                
                # Z-axis movement (try Page Up/Down simulation)
                elif key in ['[', '5']:  # Page Up simulation
                    print(f"↑ Moving Z+ ({self.step_size}mm)")
                    self.move_relative(dz=self.step_size)
                    self.display_current_position()
                
                elif key in [']', '6']:  # Page Down simulation
                    print(f"↓ Moving Z- ({self.step_size}mm)")
                    self.move_relative(dz=-self.step_size)
                    self.display_current_position()
                
                # Step size control
                elif key == '1':
                    self.step_size = self.fine_step
                    print(f"[STEP] Fine step: {self.step_size}mm")
                
                elif key == '2':
                    self.step_size = 10
                    print(f"[STEP] Normal step: {self.step_size}mm")
                
                elif key == '3':
                    self.step_size = self.large_step
                    print(f"[STEP] Large step: {self.step_size}mm")

                # Cycle transformation mode
                elif key.lower() == 'm':
                    self.transform_mode = (self.transform_mode + 1) % len(self.transform_modes)
                    print(f"\n[TRANSFORM MODE {self.transform_mode}]")
                    print(f"  {self.transform_modes[self.transform_mode]}")
                    print(f"  Move robot and watch if dot syncs correctly!")
                    self.display_current_position()

                # Save calibration points
                elif key.lower() == 'q':
                    self.save_calibration_point('top_left')
                
                elif key.lower() == 'w':
                    self.save_calibration_point('top_right')
                
                elif key.lower() == 'e':
                    self.save_calibration_point('bottom_right')
                
                elif key.lower() == 'r':
                    self.save_calibration_point('bottom_left')
                
                elif key.lower() == 't':
                    self.save_calibration_point('center')
                
                # Gripper controls
                elif key.lower() == 'o':
                    self.test_gripper('open')
                
                elif key.lower() == 'c':
                    self.test_gripper('close')
                
                elif key.lower() == 's':
                    self.test_gripper('stop')
                
                # System controls
                elif key.lower() == 'h':
                    self.go_home()
                    self.display_current_position()
                
                elif key.lower() == 'p':
                    self.display_current_position()
                
                elif key.lower() == 'v':
                    self.view_calibration_points()
                
                elif key.lower() == 'x':
                    filename = input("\nEnter filename (or press Enter for default): ").strip()
                    if not filename:
                        filename = "robot_area_calibration.json"
                    self.export_calibration(filename)
                
                elif key == '?':
                    self.display_help()
                
                elif key in ['z', 'esc']:
                    print("\n[INFO] Exiting calibration mode...")
                    break
        
        except KeyboardInterrupt:
            print("\n\n[INFO] Calibration interrupted by user.")
        
        except Exception as e:
            print(f"\n[ERROR] Calibration error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Stop camera thread
            self.camera_running = False
            time.sleep(0.5)  # Give thread time to stop
    
    def shutdown(self):
        """Shutdown robot and camera safely."""
        print("\n[Robot] Shutting down...")
        
        # Stop camera
        self.camera_running = False
        if self.camera is not None:
            try:
                self.camera.EndAcquisition()
                self.camera.DeInit()
                del self.camera
                print("[Camera] ✅ Camera shutdown complete.")
            except Exception as e:
                print(f"[Camera] Shutdown error: {e}")
        
        if self.camera_system is not None:
            try:
                cam_list = self.camera_system.GetCameras()
                cam_list.Clear()
                self.camera_system.ReleaseInstance()
            except Exception as e:
                print(f"[Camera System] Cleanup error: {e}")
        
        # Shutdown robot
        try:
            self.go_home()
            self._arm.set_state(4)
            print("[Robot] ✅ Robot shutdown complete.")
        except Exception as e:
            print(f"[Robot] Shutdown error: {e}")
        
        # Close any remaining OpenCV windows
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("ROBOT AREA CALIBRATION TOOL - KEYBOARD CONTROL WITH CAMERA VIEW")
    print("="*80)
    print("\nThis tool helps you:")
    print("  • View live camera feed with workspace overlay")
    print("  • See robot position in real-time on camera view")
    print("  • Manually move the robot using keyboard controls")
    print("  • Mark important workspace positions (corners, center)")
    print("  • Test robot movements and gripper")
    print("  • Export calibration data for verification")
    print("\nCamera window shows:")
    print("  • White box: Workspace boundary")
    print("  • Yellow crosshair: Workspace center (150, 150)")
    print("  • Red dot: Current robot position")
    print("  • Green dots: Saved calibration points")
    print("\nNOTE: For Page Up/Down simulation, use [ and ] keys")
    print("="*80 + "\n")
    
    try:
        calibrator = RobotAreaCalibrator()
        calibrator.run_interactive_calibration()
        calibrator.shutdown()
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[INFO] Calibration tool closed.\n")


if __name__ == "__main__":
    main()
