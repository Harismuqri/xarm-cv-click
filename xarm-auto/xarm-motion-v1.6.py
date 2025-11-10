"""
xArm Robot Controller - Click-based Movement with Inspection Camera Support
Left Click = Move, Middle Click = Pick (with auto-angle), Right Click = Place, T key = Inspect
"""

from multiprocessing import shared_memory
import struct
import json
import time
import os
import pickle
import cv2
import numpy as np
from xarm.wrapper import XArmAPI

# Shared memory configuration
CLICK_MEMORY_NAME = "ClickData"
CLICK_MEMORY_SIZE = 512
INSPECT_MEMORY_NAME = "InspectData"
INSPECT_MEMORY_SIZE = 512

# Camera offset configuration (mm)
CAMERA_OFFSET_X = 92.9
CAMERA_OFFSET_Y = -1.35
CAMERA_OFFSET_ERROR = 0.4

# Robot workspace boundaries (mm) - actual robot coordinates
ROBOT_MIN_X = 88.9
ROBOT_MAX_X = 382.0
ROBOT_MIN_Y = 14.7
ROBOT_MAX_Y = 312.0

class ClickDataManager:
    """Manages shared memory for mouse click data."""

    def __init__(self, name=CLICK_MEMORY_NAME, size=CLICK_MEMORY_SIZE):
        self.name = name
        self.size = size
        self.shm = None
        self._initialize_shared_memory()

    def _initialize_shared_memory(self):
        """Create or attach to existing shared memory."""
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            print(f"[INFO] Created click data shared memory: {self.name}")
            initial_data = {"click_x": 0, "click_y": 0, "timestamp": 0, "processed": True, "button": "none", "angle": 0.0, "width": 0.0, "height": 0.0}
            self._write_data(initial_data)
            print(f"[DEBUG] Initialized with data: {initial_data}")
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
            print(f"[INFO] Attached to existing click data shared memory: {self.name}")
            existing_data = self._read_data()
            print(f"[DEBUG] Found existing data: {existing_data}")

    def _write_data(self, data):
        """Write data to shared memory as JSON."""
        try:
            json_str = json.dumps(data)
            json_bytes = json_str.encode('utf-8')

            if len(json_bytes) > self.size - 4:
                return False

            self.shm.buf[:4] = struct.pack('I', len(json_bytes))
            self.shm.buf[4:4+len(json_bytes)] = json_bytes
            return True
        except Exception as e:
            print(f"[ERROR] Failed to write click data: {e}")
            return False

    def _read_data(self):
        """Read data from shared memory."""
        try:
            length = struct.unpack('I', bytes(self.shm.buf[:4]))[0]
            if length == 0 or length > self.size - 4:
                return {"click_x": 0, "click_y": 0, "timestamp": 0, "processed": True, "button": "none", "angle": 0.0, "width": 0.0, "height": 0.0}

            json_bytes = bytes(self.shm.buf[4:4+length])
            json_str = json_bytes.decode('utf-8')
            data = json.loads(json_str)

            # Ensure button, angle, width, and height keys exist
            if "button" not in data:
                data["button"] = "left"
            if "angle" not in data:
                data["angle"] = 0.0
            if "width" not in data:
                data["width"] = 0.0
            if "height" not in data:
                data["height"] = 0.0

            return data
        except Exception as e:
            print(f"[ERROR] Failed to read: {e}")
            return {"click_x": 0, "click_y": 0, "timestamp": 0, "processed": True, "button": "none", "angle": 0.0, "width": 0.0, "height": 0.0}

    def read_click(self):
        """Read click data."""
        return self._read_data()

    def mark_processed(self):
        """Mark the current click as processed."""
        data = self._read_data()
        data["processed"] = True
        self._write_data(data)

    def cleanup(self):
        """Close and unlink shared memory."""
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
                print(f"[INFO] Click data shared memory cleaned up: {self.name}")
            except Exception as e:
                print(f"[ERROR] Failed to cleanup: {e}")


class InspectDataManager:
    """Manages shared memory for inspection commands."""

    def __init__(self, name=INSPECT_MEMORY_NAME, size=INSPECT_MEMORY_SIZE):
        self.name = name
        self.size = size
        self.shm = None
        self._initialize_shared_memory()

    def _initialize_shared_memory(self):
        """Create or attach to existing shared memory."""
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            print(f"[INFO] Created inspect data shared memory: {self.name}")
            initial_data = {"inspect": False, "target_x": 0, "target_y": 0, "angle": 0.0, "offset_x": CAMERA_OFFSET_X, "offset_y": CAMERA_OFFSET_Y, "timestamp": 0, "processed": True}
            self._write_data(initial_data)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
            print(f"[INFO] Attached to existing inspect data shared memory: {self.name}")

    def _write_data(self, data):
        """Write data to shared memory as JSON."""
        try:
            json_str = json.dumps(data)
            json_bytes = json_str.encode('utf-8')

            if len(json_bytes) > self.size - 4:
                return False

            self.shm.buf[:4] = struct.pack('I', len(json_bytes))
            self.shm.buf[4:4+len(json_bytes)] = json_bytes
            return True
        except Exception as e:
            print(f"[ERROR] Failed to write inspect data: {e}")
            return False

    def _read_data(self):
        """Read data from shared memory."""
        try:
            length = struct.unpack('I', bytes(self.shm.buf[:4]))[0]
            if length == 0 or length > self.size - 4:
                return {"inspect": False, "target_x": 0, "target_y": 0, "angle": 0.0, "offset_x": CAMERA_OFFSET_X, "offset_y": CAMERA_OFFSET_Y, "timestamp": 0, "processed": True}

            json_bytes = bytes(self.shm.buf[4:4+length])
            json_str = json_bytes.decode('utf-8')
            data = json.loads(json_str)

            # Ensure angle and offset keys exist for backward compatibility
            if "angle" not in data:
                data["angle"] = 0.0
            if "offset_x" not in data:
                data["offset_x"] = CAMERA_OFFSET_X
            if "offset_y" not in data:
                data["offset_y"] = CAMERA_OFFSET_Y

            return data
        except Exception as e:
            print(f"[ERROR] Failed to read inspect data: {e}")
            return {"inspect": False, "target_x": 0, "target_y": 0, "angle": 0.0, "offset_x": CAMERA_OFFSET_X, "offset_y": CAMERA_OFFSET_Y, "timestamp": 0, "processed": True}

    def read_inspect(self):
        """Read inspection command."""
        return self._read_data()

    def mark_processed(self):
        """Mark the current inspection command as processed."""
        data = self._read_data()
        data["processed"] = True
        self._write_data(data)

    def write_inspect_command(self, target_x, target_y, angle=0.0):
        """Send inspection command with target position and object angle."""
        data = {
            "inspect": True,
            "target_x": float(target_x),
            "target_y": float(target_y),
            "angle": float(angle),
            "offset_x": CAMERA_OFFSET_X,
            "offset_y": CAMERA_OFFSET_Y,
            "offset_error": CAMERA_OFFSET_ERROR,
            "timestamp": time.time(),
            "processed": False
        }
        self._write_data(data)
        print(f"[INSPECT] Inspection command sent: Target ({target_x:.1f}, {target_y:.1f}) mm - Angle: {angle:.1f}°")
        print(f"[INSPECT] Camera offset: ({CAMERA_OFFSET_X:.1f}, {CAMERA_OFFSET_Y:.1f}) ± {CAMERA_OFFSET_ERROR:.1f} mm")

    def cleanup(self):
        """Close and unlink shared memory."""
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
                print(f"[INFO] Inspect data shared memory cleaned up: {self.name}")
            except Exception as e:
                print(f"[ERROR] Failed to cleanup: {e}")


class XArmController:
    """xArm robot controller for click-based movement with automatic gripper angle adjustment and inspection support."""

    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        self.robot_ip = self.config.get("robot_ip", "192.168.1.151")
        self._arm = None
        self.is_moving = False

        # Load heights from config
        click_config = self.config.get("click_control", {})
        self.safe_height = click_config.get("safe_height", 150)
        self.pick_height = click_config.get("pick_height", -5)
        self.inspection_height = click_config.get("inspection_height", 103.4)

        # Inspection height (Z offset of camera from workspace)
        self.inspect_height = self.inspection_height  # mm - actual camera Z position

        # Load homography matrix for coordinate transformation
        self.H_det_to_robot = None
        self.load_homography()

        # Home position
        self.home_position = [1.5, 6.3, 45.5, 0, 39.2, 3.2]

        # Load calibration position from config
        calib_config = self.config.get("calibration_position", {})
        self.calibration_position = {
            "x": calib_config.get("x", -95.3),
            "y": calib_config.get("y", 211.6),
            "z": calib_config.get("z", 172.1),
            "roll": calib_config.get("roll", -179.6),
            "pitch": calib_config.get("pitch", -1.2),
            "yaw": calib_config.get("yaw", -1.6)
        }

        self.connect_robot()
        self.initialize_robot()
        
        # NEW STARTUP SEQUENCE: Move to calibration position and wait for YOLO
        print("[Robot] Moving to calibration position for camera calibration...")
        self.go_to_calibration_position()
        self.wait_for_yolo_calibration()
        print("[Robot] Moving to home position...")
        self.go_home()

    def load_config(self, path="config.json"):
        """Load configuration from JSON file."""
        try:
            if not os.path.isabs(path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(base_dir, path)

            with open(path, "r") as f:
                config = json.load(f)
                print(f"[Config] Loaded from {path}")
                return config
        except Exception as e:
            print(f"[Config] Failed to load config: {e}")
            print("[Config] Using default values")
            return {
                "robot_ip": "192.168.1.151",
                "tcp_speed": 300,
                "tcp_acc": 1000,
                "angle_speed": 20,
                "angle_acc": 500,
                "click_control": {
                    "safe_height": 150,
                    "pick_height": -5,
                    "workspace_min_x": 0,
                    "workspace_max_x": 300,
                    "workspace_min_y": 0,
                    "workspace_max_y": 300
                }
            }

    def wait_for_yolo_calibration(self):
        """Wait for YOLO calibration to complete by monitoring file modification time."""
        print("\n" + "="*60)
        print("[Calibration] Waiting for YOLO camera calibration...")
        print("[Calibration] Please run yolo-mouse-v2.py now")
        print("[Calibration] Robot will remain at calibration position")
        print("="*60)

        # Look for files in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        homography_file = os.path.join(script_dir, "homography_auto.pkl")
        det_to_robot_file = os.path.join(script_dir, "homography_det_to_robot.pkl")

        # Get initial modification time if files exist
        initial_mtime = None
        if os.path.exists(homography_file):
            initial_mtime = os.path.getmtime(homography_file)
            print(f"[Calibration] Found existing calibration file")
            print(f"[Calibration] Waiting for YOLO to overwrite with new calibration...")
        else:
            print(f"[Calibration] No existing calibration - waiting for YOLO to create files...")

        wait_count = 0
        calibration_complete = False

        while not calibration_complete:
            time.sleep(1)
            wait_count += 1

            # Check if file exists and has been modified
            if os.path.exists(homography_file):
                current_mtime = os.path.getmtime(homography_file)

                if initial_mtime is None:
                    # File was just created
                    calibration_complete = True
                    print(f"\n[Calibration] New calibration file detected!")
                elif current_mtime > initial_mtime:
                    # File was modified (overwritten)
                    calibration_complete = True
                    print(f"\n[Calibration] Calibration file updated!")
                else:
                    # File exists but hasn't been modified yet
                    if wait_count % 5 == 0:
                        print(f"[Calibration] Waiting for new calibration... ({wait_count}s)")
            else:
                # File doesn't exist yet
                if wait_count % 5 == 0:
                    print(f"[Calibration] Waiting for calibration files... ({wait_count}s)")

        # Give it a moment to ensure both files are fully written
        time.sleep(1)

        print("\n" + "="*60)
        print("[Calibration] ✅ YOLO calibration detected!")
        print(f"[Calibration] Homography file: {homography_file}")
        print("[Calibration] Camera calibration complete")
        print("="*60 + "\n")

    def load_homography(self):
        """Load homography matrices for two-step coordinate transformation."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        homography_auto_file = os.path.join(script_dir, "homography_auto.pkl")
        homography_det_file = os.path.join(script_dir, "homography_det_to_robot.pkl")

        self.H_auto = None
        self.H_det_to_robot = None

        # Load first transformation (camera → detection workspace)
        try:
            with open(homography_auto_file, "rb") as f:
                self.H_auto = pickle.load(f)
            print(f"[Homography] ✅ Loaded camera-to-workspace transformation")
            print(f"[Homography] File: {homography_auto_file}")
        except FileNotFoundError:
            print(f"[Homography] ⚠️  {homography_auto_file} not found - will be created after calibration")
        except Exception as e:
            print(f"[Homography] ⚠️  Could not load camera-to-workspace: {e}")

        # Load second transformation (detection workspace → robot coordinates)
        try:
            with open(homography_det_file, "rb") as f:
                self.H_det_to_robot = pickle.load(f)
            print(f"[Homography] ✅ Loaded workspace-to-robot transformation")
            print(f"[Homography] File: {homography_det_file}")
        except FileNotFoundError:
            print(f"[Homography] ⚠️  {homography_det_file} not found - will be created after calibration")
        except Exception as e:
            print(f"[Homography] ⚠️  Could not load workspace-to-robot: {e}")

        if self.H_auto is not None and self.H_det_to_robot is not None:
            print(f"[Homography] Two-step transformation ready: Camera → Workspace → Robot")
            print(f"[Homography] This accounts for ~90° rotation between detection and robot")

    def transform_detection_to_robot(self, det_x, det_y):
        """
        Transform detection workspace coordinates to robot coordinates.

        This receives workspace coordinates [0-300mm] from YOLO detection
        and transforms them to actual robot coordinates using H_det_to_robot.

        Note: YOLO already applies H_auto internally (camera pixels → workspace),
        so this only needs to apply the second transformation (workspace → robot).

        Args:
            det_x: X coordinate from detection workspace (0-300mm)
            det_y: Y coordinate from detection workspace (0-300mm)

        Returns:
            (robot_x, robot_y): Transformed coordinates in robot space (mm)
        """
        if self.H_det_to_robot is None:
            raise RuntimeError("Homography matrix not loaded! Cannot transform coordinates.")

        # Transform workspace coordinates to robot coordinates
        det_pt = np.array([[det_x, det_y]], dtype=np.float32).reshape(-1, 1, 2)
        robot_pt = cv2.perspectiveTransform(det_pt, self.H_det_to_robot)
        robot_x, robot_y = robot_pt[0][0]

        return robot_x, robot_y

    def is_robot_position_safe(self, robot_x, robot_y):
        """
        Check if robot coordinates are within workspace boundaries.

        Args:
            robot_x, robot_y: Robot coordinates (mm)

        Returns:
            bool: True if position is safe, False otherwise
        """
        return (ROBOT_MIN_X <= robot_x <= ROBOT_MAX_X and
                ROBOT_MIN_Y <= robot_y <= ROBOT_MAX_Y)

    def calculate_optimal_pick_angle(self, object_angle, object_width, object_height):
        """
        Calculate the optimal gripper angle for picking based on object dimensions.

        The gripper should align to grip the narrower dimension of the object.

        Args:
            object_angle: Detected object angle in degrees (0-180)
            object_width: Object width in mm
            object_height: Object height in mm

        Returns:
            float: Optimal gripper yaw angle in degrees
        """
        if object_width <= 0 or object_height <= 0:
            # No dimension data, use object angle directly
            return object_angle

        # Determine which dimension is smaller (should be gripped)
        if object_width < object_height:
            # Width is smaller - gripper should align with object angle to grip width
            gripper_angle = object_angle
            print(f"[Pick Logic] Width ({object_width:.1f}mm) < Height ({object_height:.1f}mm)")
            print(f"[Pick Logic] Gripper aligns WITH object angle: {gripper_angle:.1f}°")
        else:
            # Height is smaller - gripper should rotate 90° RIGHT (subtract) to grip height
            gripper_angle = object_angle - 90
            # Normalize to -180 to 180 range (robot accepts negative angles)
            if gripper_angle < -180:
                gripper_angle += 360
            print(f"[Pick Logic] Height ({object_height:.1f}mm) < Width ({object_width:.1f}mm)")
            print(f"[Pick Logic] Gripper rotates 90° RIGHT from object: {object_angle:.1f}° → {gripper_angle:.1f}°")

        return gripper_angle

    def calculate_optimal_inspect_angle(self, object_angle):
        """
        Calculate the optimal camera viewing angle for inspection.

        The camera is mounted on the robot with an angular offset from the gripper.
        To properly view the object, we need to rotate the robot by:
        robot_yaw = object_angle + camera_mounting_offset

        Args:
            object_angle: Detected object angle in degrees (0-180)

        Returns:
            float: Optimal camera yaw angle in degrees
            
        Calculation Process:
        1. Object detected at angle (e.g., 104.7°)
        2. Camera mounted at offset angle from gripper (e.g., 90°)
        3. Robot yaw = object_angle + inspection_angle_offset
        4. This rotates robot so inspection camera points at object
        """
        # Get camera angle offset from config
        camera_config = self.config.get("camera_offset", {})
        camera_angle_offset = camera_config.get("inspection_angle_offset", 90)
        
        # Calculate robot yaw: align with object angle + camera offset
        inspect_angle = (object_angle + camera_angle_offset) % 360
        
        print(f"[Inspect Logic] Object angle: {object_angle:.1f}°")
        print(f"[Inspect Logic] Camera offset: {camera_angle_offset:.1f}°")
        print(f"[Inspect Logic] Calculated robot yaw: {inspect_angle:.1f}°")
        print(f"[Inspect Logic] → Robot rotates to this angle so camera views object")
        
        return inspect_angle

    def connect_robot(self):
        """Connect to the xArm robot."""
        try:
            print(f"[Robot] Connecting to xArm at {self.robot_ip}...")
            self._arm = XArmAPI(self.robot_ip)
            time.sleep(0.5)
            print("[Robot] ✅ Connected successfully!")
        except Exception as e:
            print(f"[Robot] ❌ Connection failed: {e}")
            raise

    def initialize_robot(self):
        """Initialize robot to ready state."""
        try:
            print("[Robot] Initializing...")
            self._arm.motion_enable(True)
            self._arm.clean_warn()
            self._arm.clean_error()
            self._arm.set_mode(0)
            self._arm.set_state(0)
            time.sleep(1)

            tcp_speed = self.config.get("tcp_speed", 300)
            tcp_acc = self.config.get("tcp_acc", 2000)
            angle_speed = self.config.get("angle_speed", 20)
            angle_acc = self.config.get("angle_acc", 500)

            self._arm.set_tcp_maxacc(tcp_acc)
            self._arm.set_joint_maxacc(angle_acc)

            print(f"[Robot] TCP speed: {tcp_speed} mm/s, acc: {tcp_acc} mm/s²")
            print(f"[Robot] Joint speed: {angle_speed}°/s, acc: {angle_acc}°/s²")
            print("[Robot] ✅ Initialization complete!")

        except Exception as e:
            print(f"[Robot] ❌ Initialization failed: {e}")
            raise

    def go_home(self, wait=True):
        """Move robot to home position.

        Args:
            wait: If True, wait for movement to complete. If False, return immediately.
        """
        try:
            print("[Robot] Moving to home position...")
            self._arm.set_servo_angle(angle=self.home_position, speed=80, wait=wait)
            if wait:
                print("[Robot] ✅ Home position reached")
            else:
                print("[Robot] Home position command sent (not waiting)")
            return True
        except Exception as e:
            print(f"[Robot] ❌ Failed to go home: {e}")
            return False

    def go_to_calibration_position(self):
        """Move robot to calibration position for camera calibration."""
        try:
            print("\n" + "="*60)
            print("[Calibration] Moving to calibration position...")
            print(f"[Calibration] Target: X={self.calibration_position['x']:.1f}, "
                  f"Y={self.calibration_position['y']:.1f}, "
                  f"Z={self.calibration_position['z']:.1f} mm")
            print(f"[Calibration] Orientation: Roll={self.calibration_position['roll']:.1f}°, "
                  f"Pitch={self.calibration_position['pitch']:.1f}°, "
                  f"Yaw={self.calibration_position['yaw']:.1f}°")
            print("="*60)
            
            code = self._arm.set_position(
                x=self.calibration_position['x'],
                y=self.calibration_position['y'],
                z=self.calibration_position['z'],
                roll=self.calibration_position['roll'],
                pitch=self.calibration_position['pitch'],
                yaw=self.calibration_position['yaw'],
                speed=self.config.get("tcp_speed", 300),
                wait=True
            )
            
            if code == 0:
                print("[Calibration] ✅ Calibration position reached")
                print("[Calibration] Robot ready for camera calibration")
                print("="*60 + "\n")
                return True
            else:
                print(f"[Calibration] ❌ Failed with code: {code}")
                print("="*60 + "\n")
                return False
                
        except Exception as e:
            print(f"[Calibration] ❌ Error: {e}")
            print("="*60 + "\n")
            return False

    def move_to_position(self, det_x, det_y, z=None):
        """
        Move TCP to specified position using detection coordinates.
        
        Args:
            det_x, det_y: Position in detection coordinate system (mm)
            z: Z height (optional, uses safe_height if None)
        """
        try:
            if z is None:
                z = self.safe_height

            # Transform detection coordinates to robot coordinates
            robot_x, robot_y = self.transform_detection_to_robot(det_x, det_y)

            # Check if robot position is within workspace
            if not self.is_robot_position_safe(robot_x, robot_y):
                print(f"\n{'='*60}")
                print(f"[Move] ❌ POSITION OUT OF BOUNDS")
                print(f"[Move] Robot position: ({robot_x:.1f}, {robot_y:.1f}) mm")
                print(f"[Move] Workspace limits: X=[{ROBOT_MIN_X:.1f}-{ROBOT_MAX_X:.1f}], Y=[{ROBOT_MIN_Y:.1f}-{ROBOT_MAX_Y:.1f}]")
                print(f"[Move] Cannot proceed - position outside robot workspace!")
                print(f"{'='*60}\n")
                return False

            print(f"\n{'='*60}")
            print(f"[Move] Detection coords: ({det_x:.1f}, {det_y:.1f}) mm")
            print(f"[Move] Robot coords: ({robot_x:.1f}, {robot_y:.1f}) mm")
            print(f"[Move] Target height: {z:.1f} mm")

            current_pos = self._arm.get_position()[1][:3]
            print(f"[Move] Current position: ({current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f})")

            target_pos = [robot_x, robot_y, z]
            print(f"[Move] Moving to: ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")

            code = self._arm.set_position(
                x=target_pos[0],
                y=target_pos[1],
                z=target_pos[2],
                roll=180,
                pitch=0,
                yaw=0,
                speed=self.config.get("tcp_speed", 300),
                wait=True
            )

            if code == 0:
                print(f"[Move] ✅ Movement successful!")
                print(f"{'='*60}\n")
                return True
            else:
                print(f"[Move] ❌ Movement failed with code: {code}")
                print(f"{'='*60}\n")
                return False

        except Exception as e:
            print(f"[Robot Error] Move failed: {e}")
            print(f"{'='*60}\n")
            return False

    def pick_sequence(self, det_x, det_y, object_angle=0.0, object_width=0.0, object_height=0.0):
        """
        Execute pick sequence with automatic gripper angle adjustment.

        Args:
            det_x, det_y: Position in detection coordinates (mm)
            object_angle: Detected object angle in degrees (0-180)
            object_width: Object width in mm
            object_height: Object height in mm
        """
        try:
            # Transform to robot coordinates
            robot_x, robot_y = self.transform_detection_to_robot(det_x, det_y)

            # Check if robot position is within workspace
            if not self.is_robot_position_safe(robot_x, robot_y):
                print(f"\n{'='*60}")
                print(f"[Pick] ❌ POSITION OUT OF BOUNDS")
                print(f"[Pick] Robot position: ({robot_x:.1f}, {robot_y:.1f}) mm")
                print(f"[Pick] Workspace limits: X=[{ROBOT_MIN_X:.1f}-{ROBOT_MAX_X:.1f}], Y=[{ROBOT_MIN_Y:.1f}-{ROBOT_MAX_Y:.1f}]")
                print(f"[Pick] Cannot proceed - position outside robot workspace!")
                print(f"{'='*60}\n")
                return False

            # Calculate optimal gripper angle based on object dimensions
            gripper_angle = self.calculate_optimal_pick_angle(object_angle, object_width, object_height)

            # Note: Lite6 gripper uses simple open/close commands (no position control)
            if object_width > 0 and object_height > 0:
                print(f"[Pick] Object size: {object_width:.1f}x{object_height:.1f} mm")

            print(f"\n{'='*60}")
            print(f"[Pick] PICK SEQUENCE START")
            print(f"[Pick] Detection coords: ({det_x:.1f}, {det_y:.1f}) mm")
            print(f"[Pick] Robot coords: ({robot_x:.1f}, {robot_y:.1f}) mm")
            print(f"[Pick] Object angle: {object_angle:.1f}° → Gripper angle: {gripper_angle:.1f}°")
            print(f"{'='*60}")

            # Step 1: Move to safe height above target
            print(f"[Pick] Step 1/4: Moving to safe height ({self.safe_height}mm)...")
            code = self._arm.set_position(
                x=robot_x, y=robot_y, z=self.safe_height,
                roll=180, pitch=0, yaw=gripper_angle,
                speed=self.config.get("tcp_speed", 300),
                wait=True
            )
            if code != 0:
                print(f"[Pick] ❌ Failed at step 1")
                return False

            # Step 2: Open gripper (Lite6 gripper - no position control, just open/close)
            print(f"[Pick] Step 2/4: Opening gripper...")
            self._arm.open_lite6_gripper()
            time.sleep(1.0)

            # Step 3: Move down to pick height
            print(f"[Pick] Step 3/4: Moving to pick height ({self.pick_height}mm)...")
            code = self._arm.set_position(
                x=robot_x, y=robot_y, z=self.pick_height,
                roll=180, pitch=0, yaw=gripper_angle,
                speed=100,
                wait=True
            )
            if code != 0:
                print(f"[Pick] ❌ Failed at step 3")
                return False

            # Step 4: Close gripper
            print(f"[Pick] Step 4/5: Closing gripper...")
            self._arm.close_lite6_gripper()
            time.sleep(1.0)

            # Step 5: Update TCP load for picked object
            print(f"[Pick] Step 5/5: Updating TCP load...")
            self._arm.set_tcp_load(0.35, [0, 0, 40])

            # Return to safe height
            print(f"[Pick] Returning to safe height...")
            code = self._arm.set_position(
                x=robot_x, y=robot_y, z=self.safe_height,
                roll=180, pitch=0, yaw=gripper_angle,
                speed=self.config.get("tcp_speed", 300),
                wait=True
            )

            print(f"[Pick] ✅ PICK SEQUENCE COMPLETE")
            print(f"{'='*60}\n")
            return True

        except Exception as e:
            print(f"[Robot Error] Pick sequence failed: {e}")
            print(f"{'='*60}\n")
            return False

    def place_sequence(self, det_x, det_y, maintain_angle=False, object_angle=0.0):
        """
        Execute place sequence.

        Args:
            det_x, det_y: Position in detection coordinates (mm)
            maintain_angle: If True, maintain the angle from pick operation
            object_angle: Angle to use for placement (degrees)
        """
        try:
            # Transform to robot coordinates
            robot_x, robot_y = self.transform_detection_to_robot(det_x, det_y)

            # Check if robot position is within workspace
            if not self.is_robot_position_safe(robot_x, robot_y):
                print(f"\n{'='*60}")
                print(f"[Place] ❌ POSITION OUT OF BOUNDS")
                print(f"[Place] Robot position: ({robot_x:.1f}, {robot_y:.1f}) mm")
                print(f"[Place] Workspace limits: X=[{ROBOT_MIN_X:.1f}-{ROBOT_MAX_X:.1f}], Y=[{ROBOT_MIN_Y:.1f}-{ROBOT_MAX_Y:.1f}]")
                print(f"[Place] Cannot proceed - position outside robot workspace!")
                print(f"{'='*60}\n")
                return False

            # Determine yaw angle - use 0 if not maintaining angle
            gripper_yaw = object_angle if maintain_angle else 0.0

            print(f"\n{'='*60}")
            print(f"[Place] PLACE SEQUENCE START")
            print(f"[Place] Detection coords: ({det_x:.1f}, {det_y:.1f}) mm")
            print(f"[Place] Robot coords: ({robot_x:.1f}, {robot_y:.1f}) mm")
            if maintain_angle:
                print(f"[Place] Maintaining angle: {gripper_yaw:.1f}°")
            else:
                print(f"[Place] Placing straight down (yaw: 0°)")
            print(f"{'='*60}")

            # Step 1: Move to safe height above target
            print(f"[Place] Step 1/4: Moving to safe height ({self.safe_height}mm)...")
            code = self._arm.set_position(
                x=robot_x, y=robot_y, z=self.safe_height,
                roll=180, pitch=0, yaw=gripper_yaw,
                speed=self.config.get("tcp_speed", 300),
                wait=True
            )
            if code != 0:
                print(f"[Place] ❌ Failed at step 1")
                return False

            # Step 2: Move down to pick height
            print(f"[Place] Step 2/4: Moving to place height ({self.pick_height}mm)...")
            code = self._arm.set_position(
                x=robot_x, y=robot_y, z=self.pick_height,
                roll=180, pitch=0, yaw=gripper_yaw,
                speed=100,
                wait=True
            )
            if code != 0:
                print(f"[Place] ❌ Failed at step 2")
                return False

            # Step 3: Open gripper
            print(f"[Place] Step 3/4: Opening gripper...")
            self._arm.open_lite6_gripper()
            time.sleep(1.0)

            # Step 4: Reset TCP load
            print(f"[Place] Step 4/5: Resetting TCP load...")
            self._arm.set_tcp_load(0.277, [0, 0, 30])

            # Step 5: Return to safe height
            print(f"[Place] Step 5/5: Returning to safe height...")
            code = self._arm.set_position(
                x=robot_x, y=robot_y, z=self.safe_height,
                roll=180, pitch=0, yaw=gripper_yaw,
                speed=self.config.get("tcp_speed", 300),
                wait=True
            )

            # Stop gripper
            self._arm.stop_lite6_gripper()

            print(f"[Place] ✅ PLACE SEQUENCE COMPLETE")
            print(f"{'='*60}\n")
            return True

        except Exception as e:
            print(f"[Robot Error] Place sequence failed: {e}")
            print(f"{'='*60}\n")
            return False

    def inspect_sequence(self, target_det_x, target_det_y, object_angle=0.0, offset_x=CAMERA_OFFSET_X, offset_y=CAMERA_OFFSET_Y):
        """
        Execute inspection sequence - move gripper so inspection camera views target.

        The inspection camera is mounted on the gripper with an offset.
        To view the target at the camera center, we need to position the gripper
        at: gripper_position = target_position - camera_offset

        Args:
            target_det_x, target_det_y: Target position in detection coordinates (mm)
            object_angle: Object angle in degrees (0-180) - used to calculate optimal camera angle
            offset_x, offset_y: Camera offset from gripper center point (mm)
        """
        try:
            # Calculate gripper position in detection coordinates
            # Gripper needs to be at target - offset
            gripper_det_x = target_det_x - offset_x
            gripper_det_y = target_det_y - offset_y

            # Transform gripper position to robot coordinates
            robot_x, robot_y = self.transform_detection_to_robot(gripper_det_x, gripper_det_y)

            # Check if robot position is within workspace
            if not self.is_robot_position_safe(robot_x, robot_y):
                print(f"\n{'='*60}")
                print(f"[Inspect] ❌ POSITION OUT OF BOUNDS")
                print(f"[Inspect] Robot position: ({robot_x:.1f}, {robot_y:.1f}) mm")
                print(f"[Inspect] Workspace limits: X=[{ROBOT_MIN_X:.1f}-{ROBOT_MAX_X:.1f}], Y=[{ROBOT_MIN_Y:.1f}-{ROBOT_MAX_Y:.1f}]")
                print(f"[Inspect] Cannot proceed - position outside robot workspace!")
                print(f"{'='*60}\n")
                return False

            # Calculate optimal camera viewing angle
            camera_angle = self.calculate_optimal_inspect_angle(object_angle)

            print(f"\n{'='*60}")
            print(f"[Inspect] INSPECTION SEQUENCE START")
            print(f"[Inspect] Target (detection): ({target_det_x:.1f}, {target_det_y:.1f}) mm")
            print(f"[Inspect] Object angle: {object_angle:.1f}° → Camera angle: {camera_angle:.1f}°")
            print(f"[Inspect] Camera offset: ({offset_x:.1f}, {offset_y:.1f}) ± {CAMERA_OFFSET_ERROR:.1f} mm")
            print(f"[Inspect] Gripper position (detection): ({gripper_det_x:.1f}, {gripper_det_y:.1f}) mm")
            print(f"[Inspect] Gripper position (robot): ({robot_x:.1f}, {robot_y:.1f}) mm")
            print(f"[Inspect] Inspection height: {self.inspect_height} mm")
            print(f"{'='*60}")

            # Step 1: Move to safe height first
            print(f"[Inspect] Step 1/2: Moving to safe height...")
            current_pos = self._arm.get_position()[1]
            code = self._arm.set_position(
                x=current_pos[0],
                y=current_pos[1],
                z=self.safe_height,
                roll=180,
                pitch=0,
                yaw=camera_angle,
                speed=self.config.get("tcp_speed", 300),
                wait=True
            )
            if code != 0:
                print(f"[Inspect] ❌ Failed at step 1")
                return False

            # Step 2: Move to inspection position with calculated camera angle
            print(f"[Inspect] Step 2/2: Moving to inspection position (angle: {camera_angle:.1f}°)...")
            code = self._arm.set_position(
                x=robot_x,
                y=robot_y,
                z=self.inspect_height,
                roll=180,
                pitch=0,
                yaw=camera_angle,
                speed=self.config.get("tcp_speed", 300),
                wait=True
            )
            
            if code == 0:
                print(f"[Inspect] ✅ INSPECTION POSITION REACHED")
                print(f"[Inspect] The inspection camera should now be viewing the target")
                print(f"[Inspect] Position error tolerance: ±{CAMERA_OFFSET_ERROR:.1f} mm")
                print(f"{'='*60}\n")
                return True
            else:
                print(f"[Inspect] ❌ Movement failed with code: {code}")
                print(f"{'='*60}\n")
                return False

        except Exception as e:
            print(f"[Robot Error] Inspection sequence failed: {e}")
            print(f"{'='*60}\n")
            return False

    def check_and_recover(self):
        """Check robot state and recover from errors if needed."""
        try:
            state = self._arm.get_state()
            if state[0] == 0 and state[1] in [3, 4]:
                print("[Robot] Error detected! Attempting recovery...")
                self._arm.clean_warn()
                self._arm.clean_error()
                self._arm.motion_enable(True)
                self._arm.set_mode(0)
                self._arm.set_state(0)
                time.sleep(0.5)
                print("[Robot] Recovery successful.")
                return True
            return False
        except Exception as e:
            print(f"[Robot] Recovery check failed: {e}")
            return False

    def shutdown(self, emergency=False):
        """Shutdown robot safely.

        Args:
            emergency: If True, skip waiting for movements (for Ctrl+C shutdown)
        """
        print("\n[Robot] Shutting down...")
        try:
            # During emergency shutdown, don't wait for home position to complete
            self.go_home(wait=not emergency)
            self._arm.set_state(4)
            print("[Robot] Shutdown complete.")
        except KeyboardInterrupt:
            print("[Robot] Shutdown interrupted - stopping immediately")
            try:
                self._arm.set_state(4)
            except:
                pass
        except Exception as e:
            print(f"[Robot] Shutdown error: {e}")


class XArmClickController:
    """Main application that monitors clicks and controls the xArm with auto-angle adjustment and inspection."""

    def __init__(self):
        """Initialize controller with automatic button-based mode selection."""
        self.click_manager = ClickDataManager()
        self.inspect_manager = InspectDataManager()
        self.arm = XArmController()
        self.running = False
        self.emergency_stop = False
        self.last_processed_click_time = 0
        self.last_processed_inspect_time = 0

        # Pick/Place toggle state
        self.has_picked = False  # Track whether robot has picked an object
        self.last_picked_angle = 0.0  # Store angle from last pick for place operation
        self.last_picked_x = 0.0  # Store pick position for place
        self.last_picked_y = 0.0

        # Workspace limits
        click_config = self.arm.config.get("click_control", {})
        self.workspace_min_x = click_config.get("workspace_min_x", 0)
        self.workspace_max_x = click_config.get("workspace_max_x", 300)
        self.workspace_min_y = click_config.get("workspace_min_y", 0)
        self.workspace_max_y = click_config.get("workspace_max_y", 300)

        print("\n" + "="*60)
        print("xArm CONTROLLER - PICK/PLACE TOGGLE + INSPECTION MODE")
        print("="*60)
        print("Mouse Button Controls:")
        print("  • LEFT CLICK   → Move to position")
        print("  • RIGHT CLICK  → Pick/Place TOGGLE")
        print("    - First click:  PICK object")
        print("    - Second click: PLACE object")
        print("    - Third click:  PICK again (repeats)")
        print("\nKeyboard Controls:")
        print("  • 'T' KEY → Inspect selected object")
        print("\nNote: You MUST place before you can pick again!")
        print(f"\nInspection:")
        print(f"  • Camera offset: ({CAMERA_OFFSET_X:.1f}, {CAMERA_OFFSET_Y:.1f}) ± {CAMERA_OFFSET_ERROR:.1f} mm")
        print(f"\nWorkspace: X=[{self.workspace_min_x}-{self.workspace_max_x}], "
              f"Y=[{self.workspace_min_y}-{self.workspace_max_y}]")
        print(f"Safe height: {self.arm.safe_height}mm, Pick height: {self.arm.pick_height}mm")
        print(f"Inspection height: {self.arm.inspect_height}mm")
        print(f"Coordinate transformation: Using homography matrix (handles rotation)")
        print("="*60 + "\n")

    def is_position_safe(self, x, y):
        """Check if position is within safe workspace bounds."""
        return (self.workspace_min_x <= x <= self.workspace_max_x and
                self.workspace_min_y <= y <= self.workspace_max_y)

    def process_click(self, click_data):
        """Process a new click and move the arm based on button with angle adjustment."""
        x = click_data.get("click_x", 0)
        y = click_data.get("click_y", 0)
        button = click_data.get("button", "left")
        angle = click_data.get("angle", 0.0)
        width = click_data.get("width", 0.0)
        height = click_data.get("height", 0.0)
        timestamp = click_data.get("timestamp", 0)

        # Ignore old or duplicate clicks
        if timestamp <= self.last_processed_click_time:
            return

        if width > 0 and height > 0:
            print(f"\n[CLICK DETECTED] Position: ({x:.1f}, {y:.1f}) mm - Button: {button.upper()} - Angle: {angle:.1f}° - Size: {width:.1f}x{height:.1f}mm")
        else:
            print(f"\n[CLICK DETECTED] Position: ({x:.1f}, {y:.1f}) mm - Button: {button.upper()} - Angle: {angle:.1f}°")

        # Safety check
        if not self.is_position_safe(x, y):
            print(f"[WARNING] Position ({x:.1f}, {y:.1f}) is outside safe workspace!")
            print(f"[WARNING] Safe range: X=[{self.workspace_min_x}-{self.workspace_max_x}], "
                  f"Y=[{self.workspace_min_y}-{self.workspace_max_y}]")
            print("[WARNING] Ignoring click for safety.")
            self.click_manager.mark_processed()
            self.last_processed_click_time = timestamp
            return

        # Check and recover from any errors
        self.arm.check_and_recover()

        # Execute based on button
        success = False
        if button == "left":
            print("[INFO] Moving arm to clicked position...")
            success = self.arm.move_to_position(x, y)

        elif button == "middle":
            # Middle button = (unused - reserved for future features)
            print(f"[INFO] Middle click not assigned (use 'T' key for inspection)")
            success = True

        elif button == "right":
            # Right button = Pick/Place toggle
            if not self.has_picked:
                # First right-click: PICK
                print(f"[INFO] Executing PICK sequence (object angle: {angle:.1f}°)...")
                success = self.arm.pick_sequence(x, y, object_angle=angle, object_width=width, object_height=height)
                if success:
                    self.has_picked = True
                    self.last_picked_angle = angle
                    self.last_picked_x = x
                    self.last_picked_y = y
                    print(f"[STATE] ✅ Object picked! Next right-click will PLACE.")
            else:
                # Second right-click: PLACE
                print(f"[INFO] Executing PLACE sequence (straight down)...")
                success = self.arm.place_sequence(x, y, maintain_angle=False, object_angle=self.last_picked_angle)
                if success:
                    self.has_picked = False
                    print(f"[STATE] ✅ Object placed! Next right-click will PICK.")
        else:
            print(f"[WARNING] Unknown button: {button}")

        if success:
            print("[SUCCESS] Operation completed!")
        else:
            print("[ERROR] Operation failed!")

        # Mark as processed
        self.click_manager.mark_processed()
        self.last_processed_click_time = timestamp

    def process_inspect(self, inspect_data):
        """Process inspection command."""
        target_x = inspect_data.get("target_x", 0)
        target_y = inspect_data.get("target_y", 0)
        angle = inspect_data.get("angle", 0.0)
        offset_x = inspect_data.get("offset_x", CAMERA_OFFSET_X)
        offset_y = inspect_data.get("offset_y", CAMERA_OFFSET_Y)
        timestamp = inspect_data.get("timestamp", 0)

        # Ignore old or duplicate commands
        if timestamp <= self.last_processed_inspect_time:
            return

        print(f"\n[INSPECT COMMAND] Target: ({target_x:.1f}, {target_y:.1f}) mm - Angle: {angle:.1f}°")

        # Check if target position is safe
        if not self.is_position_safe(target_x, target_y):
            print(f"[WARNING] Target position ({target_x:.1f}, {target_y:.1f}) is outside safe workspace!")
            print("[WARNING] Ignoring inspection command for safety.")
            self.inspect_manager.mark_processed()
            self.last_processed_inspect_time = timestamp
            return

        # Check and recover from any errors
        self.arm.check_and_recover()

        # Execute inspection sequence with object angle
        print("[INFO] Executing inspection sequence...")
        success = self.arm.inspect_sequence(target_x, target_y, object_angle=angle, offset_x=offset_x, offset_y=offset_y)

        if success:
            print("[SUCCESS] Inspection position reached!")
        else:
            print("[ERROR] Inspection failed!")

        # Mark as processed
        self.inspect_manager.mark_processed()
        self.last_processed_inspect_time = timestamp

    def monitor_commands(self):
        """Monitor shared memory for new clicks and inspection commands."""
        print("\n" + "="*60)
        print("MONITORING FOR COMMANDS")
        print("="*60)
        print("Waiting for mouse clicks and inspection commands...")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")

        self.running = True
        check_count = 0

        try:
            while self.running:
                # Read click data
                click_data = self.click_manager.read_click()

                # Read inspection data
                inspect_data = self.inspect_manager.read_inspect()

                # Debug counter
                check_count += 1

                # Check for new unprocessed click
                if not click_data.get("processed", True):
                    print(f"[DEBUG] Found unprocessed click: {click_data}")
                    self.process_click(click_data)

                # Check for new unprocessed inspection command
                if not inspect_data.get("processed", True):
                    print(f"[DEBUG] Found unprocessed inspect command: {inspect_data}")
                    self.process_inspect(inspect_data)

                time.sleep(0.05)  # Check at 20 Hz

        except KeyboardInterrupt:
            print("\n\n[INFO] Stopping arm controller...")
            self.emergency_stop = True
        finally:
            self.stop()

    def stop(self):
        """Stop the controller and cleanup."""
        self.running = False
        print("\n[INFO] Stopping controller...")
        # Use emergency=True if stopped via Ctrl+C to avoid blocking
        emergency = getattr(self, 'emergency_stop', False)
        self.arm.shutdown(emergency=emergency)
        self.click_manager.cleanup()
        self.inspect_manager.cleanup()
        print("[INFO] Controller stopped.\n")


def main():
    """Main entry point."""
    try:
        controller = XArmClickController()
        controller.monitor_commands()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()