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
        self.safe_height = click_config.get("safe_height", 200)
        self.pick_height = click_config.get("pick_height", 50)

        # Inspection height (Z offset of camera from workspace)
        self.inspect_height = 111.9  # mm - actual camera Z position

        # Load homography matrix for coordinate transformation
        self.H_det_to_robot = None
        self.load_homography()

        # Home position
        self.home_position = [1.5, 6.3, 45.5, -0.4, 41.0, -7.0]

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
                "tcp_acc": 2000,
                "angle_speed": 20,
                "angle_acc": 500,
                "click_control": {
                    "safe_height": 200,
                    "pick_height": 10,
                    "workspace_min_x": 0,
                    "workspace_max_x": 300,
                    "workspace_min_y": 0,
                    "workspace_max_y": 300
                }
            }

    def wait_for_yolo_calibration(self):
        """Wait for YOLO calibration to complete by checking for homography file."""
        print("\n" + "="*60)
        print("[Calibration] Waiting for YOLO camera calibration...")
        print("[Calibration] Please run yolo-mouse-v2.py now")
        print("[Calibration] Robot will remain at calibration position")
        print("="*60)

        # Look for files in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        homography_file = os.path.join(script_dir, "homography_auto.pkl")
        det_to_robot_file = os.path.join(script_dir, "homography_det_to_robot.pkl")

        # Remove old homography files if they exist
        for file_path in [homography_file, det_to_robot_file]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"[Calibration] Removed old file: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"[Calibration] Could not remove old file: {e}")

        wait_count = 0
        while not os.path.exists(homography_file):
            time.sleep(1)
            wait_count += 1
            if wait_count % 5 == 0:
                print(f"[Calibration] Still waiting... ({wait_count}s)")

        # Give it a moment to ensure both files are fully written
        time.sleep(1)
        
        print("\n" + "="*60)
        print("[Calibration] ✅ YOLO calibration detected!")
        print(f"[Calibration] Homography file found: {homography_file}")
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

    def go_home(self):
        """Move robot to home position."""
        try:
            print("[Robot] Moving to home position...")
            self._arm.set_servo_angle(angle=self.home_position, speed=20, wait=True)
            print("[Robot] ✅ Home position reached")
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

            # Calculate gripper opening based on object dimensions
            # The gripper needs to open wider than the narrower dimension of the object
            # Gripper position: 0 = fully closed, 850 = fully open
            # Add safety margin of 10mm
            if object_width > 0 and object_height > 0:
                # Use the smaller dimension (perpendicular to gripper fingers)
                grip_dimension = min(object_width, object_height)
                # Convert mm to gripper position (assuming ~85mm max opening at position 850)
                # gripper_pos = (grip_dimension + 10) / 85.0 * 850
                # Cap between 100 (min useful opening) and 850 (max opening)
                gripper_opening = int(min(850, max(200, (grip_dimension + 15) / 85.0 * 850)))
                print(f"[Pick] Object size: {object_width:.1f}x{object_height:.1f} mm")
                print(f"[Pick] Calculated gripper opening: {gripper_opening} (for {grip_dimension:.1f}mm grip)")
            else:
                # Default opening if no object dimensions provided
                gripper_opening = 850
                print(f"[Pick] No object dimensions, using default gripper opening: {gripper_opening}")

            print(f"\n{'='*60}")
            print(f"[Pick] PICK SEQUENCE START")
            print(f"[Pick] Detection coords: ({det_x:.1f}, {det_y:.1f}) mm")
            print(f"[Pick] Robot coords: ({robot_x:.1f}, {robot_y:.1f}) mm")
            print(f"[Pick] Object angle: {object_angle:.1f}°")
            print(f"{'='*60}")

            # Step 1: Move to safe height above target
            print(f"[Pick] Step 1/4: Moving to safe height ({self.safe_height}mm)...")
            code = self._arm.set_position(
                x=robot_x, y=robot_y, z=self.safe_height,
                roll=180, pitch=0, yaw=object_angle,
                speed=self.config.get("tcp_speed", 300),
                wait=True
            )
            if code != 0:
                print(f"[Pick] ❌ Failed at step 1")
                return False

            # Step 2: Open gripper to calculated opening
            print(f"[Pick] Step 2/4: Opening gripper to {gripper_opening}...")
            self._arm.set_gripper_position(gripper_opening, wait=True)
            time.sleep(0.5)

            # Step 3: Move down to pick height
            print(f"[Pick] Step 3/4: Moving to pick height ({self.pick_height}mm)...")
            code = self._arm.set_position(
                x=robot_x, y=robot_y, z=self.pick_height,
                roll=180, pitch=0, yaw=object_angle,
                speed=100,
                wait=True
            )
            if code != 0:
                print(f"[Pick] ❌ Failed at step 3")
                return False

            # Step 4: Close gripper
            print(f"[Pick] Step 4/4: Closing gripper...")
            self._arm.set_gripper_position(0, wait=True)
            time.sleep(0.5)

            # Return to safe height
            print(f"[Pick] Returning to safe height...")
            code = self._arm.set_position(
                x=robot_x, y=robot_y, z=self.safe_height,
                roll=180, pitch=0, yaw=object_angle,
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

    def place_sequence(self, det_x, det_y, maintain_angle=True, object_angle=0.0):
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

            print(f"\n{'='*60}")
            print(f"[Place] PLACE SEQUENCE START")
            print(f"[Place] Detection coords: ({det_x:.1f}, {det_y:.1f}) mm")
            print(f"[Place] Robot coords: ({robot_x:.1f}, {robot_y:.1f}) mm")
            print(f"[Place] Placement angle: {object_angle:.1f}°")
            print(f"{'='*60}")

            # Step 1: Move to safe height above target
            print(f"[Place] Step 1/4: Moving to safe height ({self.safe_height}mm)...")
            code = self._arm.set_position(
                x=robot_x, y=robot_y, z=self.safe_height,
                roll=180, pitch=0, yaw=object_angle,
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
                roll=180, pitch=0, yaw=object_angle,
                speed=100,
                wait=True
            )
            if code != 0:
                print(f"[Place] ❌ Failed at step 2")
                return False

            # Step 3: Open gripper
            print(f"[Place] Step 3/4: Opening gripper...")
            self._arm.set_gripper_position(850, wait=True)
            time.sleep(0.5)

            # Step 4: Return to safe height
            print(f"[Place] Step 4/4: Returning to safe height...")
            code = self._arm.set_position(
                x=robot_x, y=robot_y, z=self.safe_height,
                roll=180, pitch=0, yaw=object_angle,
                speed=self.config.get("tcp_speed", 300),
                wait=True
            )

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
            object_angle: Object angle in degrees (0-180) - gripper will match this angle
            offset_x, offset_y: Camera offset from gripper center point (mm)
        """
        try:
            print(f"\n{'='*60}")
            print(f"[Inspect] INSPECTION SEQUENCE START")
            print(f"[Inspect] Target (detection): ({target_det_x:.1f}, {target_det_y:.1f}) mm")
            print(f"[Inspect] Object angle: {object_angle:.1f}°")
            print(f"[Inspect] Camera offset: ({offset_x:.1f}, {offset_y:.1f}) ± {CAMERA_OFFSET_ERROR:.1f} mm")

            # Calculate gripper position in detection coordinates
            # Gripper needs to be at target - offset
            gripper_det_x = target_det_x - offset_x
            gripper_det_y = target_det_y - offset_y

            print(f"[Inspect] Gripper position (detection): ({gripper_det_x:.1f}, {gripper_det_y:.1f}) mm")

            # Transform gripper position to robot coordinates
            robot_x, robot_y = self.transform_detection_to_robot(gripper_det_x, gripper_det_y)

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
                yaw=object_angle,
                speed=self.config.get("tcp_speed", 300),
                wait=True
            )
            if code != 0:
                print(f"[Inspect] ❌ Failed at step 1")
                return False

            # Step 2: Move to inspection position with angle matching object
            print(f"[Inspect] Step 2/2: Moving to inspection position (angle: {object_angle:.1f}°)...")
            code = self._arm.set_position(
                x=robot_x,
                y=robot_y,
                z=self.inspect_height,
                roll=180,
                pitch=0,
                yaw=object_angle,
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

    def shutdown(self):
        """Shutdown robot safely."""
        print("\n[Robot] Shutting down...")
        try:
            self.go_home()
            self._arm.set_state(4)
            print("[Robot] Shutdown complete.")
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
        self.last_processed_click_time = 0
        self.last_processed_inspect_time = 0
        self.last_picked_angle = 0.0  # Store angle from last pick for place operation

        # Workspace limits
        click_config = self.arm.config.get("click_control", {})
        self.workspace_min_x = click_config.get("workspace_min_x", 0)
        self.workspace_max_x = click_config.get("workspace_max_x", 300)
        self.workspace_min_y = click_config.get("workspace_min_y", 0)
        self.workspace_max_y = click_config.get("workspace_max_y", 300)

        print("\n" + "="*60)
        print("xArm CONTROLLER - CLICK + INSPECTION MODE")
        print("="*60)
        print("Mouse Button Controls:")
        print("  • LEFT CLICK   → Move to position")
        print("  • MIDDLE CLICK → Pick sequence (auto-adjusts gripper angle)")
        print("  • RIGHT CLICK  → Place sequence (maintains gripper angle)")
        print("\nInspection Mode:")
        print("  • Press 'T' key → Inspect selected object")
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
            print(f"[INFO] Executing pick sequence with auto-angle (object angle: {angle:.1f}°)...")
            success = self.arm.pick_sequence(x, y, object_angle=angle, object_width=width, object_height=height)
            if success:
                self.last_picked_angle = angle  # Store for place operation
                
        elif button == "right":
            print(f"[INFO] Executing place sequence (maintaining angle: {self.last_picked_angle:.1f}°)...")
            success = self.arm.place_sequence(x, y, maintain_angle=True, object_angle=self.last_picked_angle)
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
        finally:
            self.stop()

    def stop(self):
        """Stop the controller and cleanup."""
        self.running = False
        print("\n[INFO] Stopping controller...")
        self.arm.shutdown()
        self.click_manager.cleanup()
        self.inspect_manager.cleanup()
        print("[INFO] Controller stopped.\n")


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("xArm ROBOT ARM CONTROLLER - WITH INSPECTION MODE")
    print("="*60)
    print("\nSTARTUP SEQUENCE:")
    print("  1. Robot connects and initializes")
    print("  2. Robot moves to calibration position")
    print("  3. Waits for YOLO camera calibration (run yolo-mouse-v1.9.py)")
    print("  4. Robot moves to home position")
    print("  5. Ready to receive commands")
    print("\nCOMMAND MODES:")
    print("  • LEFT CLICK   = Move to position")
    print("  • MIDDLE CLICK = Pick sequence (auto-adjusts gripper angle)")
    print("  • RIGHT CLICK  = Place sequence (maintains gripper angle)")
    print("  • T KEY        = Inspect selected object with gripper camera")
    print("\nThe gripper will automatically align with the detected object")
    print("orientation during pick operations!")
    print("\nInspection camera will be positioned to view the target object")
    print(f"using camera offset: ({CAMERA_OFFSET_X:.1f}, {CAMERA_OFFSET_Y:.1f}) ± {CAMERA_OFFSET_ERROR:.1f} mm")
    print("="*60 + "\n")

    time.sleep(2)

    try:
        controller = XArmClickController()
        controller.monitor_commands()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
