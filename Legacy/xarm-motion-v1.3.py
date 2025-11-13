"""
xArm Robot Controller - Click-based Movement with Automatic Gripper Angle Adjustment
Left Click = Move, Middle Click = Pick (with auto-angle), Right Click = Place
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
            initial_data = {"click_x": 0, "click_y": 0, "timestamp": 0, "processed": True, "button": "none", "angle": 0.0}
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
                return {"click_x": 0, "click_y": 0, "timestamp": 0, "processed": True, "button": "none", "angle": 0.0}

            json_bytes = bytes(self.shm.buf[4:4+length])
            json_str = json_bytes.decode('utf-8')
            data = json.loads(json_str)

            # Ensure button and angle keys exist
            if "button" not in data:
                data["button"] = "left"
            if "angle" not in data:
                data["angle"] = 0.0

            return data
        except Exception as e:
            print(f"[ERROR] Failed to read: {e}")
            return {"click_x": 0, "click_y": 0, "timestamp": 0, "processed": True, "button": "none", "angle": 0.0}

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


class XArmController:
    """xArm robot controller for click-based movement with automatic gripper angle adjustment."""

    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        self.robot_ip = self.config.get("robot_ip", "192.168.1.151")
        self._arm = None
        self.is_moving = False

        # Load heights from config
        click_config = self.config.get("click_control", {})
        self.safe_height = click_config.get("safe_height", 200)
        self.pick_height = click_config.get("pick_height", 50)

        # Load homography matrix for coordinate transformation
        self.H_det_to_robot = None
        self.load_homography()

        # Home position
        self.home_position = [1.5, 6.3, 45.5, -0.4, 41.0, -7.0]

        self.connect_robot()
        self.initialize_robot()
        print("[Robot] Moving to home position on startup...")
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

    def load_homography(self):
        """Load homography matrix for detection to robot coordinate transformation."""
        homography_file = "homography_det_to_robot.pkl"
        try:
            if not os.path.isabs(homography_file):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                homography_file = os.path.join(base_dir, "homography_det_to_robot.pkl")

            with open(homography_file, "rb") as f:
                self.H_det_to_robot = pickle.load(f)

            print(f"[Homography] ✅ Loaded coordinate transformation matrix")
            print(f"[Homography] File: {homography_file}")
            print(f"[Homography] This accounts for ~90° rotation between detection and robot")
        except FileNotFoundError:
            print(f"[Homography] ❌ ERROR: {homography_file} not found!")
            print(f"[Homography] Run 'create_alignment_matrix.py' first to generate it!")
            print(f"[Homography] Without this, robot will NOT move to correct positions!")
            raise FileNotFoundError(
                f"Required file '{homography_file}' not found. "
                f"Run create_alignment_matrix.py to generate the coordinate transformation matrix."
            )
        except Exception as e:
            print(f"[Homography] ERROR loading transformation matrix: {e}")
            raise

    def transform_detection_to_robot(self, det_x, det_y):
        """
        Transform detection coordinates to robot coordinates using homography.
        
        Args:
            det_x: X coordinate from detection system (mm)
            det_y: Y coordinate from detection system (mm)

        Returns:
            (robot_x, robot_y): Transformed coordinates in robot space (mm)
        """
        if self.H_det_to_robot is None:
            raise RuntimeError("Homography matrix not loaded! Cannot transform coordinates.")

        det_pt = np.array([[det_x, det_y]], dtype=np.float32).reshape(-1, 1, 2)
        robot_pt = cv2.perspectiveTransform(det_pt, self.H_det_to_robot).reshape(-1, 2)

        robot_x = robot_pt[0][0]
        robot_y = robot_pt[0][1]

        return robot_x, robot_y

    def calculate_gripper_yaw(self, object_angle_deg):
        """
        Calculate the gripper yaw angle based on the detected object angle.
        
        The object angle from detection represents the orientation of the object's
        long axis. We need to convert this to the robot's coordinate system and
        calculate the appropriate yaw angle for the gripper.
        
        Args:
            object_angle_deg: Object angle from detection (0-180 degrees)
            
        Returns:
            yaw_deg: Gripper yaw angle for the robot (-180 to 180 degrees)
        """
        # The detection angle is in the detection coordinate system
        # We need to account for the ~90° rotation between detection and robot coords
        
        # Option 1: Direct mapping (if detection and robot yaw align)
        # yaw_deg = object_angle_deg
        
        # Option 2: Apply 90° offset (if there's a consistent rotation)
        # This is common when camera and robot base are rotated relative to each other
        yaw_deg = object_angle_deg - 90.0
        
        # Normalize to [-180, 180] range
        while yaw_deg > 180:
            yaw_deg -= 360
        while yaw_deg < -180:
            yaw_deg += 360
            
        return yaw_deg

    def connect_robot(self):
        """Connect to xArm robot."""
        while True:
            try:
                print(f"[Robot] Connecting to xArm at {self.robot_ip}...")
                self._arm = XArmAPI(self.robot_ip, baud_checkset=False)
                print("[Robot] Connection established.")
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
            print("[Robot] Robot initialized and ready.")
        except Exception as e:
            print(f"[Robot] Initialization warning: {e}")

    def _check_code(self, code, label):
        """Check return code from robot command."""
        if code != 0:
            print(f"[Robot Error] {label} failed (code={code})")
            return False
        return True

    def go_home(self):
        """Move robot to home position."""
        print("[Robot] Moving to home position...")
        try:
            code = self._arm.set_servo_angle(
                angle=self.home_position,
                speed=self.config.get("angle_speed", 20),
                mvacc=self.config.get("angle_acc", 500),
                wait=True
            )
            if self._check_code(code, "Home"):
                print("[Robot] Home position reached.")
                return True
        except Exception as e:
            print(f"[Robot Error] Failed to go home: {e}")
        return False

    def move_to_position(self, x, y, z=None, roll=180, pitch=0, yaw=0):
        """Move robot to specified position using Cartesian coordinates."""
        self.is_moving = True

        if z is None:
            z = self.safe_height

        # Transform detection coordinates to robot coordinates using homography
        robot_x, robot_y = self.transform_detection_to_robot(x, y)
        robot_z = z

        print(f"\n{'='*60}")
        print(f"MOVING ARM TO POSITION")
        print(f"{'='*60}")
        print(f"Detection coords: X={x:.1f} mm, Y={y:.1f} mm")
        print(f"Robot coords:     X={robot_x:.1f} mm, Y={robot_y:.1f} mm, Z={robot_z:.1f} mm")
        print(f"Orientation:      Roll={roll}°, Pitch={pitch}°, Yaw={yaw}°")

        try:
            code = self._arm.set_position(
                robot_x, robot_y, robot_z, roll, pitch, yaw,
                speed=self.config.get("tcp_speed", 200),
                mvacc=self.config.get("tcp_acc", 2000),
                radius=0,
                wait=True
            )

            if self._check_code(code, "Move"):
                print("Status: Movement complete")
                print(f"{'='*60}\n")
                self.is_moving = False
                return True
            else:
                print("Status: Movement failed")
                print(f"{'='*60}\n")
                self.is_moving = False
                return False

        except Exception as e:
            print(f"[Robot Error] Movement exception: {e}")
            print(f"{'='*60}\n")
            self.is_moving = False
            return False

    def pick_sequence(self, x, y, object_angle=0.0):
        """
        Execute a pick sequence at the specified position with automatic gripper angle adjustment.
        
        Args:
            x: X coordinate (mm)
            y: Y coordinate (mm)
            object_angle: Detected object angle in degrees (0-180)
        """
        print(f"\n{'='*60}")
        print(f"EXECUTING PICK SEQUENCE WITH AUTO-ANGLE")
        print(f"{'='*60}")
        print(f"Position: X={x:.1f} mm, Y={y:.1f} mm")
        print(f"Object angle: {object_angle:.1f}°")
        
        # Calculate gripper yaw angle based on object angle
        gripper_yaw = self.calculate_gripper_yaw(object_angle)
        print(f"Calculated gripper yaw: {gripper_yaw:.1f}°")

        try:
            # Step 1: Move above object with correct orientation
            print("Step 1: Moving above object with aligned gripper...")
            if not self.move_to_position(x, y, z=self.safe_height, roll=180, pitch=0, yaw=gripper_yaw):
                return False

            # Step 2: Move down to pick height (maintaining orientation)
            print("Step 2: Moving down to pick height...")
            if not self.move_to_position(x, y, z=self.pick_height, roll=180, pitch=0, yaw=gripper_yaw):
                return False

            # Step 3: Close gripper
            print("Step 3: Closing gripper...")
            self._arm.close_lite6_gripper()
            time.sleep(1)

            # Step 4: Set TCP load for picked object
            print("Step 4: Updating TCP load...")
            self._arm.set_tcp_load(0.35, [0, 0, 40])

            # Step 5: Lift object (maintaining orientation)
            print("Step 5: Lifting object...")
            if not self.move_to_position(x, y, z=self.safe_height, roll=180, pitch=0, yaw=gripper_yaw):
                return False

            print("Status: Pick complete with auto-angle alignment!")
            print(f"{'='*60}\n")
            return True

        except Exception as e:
            print(f"[Robot Error] Pick sequence failed: {e}")
            print(f"{'='*60}\n")
            return False

    def place_sequence(self, x, y, maintain_angle=True, object_angle=0.0):
        """
        Execute a place sequence at the specified position.
        
        Args:
            x: X coordinate (mm)
            y: Y coordinate (mm)
            maintain_angle: Whether to maintain the gripper angle from pick
            object_angle: Object angle if maintain_angle is True
        """
        print(f"\n{'='*60}")
        print(f"EXECUTING PLACE SEQUENCE")
        print(f"{'='*60}")
        print(f"Position: X={x:.1f} mm, Y={y:.1f} mm")
        
        # Calculate yaw if we want to maintain the angle
        gripper_yaw = self.calculate_gripper_yaw(object_angle) if maintain_angle else 0.0
        if maintain_angle:
            print(f"Maintaining gripper angle: {gripper_yaw:.1f}°")

        try:
            # Step 1: Move above target
            print("Step 1: Moving above target position...")
            if not self.move_to_position(x, y, z=self.safe_height, roll=180, pitch=0, yaw=gripper_yaw):
                return False

            # Step 2: Move down to place height
            print("Step 2: Moving down to place height...")
            if not self.move_to_position(x, y, z=self.pick_height, roll=180, pitch=0, yaw=gripper_yaw):
                return False

            # Step 3: Open gripper
            print("Step 3: Opening gripper...")
            self._arm.open_lite6_gripper()
            time.sleep(1)

            # Step 4: Reset TCP load
            print("Step 4: Resetting TCP load...")
            self._arm.set_tcp_load(0.277, [0, 0, 30])

            # Step 5: Retract
            print("Step 5: Retracting...")
            if not self.move_to_position(x, y, z=self.safe_height, roll=180, pitch=0, yaw=gripper_yaw):
                return False

            # Step 6: Stop gripper
            print("Step 6: Stopping gripper...")
            self._arm.stop_lite6_gripper()

            print("Status: Place complete")
            print(f"{'='*60}\n")
            return True

        except Exception as e:
            print(f"[Robot Error] Place sequence failed: {e}")
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
    """Main application that monitors clicks and controls the xArm with auto-angle adjustment."""

    def __init__(self):
        """Initialize controller with automatic button-based mode selection."""
        self.click_manager = ClickDataManager()
        self.arm = XArmController()
        self.running = False
        self.last_processed_time = 0
        self.last_picked_angle = 0.0  # Store angle from last pick for place operation

        # Workspace limits
        click_config = self.arm.config.get("click_control", {})
        self.workspace_min_x = click_config.get("workspace_min_x", 0)
        self.workspace_max_x = click_config.get("workspace_max_x", 300)
        self.workspace_min_y = click_config.get("workspace_min_y", 0)
        self.workspace_max_y = click_config.get("workspace_max_y", 300)

        print("\n" + "="*60)
        print("xArm CLICK CONTROLLER WITH AUTO-ANGLE ADJUSTMENT")
        print("="*60)
        print("Mouse Button Controls:")
        print("  • LEFT CLICK   → Move to position")
        print("  • MIDDLE CLICK → Pick sequence (auto-adjusts gripper angle)")
        print("  • RIGHT CLICK  → Place sequence (maintains gripper angle)")
        print(f"\nWorkspace: X=[{self.workspace_min_x}-{self.workspace_max_x}], "
              f"Y=[{self.workspace_min_y}-{self.workspace_max_y}]")
        print(f"Safe height: {self.arm.safe_height}mm, Pick height: {self.arm.pick_height}mm")
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
        angle = click_data.get("angle", 0.0)  # ← Read the angle from click data!
        timestamp = click_data.get("timestamp", 0)

        # Ignore old or duplicate clicks
        if timestamp <= self.last_processed_time:
            return

        print(f"\n[CLICK DETECTED] Position: ({x:.1f}, {y:.1f}) mm - Button: {button.upper()} - Angle: {angle:.1f}°")

        # Safety check
        if not self.is_position_safe(x, y):
            print(f"[WARNING] Position ({x:.1f}, {y:.1f}) is outside safe workspace!")
            print(f"[WARNING] Safe range: X=[{self.workspace_min_x}-{self.workspace_max_x}], "
                  f"Y=[{self.workspace_min_y}-{self.workspace_max_y}]")
            print("[WARNING] Ignoring click for safety.")
            self.click_manager.mark_processed()
            self.last_processed_time = timestamp
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
            success = self.arm.pick_sequence(x, y, object_angle=angle)
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
        self.last_processed_time = timestamp

    def monitor_clicks(self):
        """Monitor shared memory for new clicks."""
        print("\n" + "="*60)
        print("MONITORING FOR CLICKS")
        print("="*60)
        print("Waiting for mouse clicks from detection system...")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")

        self.running = True
        check_count = 0

        try:
            while self.running:
                # Read click data
                click_data = self.click_manager.read_click()

                # Debug: Show we're checking (every 100 checks = ~5 seconds)
                check_count += 1
                # if check_count % 100 == 0:
                #     print(f"[DEBUG] Still monitoring... (checked {check_count} times)")

                # Check if there's a new unprocessed click
                if not click_data.get("processed", True):
                    print(f"[DEBUG] Found unprocessed click: {click_data}")
                    self.process_click(click_data)

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
        print("[INFO] Controller stopped.\n")


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("xArm ROBOT ARM CLICK CONTROLLER - AUTO GRIPPER ANGLE")
    print("="*60)
    print("\nAutomatic mode selection based on mouse buttons:")
    print("  • LEFT CLICK   = Move to position")
    print("  • MIDDLE CLICK = Pick sequence (auto-adjusts gripper angle)")
    print("  • RIGHT CLICK  = Place sequence (maintains gripper angle)")
    print("\nThe gripper will automatically align with the detected object")
    print("orientation during pick operations!")
    print("\nRobot will go to home position on startup.")
    print("Make sure the detection system is running!")
    print("="*60 + "\n")

    time.sleep(2)

    try:
        controller = XArmClickController()
        controller.monitor_clicks()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
