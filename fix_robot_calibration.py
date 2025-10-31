"""
Robot Calibration Diagnostic Tool
Helps identify and fix coordinate transformation issues
"""

import json
import os
from xarm.wrapper import XArmAPI
import cv2
import numpy as np
import pickle
import PySpin
import time

class CalibrationDiagnostic:
    """Diagnose and fix coordinate transformation issues."""
    
    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        self.robot_ip = self.config.get("robot_ip", "192.168.1.151")
        self._arm = None
        
        # Current offsets
        click_config = self.config.get("click_control", {})
        self.offset_x = click_config.get("coordinate_offset_x", 0)
        self.offset_y = click_config.get("coordinate_offset_y", -150)
        
        # Camera setup
        self.camera = None
        self.camera_system = None
        self.H = None
        self.H_inv = None
        
        print("\n" + "="*70)
        print("ROBOT CALIBRATION DIAGNOSTIC TOOL")
        print("="*70)
        print(f"Current offsets: X={self.offset_x}, Y={self.offset_y}")
        print("="*70 + "\n")
    
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
            return {}
    
    def connect_robot(self):
        """Connect to robot."""
        try:
            print(f"[Robot] Connecting to {self.robot_ip}...")
            self._arm = XArmAPI(self.robot_ip, baud_checkset=False)
            self._arm.clean_warn()
            self._arm.clean_error()
            self._arm.motion_enable(True)
            self._arm.set_mode(0)
            self._arm.set_state(0)
            time.sleep(0.5)
            print("[Robot] ✅ Connected")
            return True
        except Exception as e:
            print(f"[Robot] ❌ Connection failed: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize camera."""
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
            
            print("[Camera] ❌ No homography file found")
            return False
            
        except Exception as e:
            print(f"[Camera] ❌ Failed: {e}")
            return False
    
    def get_robot_position(self):
        """Get current robot position."""
        try:
            code, pos = self._arm.get_position()
            if code == 0:
                return pos[0], pos[1], pos[2]  # X, Y, Z
        except:
            pass
        return None, None, None
    
    def convert_pyspin_to_cv2(self, image):
        """Convert PySpin to OpenCV."""
        if not image.IsValid():
            return None
        img_array = image.GetNDArray()
        if len(img_array.shape) == 2:
            return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif len(img_array.shape) == 3:
            return img_array
        return None
    
    def get_camera_frame(self):
        """Get camera frame."""
        try:
            image = self.camera.GetNextImage()
            if image.IsIncomplete():
                image.Release()
                return None
            frame = self.convert_pyspin_to_cv2(image)
            image.Release()
            return frame
        except:
            return None
    
    def test_coordinate_transform(self):
        """Test if coordinate transformation is correct."""
        print("\n" + "="*70)
        print("COORDINATE TRANSFORMATION TEST")
        print("="*70)
        
        # Get robot position
        robot_x, robot_y, robot_z = self.get_robot_position()
        if robot_x is None:
            print("❌ Cannot get robot position")
            return False
        
        print(f"\n1. Raw Robot Position:")
        print(f"   X = {robot_x:.1f} mm")
        print(f"   Y = {robot_y:.1f} mm")
        print(f"   Z = {robot_z:.1f} mm")
        
        # Apply current offsets
        camera_x = robot_x - self.offset_x
        camera_y = robot_y - self.offset_y
        
        print(f"\n2. After Applying Offsets (offset_x={self.offset_x}, offset_y={self.offset_y}):")
        print(f"   Camera X = {robot_x} - {self.offset_x} = {camera_x:.1f} mm")
        print(f"   Camera Y = {robot_y} - {self.offset_y} = {camera_y:.1f} mm")
        
        # Transform to pixel coordinates
        if self.H_inv is not None:
            pos_real = np.array([[camera_x, camera_y]], dtype=np.float32).reshape(-1, 1, 2)
            pos_img = cv2.perspectiveTransform(pos_real, self.H_inv).reshape(-1, 2)
            pixel_x, pixel_y = pos_img[0]
            
            print(f"\n3. Transformed to Pixel Coordinates:")
            print(f"   Pixel X = {pixel_x:.1f}")
            print(f"   Pixel Y = {pixel_y:.1f}")
        
        print("\n" + "="*70)
        return True
    
    def interactive_offset_calibration(self):
        """Interactive tool to find correct offsets."""
        print("\n" + "="*70)
        print("INTERACTIVE OFFSET CALIBRATION")
        print("="*70)
        print("\nThis tool helps you find the correct coordinate offsets.")
        print("\nInstructions:")
        print("  1. Camera window will show current view")
        print("  2. A RED DOT shows where system THINKS robot is")
        print("  3. A YELLOW CROSSHAIR shows the workspace center (reference)")
        print("  4. Click on where robot ACTUALLY is in the image")
        print("  5. System calculates correct offsets")
        print("\nPress ENTER to start...")
        input()
        
        # Get robot position
        robot_x, robot_y, robot_z = self.get_robot_position()
        if robot_x is None:
            print("❌ Cannot get robot position")
            return
        
        print(f"\n✅ Robot is at: ({robot_x:.1f}, {robot_y:.1f}, {robot_z:.1f})")
        
        # Get camera frame
        frame = self.get_camera_frame()
        if frame is None:
            print("❌ Cannot get camera frame")
            return
        
        frame_height, frame_width = frame.shape[:2]
        
        # Current calculated position (wrong)
        camera_x_old = robot_x - self.offset_x
        camera_y_old = robot_y - self.offset_y
        pos_real_old = np.array([[camera_x_old, camera_y_old]], dtype=np.float32).reshape(-1, 1, 2)
        pos_img_old = cv2.perspectiveTransform(pos_real_old, self.H_inv).reshape(-1, 2).astype(int)
        
        # Draw on frame
        display_frame = frame.copy()
        
        # Draw workspace boundary (white box)
        box_real = np.array([
            [0, 0], [300, 0], [300, 300], [0, 300]
        ], dtype=np.float32).reshape(-1, 1, 2)
        box_img = cv2.perspectiveTransform(box_real, self.H_inv).reshape(-1, 2).astype(int)
        cv2.polylines(display_frame, [box_img], isClosed=True, color=(255, 255, 255), thickness=2)
        
        # Draw workspace center (150, 150) - SMALL DOT REFERENCE
        center_real = np.array([[150, 150]], dtype=np.float32).reshape(-1, 1, 2)
        center_img = cv2.perspectiveTransform(center_real, self.H_inv).reshape(-1, 2).astype(int)
        center_pt = tuple(center_img[0])
        
        # Small yellow dot at center
        cv2.circle(display_frame, center_pt, 4, (0, 255, 255), -1)  # Small filled circle
        cv2.circle(display_frame, center_pt, 6, (255, 255, 255), 1)  # Thin white border
        
        # Small label
        cv2.putText(display_frame, "CENTER", 
                   (center_pt[0] + 10, center_pt[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Draw grid lines for reference
        # Vertical lines at 75, 150, 225
        for x_val in [75, 150, 225]:
            p1_real = np.array([[x_val, 0]], dtype=np.float32).reshape(-1, 1, 2)
            p2_real = np.array([[x_val, 300]], dtype=np.float32).reshape(-1, 1, 2)
            p1_img = cv2.perspectiveTransform(p1_real, self.H_inv).reshape(-1, 2).astype(int)
            p2_img = cv2.perspectiveTransform(p2_real, self.H_inv).reshape(-1, 2).astype(int)
            line_color = (0, 255, 255) if x_val == 150 else (100, 100, 100)
            cv2.line(display_frame, tuple(p1_img[0]), tuple(p2_img[0]), line_color, 1)
        
        # Horizontal lines at 75, 150, 225
        for y_val in [75, 150, 225]:
            p1_real = np.array([[0, y_val]], dtype=np.float32).reshape(-1, 1, 2)
            p2_real = np.array([[300, y_val]], dtype=np.float32).reshape(-1, 1, 2)
            p1_img = cv2.perspectiveTransform(p1_real, self.H_inv).reshape(-1, 2).astype(int)
            p2_img = cv2.perspectiveTransform(p2_real, self.H_inv).reshape(-1, 2).astype(int)
            line_color = (0, 255, 255) if y_val == 150 else (100, 100, 100)
            cv2.line(display_frame, tuple(p1_img[0]), tuple(p2_img[0]), line_color, 1)
        
        # Draw current (wrong) robot position
        pixel_x_old, pixel_y_old = pos_img_old[0]
        if 0 <= pixel_x_old < frame_width and 0 <= pixel_y_old < frame_height:
            cv2.circle(display_frame, tuple(pos_img_old[0]), 15, (0, 0, 255), -1)
            cv2.circle(display_frame, tuple(pos_img_old[0]), 18, (255, 255, 255), 2)
            cv2.putText(display_frame, "WRONG POSITION?", 
                       (pos_img_old[0][0] + 20, pos_img_old[0][1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Draw arrow if off-screen
            center_x, center_y = frame_width // 2, frame_height // 2
            arrow_x = max(50, min(frame_width - 50, pixel_x_old))
            arrow_y = max(50, min(frame_height - 50, pixel_y_old))
            cv2.arrowedLine(display_frame, (center_x, center_y), (arrow_x, arrow_y),
                           (0, 0, 255), 3, tipLength=0.2)
            cv2.putText(display_frame, "CALCULATED POSITION OFF-SCREEN", 
                       (center_x - 150, center_y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add instructions
        cv2.putText(display_frame, "CLICK on the ACTUAL ROBOT POSITION, then press any key",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Small yellow dot = Workspace center reference",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Get user click
        clicked_point = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_point.clear()
                clicked_point.append((x, y))
                
                # Redraw frame with click
                temp_frame = display_frame.copy()
                cv2.circle(temp_frame, (x, y), 15, (0, 255, 0), -1)
                cv2.circle(temp_frame, (x, y), 18, (255, 255, 255), 2)
                cv2.drawMarker(temp_frame, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
                cv2.putText(temp_frame, "ACTUAL POSITION", 
                           (x + 20, y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(temp_frame, "Press any key to confirm",
                           (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Offset Calibration", temp_frame)
        
        cv2.namedWindow("Offset Calibration")
        cv2.setMouseCallback("Offset Calibration", mouse_callback)
        cv2.imshow("Offset Calibration", display_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if not clicked_point:
            print("❌ No click detected")
            return
        
        # Calculate correct offsets
        click_x, click_y = clicked_point[0]
        print(f"\n✅ You clicked at pixel: ({click_x}, {click_y})")
        
        # Transform click to real coordinates
        click_pt = np.array([[click_x, click_y]], dtype=np.float32).reshape(-1, 1, 2)
        real_pt = cv2.perspectiveTransform(click_pt, self.H).reshape(-1, 2)
        camera_x_correct = real_pt[0][0]
        camera_y_correct = real_pt[0][1]
        
        print(f"✅ This corresponds to camera coords: ({camera_x_correct:.1f}, {camera_y_correct:.1f})")
        
        # Calculate new offsets
        # robot_x - offset_x = camera_x_correct
        # offset_x = robot_x - camera_x_correct
        new_offset_x = robot_x - camera_x_correct
        new_offset_y = robot_y - camera_y_correct
        
        print(f"\n" + "="*70)
        print("OFFSET CALCULATION RESULTS")
        print("="*70)
        print(f"\nCurrent offsets:")
        print(f"  offset_x = {self.offset_x}")
        print(f"  offset_y = {self.offset_y}")
        print(f"\nCalculated correct offsets:")
        print(f"  offset_x = {new_offset_x:.1f}")
        print(f"  offset_y = {new_offset_y:.1f}")
        print(f"\nChange needed:")
        print(f"  Δoffset_x = {new_offset_x - self.offset_x:.1f}")
        print(f"  Δoffset_y = {new_offset_y - self.offset_y:.1f}")
        print("="*70)
        
        # Ask to update config
        update = input("\nUpdate config.json with new offsets? (y/n): ").strip().lower()
        if update == 'y':
            self.update_config_offsets(new_offset_x, new_offset_y)
        
        return new_offset_x, new_offset_y
    
    def update_config_offsets(self, new_offset_x, new_offset_y):
        """Update config.json with new offsets."""
        try:
            config_path = "config.json"
            if not os.path.isabs(config_path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(base_dir, config_path)
            
            with open(config_path, "r") as f:
                config = json.load(f)
            
            config["click_control"]["coordinate_offset_x"] = round(new_offset_x, 1)
            config["click_control"]["coordinate_offset_y"] = round(new_offset_y, 1)
            
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            
            print(f"\n✅ Config updated successfully!")
            print(f"   New offset_x: {new_offset_x:.1f}")
            print(f"   New offset_y: {new_offset_y:.1f}")
            
        except Exception as e:
            print(f"\n❌ Failed to update config: {e}")
    
    def cleanup(self):
        """Cleanup resources."""
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


def main():
    """Main function."""
    print("\n" + "="*70)
    print("ROBOT CALIBRATION DIAGNOSTIC TOOL")
    print("="*70)
    print("\nThis tool helps fix coordinate transformation issues.")
    print("="*70 + "\n")
    
    diag = CalibrationDiagnostic()
    
    try:
        # Connect to robot
        if not diag.connect_robot():
            return
        
        # Initialize camera
        if not diag.initialize_camera():
            return
        
        while True:
            print("\n" + "="*70)
            print("MENU")
            print("="*70)
            print("1. Test current coordinate transformation")
            print("2. Interactive offset calibration (FIX THE PROBLEM)")
            print("3. Exit")
            print("="*70)
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                diag.test_coordinate_transform()
            
            elif choice == '2':
                diag.interactive_offset_calibration()
                print("\n✅ Calibration complete!")
                print("   Restart robot_keyboard_calibration.py to see the fix!")
            
            elif choice == '3':
                break
            
            else:
                print("Invalid option")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        diag.cleanup()
        print("\n[INFO] Diagnostic tool closed.\n")


if __name__ == "__main__":
    main()
