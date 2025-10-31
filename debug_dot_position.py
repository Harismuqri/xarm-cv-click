"""
Debug Tool - Check Dot Position
Helps diagnose why the red dot might not be visible
"""

import json
import os
from xarm.wrapper import XArmAPI
import cv2
import numpy as np
import pickle
import PySpin
import time

class DotPositionDebug:
    """Debug tool to check dot position calculation."""
    
    def __init__(self):
        self.config = self.load_config()
        self.robot_ip = self.config.get("robot_ip", "192.168.1.151")
        self._arm = None
        
        click_config = self.config.get("click_control", {})
        self.offset_x = click_config.get("coordinate_offset_x", 0)
        self.offset_y = click_config.get("coordinate_offset_y", -150)
        
        self.camera = None
        self.camera_system = None
        self.H = None
        self.H_inv = None
    
    def load_config(self, path="config.json"):
        """Load configuration."""
        try:
            if not os.path.isabs(path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(base_dir, path)
            with open(path, "r") as f:
                return json.load(f)
        except:
            return {"robot_ip": "192.168.1.151", "click_control": {"coordinate_offset_x": 0, "coordinate_offset_y": -150}}
    
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
            print("[Robot] ✅ Connected\n")
            return True
        except Exception as e:
            print(f"[Robot] ❌ Failed: {e}")
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
                    print(f"[Camera] ✅ Loaded {filename}\n")
                    return True
            
            print("[Camera] ❌ No homography file found")
            return False
            
        except Exception as e:
            print(f"[Camera] ❌ Failed: {e}")
            return False
    
    def get_robot_position(self):
        """Get robot position."""
        try:
            code, pos = self._arm.get_position()
            if code == 0:
                return pos[0], pos[1], pos[2]
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
    
    def debug_dot_position(self):
        """Debug where the dot should appear."""
        print("="*70)
        print("DOT POSITION DEBUG")
        print("="*70)
        
        # Get robot position
        robot_x, robot_y, robot_z = self.get_robot_position()
        if robot_x is None:
            print("❌ Cannot get robot position")
            return
        
        print(f"\n1. RAW ROBOT POSITION:")
        print(f"   Robot X = {robot_x:.1f} mm")
        print(f"   Robot Y = {robot_y:.1f} mm")
        print(f"   Robot Z = {robot_z:.1f} mm")
        
        print(f"\n2. CONFIGURED OFFSETS:")
        print(f"   offset_x = {self.offset_x}")
        print(f"   offset_y = {self.offset_y}")
        
        # Try different transformation methods
        print(f"\n3. DIFFERENT COORDINATE TRANSFORMATIONS:")
        
        # Method 1: Simple offset (old way)
        simple_x = robot_x - self.offset_x
        simple_y = robot_y - self.offset_y
        print(f"\n   Method 1 (Simple offset):")
        print(f"   camera_x = {robot_x} - {self.offset_x} = {simple_x:.1f}")
        print(f"   camera_y = {robot_y} - {self.offset_y} = {simple_y:.1f}")
        
        # Method 2: With rotation (current way)
        rotated_x = -(robot_y - self.offset_y)
        rotated_y = -(robot_x - self.offset_x)
        print(f"\n   Method 2 (With rotation - current):")
        print(f"   camera_x = -({robot_y} - {self.offset_y}) = {rotated_x:.1f}")
        print(f"   camera_y = -({robot_x} - {self.offset_x}) = {rotated_y:.1f}")
        
        # Method 3: Without negatives
        no_neg_x = robot_y - self.offset_y
        no_neg_y = robot_x - self.offset_x
        print(f"\n   Method 3 (Rotation without negatives):")
        print(f"   camera_x = {robot_y} - {self.offset_y} = {no_neg_x:.1f}")
        print(f"   camera_y = {robot_x} - {self.offset_x} = {no_neg_y:.1f}")
        
        # Get camera frame
        frame = self.get_camera_frame()
        if frame is None:
            print("\n❌ Cannot get camera frame")
            return
        
        frame_height, frame_width = frame.shape[:2]
        print(f"\n4. CAMERA FRAME SIZE:")
        print(f"   Width: {frame_width} pixels")
        print(f"   Height: {frame_height} pixels")
        
        # Transform to pixel coordinates for each method
        print(f"\n5. PIXEL COORDINATES (using homography):")
        
        methods = [
            ("Simple offset", simple_x, simple_y, (0, 255, 0)),
            ("With rotation", rotated_x, rotated_y, (0, 0, 255)),
            ("No negatives", no_neg_x, no_neg_y, (255, 0, 255))
        ]
        
        display_frame = frame.copy()
        
        for method_name, cam_x, cam_y, color in methods:
            pos_real = np.array([[cam_x, cam_y]], dtype=np.float32).reshape(-1, 1, 2)
            pos_img = cv2.perspectiveTransform(pos_real, self.H_inv).reshape(-1, 2).astype(int)
            pixel_x, pixel_y = pos_img[0]
            
            print(f"\n   {method_name}:")
            print(f"   Camera coords: ({cam_x:.1f}, {cam_y:.1f})")
            print(f"   Pixel coords: ({pixel_x}, {pixel_y})")
            
            # Check if in frame
            in_frame = (0 <= pixel_x < frame_width and 0 <= pixel_y < frame_height)
            status = "✅ VISIBLE" if in_frame else "❌ OFF-SCREEN"
            print(f"   Status: {status}")
            
            # Draw on frame
            if in_frame:
                cv2.circle(display_frame, (pixel_x, pixel_y), 15, color, -1)
                cv2.circle(display_frame, (pixel_x, pixel_y), 18, (255, 255, 255), 2)
                cv2.putText(display_frame, method_name[:6], (pixel_x + 20, pixel_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Draw arrow pointing to off-screen location
                center_x, center_y = frame_width // 2, frame_height // 2
                # Clamp to edge
                arrow_x = max(50, min(frame_width - 50, pixel_x))
                arrow_y = max(50, min(frame_height - 50, pixel_y))
                cv2.arrowedLine(display_frame, (center_x, center_y), (arrow_x, arrow_y),
                               color, 3, tipLength=0.3)
                cv2.putText(display_frame, f"{method_name} OFF-SCREEN", (center_x + 10, center_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw workspace boundary
        box_real = np.array([
            [0, 0], [300, 0], [300, 300], [0, 300]
        ], dtype=np.float32).reshape(-1, 1, 2)
        box_img = cv2.perspectiveTransform(box_real, self.H_inv).reshape(-1, 2).astype(int)
        cv2.polylines(display_frame, [box_img], isClosed=True, color=(255, 255, 255), thickness=2)
        
        # Add legend
        cv2.putText(display_frame, "Green = Simple offset", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, "Red = With rotation (current)", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(display_frame, "Magenta = No negatives", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(display_frame, "White box = Workspace (0-300mm)", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        print(f"\n6. DISPLAYING RESULTS:")
        print("   Camera window shows all three methods with different colors")
        print("   - Green = Simple offset")
        print("   - Red = With rotation (current code)")
        print("   - Magenta = Without negatives")
        print("   Press any key to close...")
        
        cv2.imshow("Dot Position Debug", display_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print("RECOMMENDATION:")
        print("="*70)
        print("\nWhich colored dot is at the actual robot position?")
        print("  - If GREEN: Use simple offset (no rotation)")
        print("  - If RED: Current code is correct")
        print("  - If MAGENTA: Remove negative signs")
        print("  - If NONE: Offsets or homography need adjustment")
        print("="*70)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.camera:
            try:
                self.camera.EndAcquisition()
                self.camera.DeInit()
            except:
                pass
        if self.camera_system:
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
    print("DOT POSITION DEBUG TOOL")
    print("="*70)
    print("\nThis tool shows where the dot appears with different coordinate")
    print("transformations to help identify the correct one.")
    print("="*70 + "\n")
    
    debug = DotPositionDebug()
    
    try:
        if not debug.connect_robot():
            return
        
        if not debug.initialize_camera():
            return
        
        debug.debug_dot_position()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        debug.cleanup()


if __name__ == "__main__":
    main()
