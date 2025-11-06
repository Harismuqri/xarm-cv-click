"""
YOLO Detection System with Dual Camera Support
Left Click = Move, Right Click = Pick/Place Toggle, T Key = Inspection
"""

import cv2
import numpy as np
import pickle
from ultralytics import YOLO
import PySpin
import time
from multiprocessing import shared_memory
import struct
import json
import os

# Load configuration
def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    try:
        if not os.path.isabs(config_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, config_path)
        
        with open(config_path, "r") as f:
            config = json.load(f)
            print(f"[Config] Loaded from {config_path}")
            return config
    except Exception as e:
        print(f"[Config] Failed to load config: {e}")
        print("[Config] Using default values")
        return {
            "yolo_model_path": "D:\\2. yolo\\train30\\weights\\best.pt",
            "detection_confidence": 0.8,
            "camera_config": {
                "detection_camera_index": 1,
                "inspection_camera_index": 0
            },
            "camera_offset": {
                "offset_x": 92.9,
                "offset_y": -1.35,
                "offset_error": 0.4
            },
            "window_config": {
                "detection_width": 960,
                "detection_height": 720,
                "detection_pos_x": 950,
                "detection_pos_y": 50,
                "inspection_width": 960,
                "inspection_height": 720,
                "inspection_pos_x": 0,
                "inspection_pos_y": 50,
                "calibration_width": 960,
                "calibration_height": 720,
                "calibration_pos_x": 480,
                "calibration_pos_y": 180
            },
            "workspace": {
                "width": 300,
                "height": 300
            }
        }

# Load config
CONFIG = load_config()

# Load your OBB YOLOv8 model
model_path = CONFIG.get("yolo_model_path", "D:\\2. yolo\\train30\\weights\\best.pt")
model = YOLO(model_path)
model.overrides['verbose'] = False
print(f"[YOLO] Model loaded from: {model_path}")

# Configuration
DEBUG_MODE = True
SHOW_CALIBRATION_TIME = 2000

# Camera offset configuration (mm)
camera_offset_config = CONFIG.get("camera_offset", {})
CAMERA_OFFSET_X = camera_offset_config.get("offset_x", 92.9)
CAMERA_OFFSET_Y = camera_offset_config.get("offset_y", -1.35)
CAMERA_OFFSET_ERROR = camera_offset_config.get("offset_error", 0.4)

print(f"[Config] Camera offset: ({CAMERA_OFFSET_X:.1f}, {CAMERA_OFFSET_Y:.1f}) ± {CAMERA_OFFSET_ERROR:.1f} mm")

# Camera indices
camera_config = CONFIG.get("camera_config", {})
DETECTION_CAMERA_INDEX = camera_config.get("detection_camera_index", 0)
INSPECTION_CAMERA_INDEX = camera_config.get("inspection_camera_index", 1)

print(f"[Config] Detection camera index: {DETECTION_CAMERA_INDEX}")
print(f"[Config] Inspection camera index: {INSPECTION_CAMERA_INDEX}")

# Detection confidence
DETECTION_CONFIDENCE = CONFIG.get("detection_confidence", 0.8)
print(f"[Config] Detection confidence: {DETECTION_CONFIDENCE}")

# Workspace configuration
workspace_config = CONFIG.get("workspace", {})
WORKSPACE_WIDTH = workspace_config.get("width", 300)
WORKSPACE_HEIGHT = workspace_config.get("height", 300)

print(f"[Config] Workspace: {WORKSPACE_WIDTH}x{WORKSPACE_HEIGHT} mm")

# Shared memory configuration
SHARED_MEMORY_NAME = "DetectionData"
SHARED_MEMORY_SIZE = 4096
CLICK_MEMORY_NAME = "ClickData"
CLICK_MEMORY_SIZE = 512
INSPECT_MEMORY_NAME = "InspectData"
INSPECT_MEMORY_SIZE = 512

# Global variables for mouse interaction
mouse_x, mouse_y = 0, 0
mouse_clicked = False
mouse_button = "left"  # "left", "middle", or "right"
selected_object = None
show_coordinates = False
inspection_mode = False  # Track if we're in inspection mode
last_click_x_mm, last_click_y_mm = 0.0, 0.0  # Store last clicked position in detection coordinates (mm)

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events on the detection window"""
    global mouse_x, mouse_y, mouse_clicked, show_coordinates, mouse_button

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y
        mouse_clicked = True
        mouse_button = "left"  # Move to position
        show_coordinates = True
    elif event == cv2.EVENT_MBUTTONDOWN:
        mouse_x, mouse_y = x, y
        mouse_clicked = True
        mouse_button = "middle"  # Unused (reserved for future features)
        show_coordinates = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        mouse_x, mouse_y = x, y
        mouse_clicked = True
        mouse_button = "right"  # Pick/Place toggle
        show_coordinates = True

def convert_pyspin_image_to_cv2(image):
    if not image.IsValid():
        return None
    img_array = image.GetNDArray()
    if len(img_array.shape) == 2:
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif len(img_array.shape) == 3:
        return img_array
    return None

def Aut_transform_points(points, H):
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H)
    return warped.reshape(-1, 2)

def Is_inside_box(pts, width=None, height=None):
    if width is None:
        width = WORKSPACE_WIDTH
    if height is None:
        height = WORKSPACE_HEIGHT
    x, y = pts[:, 0], pts[:, 1]
    return np.all((x >= 0) & (x <= width) & (y >= 0) & (y <= height))

def Sget_angle(obb_pts):
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

def Ydraw_transformed_box(frame, H_inv, width_mm=None, height_mm=None):
    if width_mm is None:
        width_mm = WORKSPACE_WIDTH
    if height_mm is None:
        height_mm = WORKSPACE_HEIGHT
    box_real = np.array([
        [0, 0],
        [width_mm, 0],
        [width_mm, height_mm],
        [0, height_mm]
    ], dtype=np.float32).reshape(-1, 1, 2)
    box_img = cv2.perspectiveTransform(box_real, H_inv).reshape(-1, 2).astype(int)
    cv2.polylines(frame, [box_img], isClosed=True, color=(255, 255, 255), thickness=3)

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon"""
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def draw_info_panel(frame, x, y, info_lines, title="Info"):
    """Draw an information panel at specified position"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    padding = 10
    line_height = 20

    max_width = 0
    for line in info_lines:
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, w)

    panel_width = max_width + 2 * padding
    panel_height = len(info_lines) * line_height + 2 * padding + 25

    if x + panel_width > frame.shape[1]:
        x = frame.shape[1] - panel_width - 10
    if y + panel_height > frame.shape[0]:
        y = frame.shape[0] - panel_height - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), (0, 255, 255), 2)

    cv2.rectangle(frame, (x, y), (x + panel_width, y + 25), (0, 255, 255), -1)
    cv2.putText(frame, title, (x + padding, y + 18), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    y_offset = y + 40
    for line in info_lines:
        cv2.putText(frame, line, (x + padding, y_offset), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y_offset += line_height

    return frame

def draw_inspection_crosshair(frame, x, y, offset_error=CAMERA_OFFSET_ERROR):
    """Draw inspection crosshair with error tolerance circle"""
    # Red crosshair
    line_length = 40
    thickness = 3
    color = (0, 0, 255)
    
    cv2.line(frame, (x - line_length, y), (x + line_length, y), color, thickness)
    cv2.line(frame, (x, y - line_length), (x, y + line_length), color, thickness)
    
    # Green center circle
    cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
    
    # Error tolerance circle (yellow, dashed appearance)
    error_radius_px = 15
    for angle in range(0, 360, 30):
        angle_rad = np.radians(angle)
        x1 = int(x + error_radius_px * np.cos(angle_rad))
        y1 = int(y + error_radius_px * np.sin(angle_rad))
        x2 = int(x + error_radius_px * np.cos(angle_rad + np.radians(15)))
        y2 = int(y + error_radius_px * np.sin(angle_rad + np.radians(15)))
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

class SharedMemoryManager:
    """Manages shared memory for detection data."""

    def __init__(self, name=SHARED_MEMORY_NAME, size=SHARED_MEMORY_SIZE):
        self.name = name
        self.size = size
        self.shm = None
        self._initialize_shared_memory()

    def _initialize_shared_memory(self):
        """Create or attach to existing shared memory."""
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            print(f"[INFO] Created new shared memory: {self.name}")
            self._write_data({"status": "Not Ready", "timestamp": time.time(), "objects": {}})
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
            print(f"[INFO] Attached to existing shared memory: {self.name}")

    def _write_data(self, data):
        """Write data to shared memory as JSON."""
        try:
            json_str = json.dumps(data)
            json_bytes = json_str.encode('utf-8')

            if len(json_bytes) > self.size - 4:
                print(f"[WARNING] Data too large for shared memory ({len(json_bytes)} > {self.size-4})")
                return False

            self.shm.buf[:4] = struct.pack('I', len(json_bytes))
            self.shm.buf[4:4+len(json_bytes)] = json_bytes
            return True
        except Exception as e:
            print(f"[ERROR] Failed to write to shared memory: {e}")
            return False

    def update_status(self, status):
        """Update only the status field."""
        data = self._read_data()
        data["status"] = status
        data["timestamp"] = time.time()
        self._write_data(data)

    def update_object(self, object_id, x, y, angle, width, height):
        """Update a single object's data."""
        data = self._read_data()
        if "objects" not in data:
            data["objects"] = {}
        data["objects"][str(object_id)] = {
            "x": float(round(x, 1)),
            "y": float(round(y, 1)),
            "angle": float(round(angle, 1)),
            "width": float(round(width, 1)),
            "height": float(round(height, 1))
        }
        data["timestamp"] = time.time()
        self._write_data(data)

    def clear_object(self, object_id):
        """Remove an object from shared memory."""
        data = self._read_data()
        if str(object_id) in data["objects"]:
            del data["objects"][str(object_id)]
            data["timestamp"] = time.time()
            self._write_data(data)

    def clear_all_objects(self):
        """Clear all objects."""
        data = self._read_data()
        data["objects"] = {}
        data["timestamp"] = time.time()
        self._write_data(data)

    def _read_data(self):
        """Read data from shared memory."""
        try:
            length = struct.unpack('I', bytes(self.shm.buf[:4]))[0]
            if length == 0 or length > self.size - 4:
                return {"status": "Not Ready", "timestamp": time.time(), "objects": {}}

            json_bytes = bytes(self.shm.buf[4:4+length])
            json_str = json_bytes.decode('utf-8')
            data = json.loads(json_str)

            if "objects" not in data:
                data["objects"] = {}

            return data
        except Exception as e:
            print(f"[ERROR] Failed to read from shared memory: {e}")
            return {"status": "Not Ready", "timestamp": time.time(), "objects": {}}

    def get_data(self):
        """Public method to read current data."""
        return self._read_data()

    def cleanup(self):
        """Close and unlink shared memory."""
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
                print(f"[INFO] Shared memory cleaned up: {self.name}")
            except Exception as e:
                print(f"[ERROR] Failed to cleanup shared memory: {e}")


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
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
            print(f"[INFO] Attached to existing click data shared memory: {self.name}")
            existing_data = self._read_data()
            print(f"[DEBUG] Found existing click data: {existing_data}")
        except FileNotFoundError:
            try:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
                print(f"[INFO] Created click data shared memory: {self.name}")
                self._write_data({"click_x": 0, "click_y": 0, "timestamp": 0, "processed": True, "button": "none", "angle": 0.0})
            except Exception as e:
                print(f"[ERROR] Failed to create shared memory: {e}")
                raise

    def _write_data(self, data):
        """Write data to shared memory as JSON."""
        print(f"[DEBUG-TRACE] _write_data called with: {data}")
        try:
            json_str = json.dumps(data)
            json_bytes = json_str.encode('utf-8')

            if len(json_bytes) > self.size - 4:
                print(f"[ERROR] Data too large: {len(json_bytes)} > {self.size-4}")
                return False

            self.shm.buf[:4] = struct.pack('I', len(json_bytes))
            self.shm.buf[4:4+len(json_bytes)] = json_bytes

            print(f"[DEBUG] Wrote {len(json_bytes)} bytes to shared memory '{self.name}'")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to write click data: {e}")
            import traceback
            traceback.print_exc()
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
            print(f"[ERROR] Failed to read click data: {e}")
            return {"click_x": 0, "click_y": 0, "timestamp": 0, "processed": True, "button": "none", "angle": 0.0, "width": 0.0, "height": 0.0}

    def write_click(self, x, y, button="left", angle=0.0, width=0.0, height=0.0):
        """Write a new click position with button type, object angle, and dimensions."""
        data = {
            "click_x": float(x),
            "click_y": float(y),
            "button": button,
            "angle": float(angle),
            "width": float(width),
            "height": float(height),
            "timestamp": time.time(),
            "processed": False
        }
        self._write_data(data)
        button_action = {"left": "MOVE", "middle": "UNUSED", "right": "PICK/PLACE"}
        print(f"[CLICK-{button_action.get(button, button).upper()}] Sent to robot: ({x:.1f}, {y:.1f}) mm, Angle: {angle:.1f}°, Size: {width:.1f}x{height:.1f}mm")

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
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
            print(f"[INFO] Attached to existing inspect data shared memory: {self.name}")
        except FileNotFoundError:
            try:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
                print(f"[INFO] Created inspect data shared memory: {self.name}")
                self._write_data({"inspect": False, "target_x": 0, "target_y": 0, "angle": 0.0, "offset_x": CAMERA_OFFSET_X, "offset_y": CAMERA_OFFSET_Y, "timestamp": 0, "processed": True})
            except Exception as e:
                print(f"[ERROR] Failed to create inspect shared memory: {e}")
                raise

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


def auto_calibrate_homography(frame):
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

        print(f"[DEBUG] {len(filtered)} circles passed filtering")

        if DEBUG_MODE:
            debug_frame = frame.copy()
            for (x, y, r) in filtered:
                cv2.circle(debug_frame, (x, y), r, (0, 255, 0), 2)
                cv2.circle(debug_frame, (x, y), 2, (0, 0, 255), 3)

            window_config = CONFIG.get("window_config", {})
            cv2.namedWindow("Filtered Circles", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("Filtered Circles", 
                           window_config.get("calibration_width", 960), 
                           window_config.get("calibration_height", 720))
            cv2.moveWindow("Filtered Circles", 
                         window_config.get("calibration_pos_x", 480), 
                         window_config.get("calibration_pos_y", 180))
            cv2.imshow("Filtered Circles", debug_frame)
            cv2.waitKey(SHOW_CALIBRATION_TIME)
            cv2.destroyWindow("Filtered Circles")

        if len(filtered) >= 4:
            image_points = np.array([[x, y] for (x, y, _) in filtered[:4]], dtype=np.float32)
            image_points = sorted(image_points, key=lambda pt: (pt[1], pt[0]))
            top = sorted(image_points[:2], key=lambda pt: pt[0])
            bottom = sorted(image_points[2:], key=lambda pt: pt[0])
            sorted_img_pts = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)

            real_pts = np.array([
                [0, WORKSPACE_HEIGHT],
                [WORKSPACE_WIDTH, WORKSPACE_HEIGHT],
                [WORKSPACE_WIDTH, 0],
                [0, 0]
            ], dtype=np.float32)

            H, _ = cv2.findHomography(sorted_img_pts, real_pts)
            if H is not None:
                # Get script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))

                # Save homography_auto.pkl (camera → detection workspace)
                homography_auto_file = os.path.join(script_dir, "homography_auto.pkl")
                with open(homography_auto_file, "wb") as f:
                    pickle.dump(H, f)
                print(f"[INFO] Homography calibrated and saved as '{homography_auto_file}'")

                # Auto-create homography_det_to_robot.pkl (detection workspace → robot coordinates)
                # Detection workspace corners (0-300mm)
                detection_corners = np.array([
                    [0, 0],         # BL - Bottom Left
                    [WORKSPACE_WIDTH, 0],      # BR - Bottom Right
                    [WORKSPACE_WIDTH, WORKSPACE_HEIGHT],  # TR - Top Right
                    [0, WORKSPACE_HEIGHT]      # TL - Top Left
                ], dtype=np.float32)

                # Robot coordinates (actual measured positions in mm)
                robot_corners = np.array([
                    [88.9, 312],    # BL - Bottom Left
                    [88.9, 14.7],   # BR - Bottom Right
                    [382, 14.7],    # TR - Top Right
                    [382, 312]      # TL - Top Left
                ], dtype=np.float32)

                # Create second transformation matrix
                H_det_to_robot, _ = cv2.findHomography(detection_corners, robot_corners)
                homography_det_file = os.path.join(script_dir, "homography_det_to_robot.pkl")
                with open(homography_det_file, "wb") as f:
                    pickle.dump(H_det_to_robot, f)
                print(f"[INFO] Detection-to-robot transformation saved as '{homography_det_file}'")
                print("[INFO] Two-step calibration complete: Camera → Workspace → Robot")

                return H

    print("[WARNING] Not enough valid circles detected for calibration.")
    return None

def main():
    global mouse_clicked, mouse_x, mouse_y, selected_object, show_coordinates, inspection_mode

    shm_manager = SharedMemoryManager()
    click_manager = ClickDataManager()
    inspect_manager = InspectDataManager()

    last_saved_status = None
    last_saved_objects = {}

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()
    print(f"[INFO] Number of cameras detected: {num_cameras}")

    if num_cameras == 0:
        print("[ERROR] No FLIR cameras found.")
        system.ReleaseInstance()
        shm_manager.cleanup()
        return

    # Validate and adjust camera indices if needed
    if DETECTION_CAMERA_INDEX >= num_cameras:
        print(f"[WARNING] Detection camera index {DETECTION_CAMERA_INDEX} not available (only {num_cameras} camera(s) found)")
        print(f"[INFO] Using camera index 0 instead")
        actual_detection_index = 0
    else:
        actual_detection_index = DETECTION_CAMERA_INDEX

    # Initialize detection camera (configured index - top-down view)
    try:
        cam_detect = cam_list.GetByIndex(actual_detection_index)
        cam_detect.Init()
        serial_detect = cam_detect.TLDevice.DeviceSerialNumber.GetValue()
        print(f"[INFO] Detection camera initialized (Index {actual_detection_index}, Serial: {serial_detect})")
    except Exception as e:
        print(f"[ERROR] Failed to initialize detection camera: {e}")
        cam_list.Clear()
        system.ReleaseInstance()
        shm_manager.cleanup()
        return

    # Initialize inspection camera (configured index - gripper-mounted)
    cam_inspect = None
    if num_cameras >= 2:
        if INSPECTION_CAMERA_INDEX >= num_cameras or INSPECTION_CAMERA_INDEX == actual_detection_index:
            if INSPECTION_CAMERA_INDEX == actual_detection_index:
                print(f"[WARNING] Inspection camera index {INSPECTION_CAMERA_INDEX} conflicts with detection camera")
            else:
                print(f"[WARNING] Inspection camera index {INSPECTION_CAMERA_INDEX} not available")
            print(f"[INFO] Inspection mode will be disabled")
        else:
            try:
                cam_inspect = cam_list.GetByIndex(INSPECTION_CAMERA_INDEX)
                cam_inspect.Init()
                serial_inspect = cam_inspect.TLDevice.DeviceSerialNumber.GetValue()
                print(f"[INFO] Inspection camera initialized (Index {INSPECTION_CAMERA_INDEX}, Serial: {serial_inspect})")
            except Exception as e:
                print(f"[WARNING] Failed to initialize inspection camera: {e}")
                print(f"[INFO] Inspection mode will be disabled")
                cam_inspect = None
    else:
        print("[WARNING] Only 1 camera found. Inspection mode will be disabled.")

    # Start acquisition for detection camera
    cam_detect.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    cam_detect.BeginAcquisition()

    # Start acquisition for inspection camera if available
    if cam_inspect:
        cam_inspect.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        cam_inspect.BeginAcquisition()

    # Always perform auto-calibration (even if files exist, we'll overwrite them)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    homography_auto_file = os.path.join(script_dir, "homography_auto.pkl")
    homography_det_file = os.path.join(script_dir, "homography_det_to_robot.pkl")

    # Check if old calibration files exist (for information only)
    if os.path.exists(homography_auto_file):
        print(f"[INFO] Existing calibration found - will be overwritten with fresh calibration")

    print("[INFO] Starting auto-calibration...")
    print("[INFO] Please ensure 4 white calibration circles are visible in the camera view")

    # Get initial frame for calibration
    image = cam_detect.GetNextImage()
    frame = convert_pyspin_image_to_cv2(image)
    image.Release()

    # Run auto-calibration
    H = auto_calibrate_homography(frame)

    if H is None:
        print("\n" + "="*60)
        print("[ERROR] CALIBRATION FAILED")
        print("="*60)
        print("Calibration requires 4 white circles in the camera view.")
        print("\nTo fix this:")
        print("1. Place 4 white calibration circles (20-30mm diameter) at workspace corners")
        print("2. Ensure good lighting and contrast (white circles on dark background)")
        print("3. Use calibration-helper.py to see live camera view")
        print("="*60 + "\n")

        cam_detect.EndAcquisition()
        cam_detect.DeInit()
        if cam_inspect:
            cam_inspect.EndAcquisition()
            cam_inspect.DeInit()
        del cam_detect
        if cam_inspect:
            del cam_inspect
        cam_list.Clear()
        system.ReleaseInstance()
        shm_manager.cleanup()
        return

    H_inv = np.linalg.inv(H)

    last_outputs = {}
    last_change_time = {}

    # Set up windows
    window_detect = "Detection Camera (Click object then press T for inspection)"
    window_inspect = "Inspection Camera"
    
    window_config = CONFIG.get("window_config", {})
    
    cv2.namedWindow(window_detect, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_detect, 
                    window_config.get("detection_width", 960), 
                    window_config.get("detection_height", 720))
    cv2.moveWindow(window_detect, 
                  window_config.get("detection_pos_x", 950), 
                  window_config.get("detection_pos_y", 50))
    cv2.setMouseCallback(window_detect, mouse_callback)
    
    if cam_inspect:
        cv2.namedWindow(window_inspect, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_inspect, 
                        window_config.get("inspection_width", 960), 
                        window_config.get("inspection_height", 720))
        cv2.moveWindow(window_inspect, 
                      window_config.get("inspection_pos_x", 0), 
                      window_config.get("inspection_pos_y", 50))

    print("\n" + "="*60)
    print("DUAL CAMERA MODE - DETECTION & INSPECTION")
    print("="*60)
    print("Detection camera controls:")
    print("• Left click: MOVE to position")
    print("• Middle click: PICK at position (auto-adjusts gripper angle)")
    print("• Right click: PLACE at position")
    print("\nInspection mode:")
    print("• Click on object, then press 'T': Inspect selected object")
    print("• Press 'Q': Quit")
    print(f"\nCamera offset: ({CAMERA_OFFSET_X:.1f}, {CAMERA_OFFSET_Y:.1f}) ± {CAMERA_OFFSET_ERROR:.1f} mm")
    print("="*60 + "\n")

    try:
        while True:
            # Get detection camera frame
            image_detect = cam_detect.GetNextImage()
            if image_detect.IsIncomplete():
                image_detect.Release()
                continue

            frame_detect = convert_pyspin_image_to_cv2(image_detect)
            if frame_detect is None:
                print("[WARNING] Invalid detection frame")
                image_detect.Release()
                continue

            # Get inspection camera frame if available
            frame_inspect = None
            if cam_inspect:
                try:
                    image_inspect = cam_inspect.GetNextImage(100)  # 100ms timeout
                    if not image_inspect.IsIncomplete():
                        frame_inspect = convert_pyspin_image_to_cv2(image_inspect)
                        if frame_inspect is not None:
                            frame_inspect = frame_inspect.copy()
                    image_inspect.Release()
                except:
                    pass

            # Run YOLO detection
            results = model(frame_detect, conf=DETECTION_CONFIDENCE)
            obb_preds = results[0].obb

            annotated_frame = frame_detect.copy()
            current_time = time.time()
            detected_ids = set()

            # Store object data for mouse interaction
            object_data_list = []

            for i, obb in enumerate(obb_preds, 1):
                if hasattr(obb, "xyxyxyxy"):
                    corners = obb.xyxyxyxy.cpu().numpy().reshape(-1, 2)
                elif hasattr(obb, "xyxy"):
                    corners = obb.xyxy.cpu().numpy().reshape(-1, 2)
                else:
                    continue

                transformed = Aut_transform_points(corners, H)
                is_inside = Is_inside_box(transformed)

                if is_inside:
                    center = np.mean(transformed, axis=0)
                    angle = Sget_angle(transformed)

                    side1 = np.linalg.norm(transformed[1] - transformed[0])
                    side2 = np.linalg.norm(transformed[2] - transformed[1])

                    width_mm = round(max(side1, side2), 1)
                    height_mm = round(min(side1, side2), 1)

                    x_mm, y_mm = round(center[0], 1), round(center[1], 1)
                    angle_deg = round(angle, 2)

                    # Store object data for click detection
                    object_data_list.append({
                        'id': i,
                        'corners': corners.astype(int),
                        'x_mm': x_mm,
                        'y_mm': y_mm,
                        'angle': angle_deg,
                        'width': width_mm,
                        'height': height_mm
                    })

                    prev = last_outputs.get(i, (None, None, None, None, None))
                    if (
                        prev[0] is None or
                        abs(prev[0] - x_mm) >= 2.0 or
                        abs(prev[1] - y_mm) >= 2.0 or
                        abs(prev[2] - angle_deg) >= 2.0 or
                        abs(prev[3] - width_mm) >= 2.0 or
                        abs(prev[4] - height_mm) >= 2.0
                    ):
                        last_outputs[i] = (x_mm, y_mm, angle_deg, width_mm, height_mm)
                        last_change_time[i] = current_time
                        shm_manager.update_object(i, x_mm, y_mm, angle_deg, width_mm, height_mm)

                    detected_ids.add(i)
                    color = (0, 255, 0)

                else:
                    color = (0, 0, 255)

                corners_int = corners.astype(int)
                cv2.polylines(annotated_frame, [corners_int], isClosed=True, color=color, thickness=2)

                # Draw center point
                if is_inside:
                    center_img = np.mean(corners, axis=0).astype(int)
                    cv2.circle(annotated_frame, tuple(center_img), 5, (0, 0, 255), -1)
                    cv2.circle(annotated_frame, tuple(center_img), 5, (255, 255, 255), 1)
                    cv2.drawMarker(annotated_frame, tuple(center_img), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)

                label_pos = tuple(corners_int[0] - [0, 10])
                cv2.putText(
                    annotated_frame,
                    f"Object {i}",
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA
                )

            # Handle mouse click
            if mouse_clicked:
                mouse_clicked = False

                # Convert click position to real-world coordinates
                click_pt = np.array([[mouse_x, mouse_y]], dtype=np.float32).reshape(-1, 1, 2)
                real_coord = cv2.perspectiveTransform(click_pt, H).reshape(-1, 2)
                click_x_mm = real_coord[0][0]
                click_y_mm = real_coord[0][1]

                # Store the clicked position globally (for inspection)
                global last_click_x_mm, last_click_y_mm
                last_click_x_mm = click_x_mm
                last_click_y_mm = click_y_mm

                # Check if clicked on any object
                clicked_on_object = False
                for obj_data in object_data_list:
                    if point_in_polygon((mouse_x, mouse_y), obj_data['corners']):
                        selected_object = obj_data
                        show_coordinates = True
                        clicked_on_object = True
                        # Send object center position, ANGLE, and DIMENSIONS to robot
                        click_manager.write_click(
                            obj_data['x_mm'],
                            obj_data['y_mm'],
                            mouse_button,
                            obj_data['angle'],
                            obj_data['width'],
                            obj_data['height']
                        )
                        print(f"[CLICK] Object clicked at ({click_x_mm:.1f}, {click_y_mm:.1f}) mm - Center at ({obj_data['x_mm']:.1f}, {obj_data['y_mm']:.1f}) mm")
                        break

                # If clicked on empty space, send clicked position to robot
                if not clicked_on_object:
                    selected_object = None
                    show_coordinates = True
                    if 0 <= click_x_mm <= WORKSPACE_WIDTH and 0 <= click_y_mm <= WORKSPACE_HEIGHT:
                        click_manager.write_click(click_x_mm, click_y_mm, mouse_button, angle=0.0, width=0.0, height=0.0)
                    else:
                        print(f"[WARNING] Click outside workspace: ({click_x_mm:.1f}, {click_y_mm:.1f}) mm")

            # Draw info panel if showing coordinates
            if show_coordinates:
                if selected_object:
                    # Show object info
                    info_lines = [
                        f"Object ID: {selected_object['id']}",
                        f"Center: ({selected_object['x_mm']:.1f}, {selected_object['y_mm']:.1f}) mm",
                        f"Clicked: ({last_click_x_mm:.1f}, {last_click_y_mm:.1f}) mm",
                        f"Angle: {selected_object['angle']:.1f} degrees",
                        f"Width: {selected_object['width']:.1f} mm",
                        f"Height: {selected_object['height']:.1f} mm",
                        "Press 'T' to inspect clicked position"
                    ]

                    draw_info_panel(annotated_frame, mouse_x + 10, mouse_y + 10, info_lines, f"Object {selected_object['id']}")
                    cv2.polylines(annotated_frame, [selected_object['corners']], isClosed=True, color=(0, 255, 255), thickness=3)
                else:
                    # Show coordinate info at clicked position
                    click_pt = np.array([[mouse_x, mouse_y]], dtype=np.float32).reshape(-1, 1, 2)
                    real_coord = cv2.perspectiveTransform(click_pt, H).reshape(-1, 2)

                    info_lines = [
                        f"Pixel: ({mouse_x}, {mouse_y})",
                        f"Real: ({real_coord[0][0]:.1f}, {real_coord[0][1]:.1f}) mm"
                    ]

                    in_workspace = (0 <= real_coord[0][0] <= WORKSPACE_WIDTH and 0 <= real_coord[0][1] <= WORKSPACE_HEIGHT)
                    info_lines.append(f"In workspace: {'Yes' if in_workspace else 'No'}")

                    draw_info_panel(annotated_frame, mouse_x + 10, mouse_y + 10, info_lines, "Coordinates")
                    cv2.drawMarker(annotated_frame, (mouse_x, mouse_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

            # Status logic
            if len(detected_ids) <= 1:
                status_text = "Not Ready"
            elif len(detected_ids) >= 2 and all(current_time - last_change_time.get(obj_id, 0) >= 1.0 for obj_id in detected_ids):
                status_text = "Ready"
            else:
                status_text = "Detect"

            if status_text == "Ready":
                if status_text != last_saved_status:
                    shm_manager.update_status(status_text)
                    last_saved_status = status_text

                for i in detected_ids:
                    if i in last_outputs:
                        x_mm, y_mm, angle_deg, width_mm, height_mm = last_outputs[i]
                        prev_obj = last_saved_objects.get(i)
                        curr_obj = (x_mm, y_mm, angle_deg, width_mm, height_mm)
                        if prev_obj != curr_obj:
                            shm_manager.update_object(i, x_mm, y_mm, angle_deg, width_mm, height_mm)
                            last_saved_objects[i] = curr_obj

            elif status_text == "Not Ready":
                if status_text != last_saved_status:
                    shm_manager.update_status(status_text)
                    last_saved_status = status_text

                shm_manager.clear_all_objects()
                last_saved_objects.clear()

            else:
                if status_text != last_saved_status:
                    shm_manager.update_status(status_text)
                    last_saved_status = status_text

            # Status color
            if status_text == "Not Ready":
                status_color = (0, 100, 200)
            elif status_text == "Detect":
                status_color = (200, 150, 0)
            elif status_text == "Ready":
                status_color = (0, 150, 0)

            # Draw status
            cv2.putText(
                annotated_frame,
                f"Status: {status_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                status_color,
                2,
                cv2.LINE_AA
            )

            Ydraw_transformed_box(annotated_frame, H_inv)

            # Display detection frame
            cv2.imshow(window_detect, annotated_frame)

            # Display inspection frame if available
            if frame_inspect is not None and cam_inspect:
                h, w = frame_inspect.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                # Draw inspection crosshair
                draw_inspection_crosshair(frame_inspect, center_x, center_y)
                
                # Draw info overlay
                overlay = frame_inspect.copy()
                cv2.rectangle(overlay, (10, 10), (500, 140), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame_inspect, 0.4, 0, frame_inspect)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                info_lines = [
                    "INSPECTION CAMERA",
                    f"Offset: ({CAMERA_OFFSET_X:.1f}, {CAMERA_OFFSET_Y:.1f}) mm",
                    f"Error: {CAMERA_OFFSET_ERROR:.1f} mm"
                ]
                
                y_offset = 50
                for line in info_lines:
                    cv2.putText(frame_inspect, line, (25, y_offset), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
                    y_offset += 40
                
                cv2.imshow(window_inspect, frame_inspect)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('t') or key == ord('T'):
                # Trigger inspection mode - inspect at clicked position, not object center
                if selected_object and cam_inspect:
                    print(f"\n[INSPECT MODE] Inspecting Object {selected_object['id']}")
                    print(f"[INSPECT MODE] Clicked position: ({last_click_x_mm:.1f}, {last_click_y_mm:.1f}) mm")
                    print(f"[INSPECT MODE] Object center: ({selected_object['x_mm']:.1f}, {selected_object['y_mm']:.1f}) mm")
                    inspect_manager.write_inspect_command(
                        last_click_x_mm,  # Use clicked position, not object center
                        last_click_y_mm,  # Use clicked position, not object center
                        selected_object['angle']  # Use object angle for camera orientation
                    )
                elif not cam_inspect:
                    print("[WARNING] Inspection camera not available")
                else:
                    print("[WARNING] No object selected. Click on an object first.")

            image_detect.Release()

    finally:
        cam_detect.EndAcquisition()
        cam_detect.DeInit()
        if cam_inspect:
            cam_inspect.EndAcquisition()
            cam_inspect.DeInit()
        del cam_detect
        if cam_inspect:
            del cam_inspect
        cam_list.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()
        shm_manager.cleanup()
        click_manager.cleanup()
        inspect_manager.cleanup()

if __name__ == "__main__":
    main()
