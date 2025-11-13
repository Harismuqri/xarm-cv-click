import cv2
import numpy as np
import pickle
from ultralytics import YOLO
import PySpin
import time
from multiprocessing import shared_memory
import struct
import json

# Load your OBB YOLOv8 model
model = YOLO("D:\\2. yolo\\train30\\weights\\best.pt")
model.overrides['verbose'] = False

# Configuration
DEBUG_MODE = True
SHOW_CALIBRATION_TIME = 2000

# Shared memory configuration
SHARED_MEMORY_NAME = "DetectionData"
SHARED_MEMORY_SIZE = 4096
CLICK_MEMORY_NAME = "ClickData"
CLICK_MEMORY_SIZE = 512

# Global variables for mouse interaction
mouse_x, mouse_y = 0, 0
mouse_clicked = False
mouse_button = "left"  # "left", "middle", or "right"
selected_object = None
show_coordinates = False

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events on the detection window"""
    global mouse_x, mouse_y, mouse_clicked, show_coordinates, mouse_button

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y
        mouse_clicked = True
        mouse_button = "left"  # Move
        show_coordinates = True
    elif event == cv2.EVENT_MBUTTONDOWN:
        mouse_x, mouse_y = x, y
        mouse_clicked = True
        mouse_button = "middle"  # Pick
        show_coordinates = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        mouse_x, mouse_y = x, y
        mouse_clicked = True
        mouse_button = "right"  # Place
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

def Is_inside_box(pts, width=300, height=300):
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

def Ydraw_transformed_box(frame, H_inv, width_mm=300, height_mm=300):
    box_real = np.array([
        [0, 0],
        [width_mm, 0],
        [width_mm, height_mm],
        [0, height_mm]
    ], dtype=np.float32).reshape(-1, 1, 2)
    box_img = cv2.perspectiveTransform(box_real, H_inv).reshape(-1, 2).astype(int)
    cv2.polylines(frame, [box_img], isClosed=True, color=(0, 0, 0), thickness=3)  # Black, thicker for white background

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon"""
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def draw_info_panel(frame, x, y, info_lines, title="Info"):
    """Draw an information panel at specified position"""
    # Calculate panel size
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
    panel_height = len(info_lines) * line_height + 2 * padding + 25  # +25 for title

    # Adjust position if panel goes off screen
    if x + panel_width > frame.shape[1]:
        x = frame.shape[1] - panel_width - 10
    if y + panel_height > frame.shape[0]:
        y = frame.shape[0] - panel_height - 10

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw border
    cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), (0, 255, 255), 2)

    # Draw title
    cv2.rectangle(frame, (x, y), (x + panel_width, y + 25), (0, 255, 255), -1)
    cv2.putText(frame, title, (x + padding, y + 18), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # Draw info lines
    y_offset = y + 40
    for line in info_lines:
        cv2.putText(frame, line, (x + padding, y_offset), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y_offset += line_height

    return frame

class SharedMemoryManager:
    """
    Manages shared memory for detection data.

    Data structure (JSON format):
    {
        "status": "Ready" | "Detect" | "Not Ready",
        "timestamp": float,
        "objects": {
            "1": {"x": 150.5, "y": 200.3, "angle": 45.2, "width": 50.0, "height": 30.0},
            "2": {"x": 100.0, "y": 150.0, "angle": 90.0, "width": 40.0, "height": 25.0}
        }
    }
    """

    def __init__(self, name=SHARED_MEMORY_NAME, size=SHARED_MEMORY_SIZE):
        self.name = name
        self.size = size
        self.shm = None
        self._initialize_shared_memory()

    def _initialize_shared_memory(self):
        """Create or attach to existing shared memory."""
        try:
            # Try to create new shared memory
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            print(f"[INFO] Created new shared memory: {self.name}")
            # Initialize with empty data
            self._write_data({"status": "Not Ready", "timestamp": time.time(), "objects": {}})
        except FileExistsError:
            # Shared memory already exists, attach to it
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

            # Write length (4 bytes) + JSON data
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

            # Ensure objects key exists
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
        # Always try to attach first
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
            print(f"[INFO] Attached to existing click data shared memory: {self.name}")
            existing_data = self._read_data()
            print(f"[DEBUG] Found existing click data: {existing_data}")
        except FileNotFoundError:
            # Only create if doesn't exist
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

            # Debug: Verify write
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
            print(f"[ERROR] Failed to read click data: {e}")
            return {"click_x": 0, "click_y": 0, "timestamp": 0, "processed": True, "button": "none", "angle": 0.0, "width": 0.0, "height": 0.0}

    def write_click(self, x, y, button="left", angle=0.0, width=0.0, height=0.0):
        """Write a new click position with button type, object angle, and dimensions."""
        data = {
            "click_x": float(x),
            "click_y": float(y),
            "button": button,  # "left", "middle", or "right"
            "angle": float(angle),  # Object angle in degrees
            "width": float(width),  # Object width in mm
            "height": float(height),  # Object height in mm
            "timestamp": time.time(),
            "processed": False
        }
        self._write_data(data)
        button_action = {"left": "MOVE", "middle": "PICK", "right": "PLACE"}
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
                [0, 300],
                [300, 300],
                [300, 0],
                [0, 0]
            ], dtype=np.float32)

            H, _ = cv2.findHomography(sorted_img_pts, real_pts)
            if H is not None:
                with open("homography_auto.pkl", "wb") as f:
                    pickle.dump(H, f)
                print("[INFO] Homography calibrated and saved as 'homography_auto.pkl'")
                return H

    print("[WARNING] Not enough valid circles detected for calibration.")
    return None

def main():
    global mouse_clicked, mouse_x, mouse_y, selected_object, show_coordinates

    shm_manager = SharedMemoryManager()
    click_manager = ClickDataManager()

    last_saved_status = None
    last_saved_objects = {}

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("No FLIR cameras found.")
        system.ReleaseInstance()
        shm_manager.cleanup()
        return

    cam = cam_list.GetByIndex(0)
    cam.Init()
    cam.BeginAcquisition()

    image = cam.GetNextImage()
    frame = convert_pyspin_image_to_cv2(image)
    image.Release()

    H = auto_calibrate_homography(frame)
    if H is None:
        print("[ERROR] Failed to compute homography. Exiting.")
        cam.EndAcquisition()
        cam.DeInit()
        del cam
        cam_list.Clear()
        system.ReleaseInstance()
        shm_manager.cleanup()
        return

    H_inv = np.linalg.inv(H)

    last_outputs = {}
    last_change_time = {}

    # Set up mouse callback
    window_name = "YOLOv8 OBB - FLIR Camera (Click to see info)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "="*60)
    print("INTERACTIVE MODE ENABLED - WITH AUTO GRIPPER ANGLE")
    print("="*60)
    print("• Left click: MOVE to position")
    print("• Middle click: PICK at position (auto-adjusts gripper angle)")
    print("• Right click: PLACE at position")
    print("• Press 'q': Quit")
    print("="*60 + "\n")

    try:
        while True:
            image = cam.GetNextImage()
            if image.IsIncomplete():
                image.Release()
                continue

            frame = convert_pyspin_image_to_cv2(image)
            if frame is None:
                print("Invalid frame")
                image.Release()
                continue

            # control background 
            results = model(frame, conf=0.8)

            # OR Method 2: Add preprocessing (better)
            # frame_enhanced = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)
            # results = model(frame_enhanced, conf=0.7)

            obb_preds = results[0].obb

            annotated_frame = frame.copy()
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

                    # Store object data for click detection (NOW INCLUDING ANGLE!)
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
                        # Update object data
                        last_outputs[i] = (x_mm, y_mm, angle_deg, width_mm, height_mm)
                        last_change_time[i] = current_time
                        shm_manager.update_object(i, x_mm, y_mm, angle_deg, width_mm, height_mm)

                    detected_ids.add(i)
                    color = (0, 255, 0)  # Green border for detected objects

                else:
                    color = (0, 0, 255)  # Red for objects outside workspace

                corners_int = corners.astype(int)
                cv2.polylines(annotated_frame, [corners_int], isClosed=True, color=color, thickness=2)

                # Draw center point
                if is_inside:
                    # Calculate center in image coordinates
                    center_img = np.mean(corners, axis=0).astype(int)
                    # Draw center point (circle with cross) - RED for better visibility on white background
                    cv2.circle(annotated_frame, tuple(center_img), 5, (0, 0, 255), -1)  # Red filled circle
                    cv2.circle(annotated_frame, tuple(center_img), 5, (255, 255, 255), 1)  # White border
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

            # Handle mouse click - NOW WITH ANGLE!
            if mouse_clicked:
                mouse_clicked = False

                # Convert click position to real-world coordinates
                click_pt = np.array([[mouse_x, mouse_y]], dtype=np.float32).reshape(-1, 1, 2)
                real_coord = cv2.perspectiveTransform(click_pt, H).reshape(-1, 2)
                click_x_mm = real_coord[0][0]
                click_y_mm = real_coord[0][1]

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
                        break

                # If clicked on empty space, send clicked position to robot (angle = 0)
                if not clicked_on_object:
                    selected_object = None
                    show_coordinates = True
                    # Send clicked position to robot (if inside workspace)
                    if 0 <= click_x_mm <= 300 and 0 <= click_y_mm <= 300:
                        click_manager.write_click(click_x_mm, click_y_mm, mouse_button, angle=0.0, width=0.0, height=0.0)
                    else:
                        print(f"[WARNING] Click outside workspace: ({click_x_mm:.1f}, {click_y_mm:.1f}) mm")

            # Draw info panel if showing coordinates
            if show_coordinates:
                if selected_object:
                    # Show object info
                    info_lines = [
                        f"Object ID: {selected_object['id']}",
                        f"Position: ({selected_object['x_mm']:.1f}, {selected_object['y_mm']:.1f}) mm",
                        f"Angle: {selected_object['angle']:.1f} degrees",
                        f"Width: {selected_object['width']:.1f} mm",
                        f"Height: {selected_object['height']:.1f} mm"
                    ]

                    # Draw info panel
                    draw_info_panel(annotated_frame, mouse_x + 10, mouse_y + 10, info_lines, f"Object {selected_object['id']}")

                    # Highlight selected object in main view
                    cv2.polylines(annotated_frame, [selected_object['corners']], isClosed=True, color=(0, 255, 255), thickness=3)
                else:
                    # Show coordinate info at clicked position
                    click_pt = np.array([[mouse_x, mouse_y]], dtype=np.float32).reshape(-1, 1, 2)
                    real_coord = cv2.perspectiveTransform(click_pt, H).reshape(-1, 2)

                    info_lines = [
                        f"Pixel: ({mouse_x}, {mouse_y})",
                        f"Real: ({real_coord[0][0]:.1f}, {real_coord[0][1]:.1f}) mm"
                    ]

                    in_workspace = (0 <= real_coord[0][0] <= 300 and 0 <= real_coord[0][1] <= 300)
                    info_lines.append(f"In workspace: {'Yes' if in_workspace else 'No'}")

                    draw_info_panel(annotated_frame, mouse_x + 10, mouse_y + 10, info_lines, "Coordinates")

                    # Draw crosshair at clicked position
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

            # Status color - darker for white background visibility
            if status_text == "Not Ready":
                status_color = (0, 100, 200)  # Dark orange
            elif status_text == "Detect":
                status_color = (200, 150, 0)  # Dark cyan/teal
            elif status_text == "Ready":
                status_color = (0, 150, 0)  # Dark green

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

            # Draw instruction - dark color for white background
            cv2.putText(
                annotated_frame,
                "L-Click:Move | M-Click:Pick(Auto-Angle) | R-Click:Place | Q:Quit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text for white background
                2,  # Thicker for better visibility
                cv2.LINE_AA
            )

            Ydraw_transformed_box(annotated_frame, H_inv)

            cv2.imshow(window_name, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            image.Release()

    finally:
        cam.EndAcquisition()
        cam.DeInit()
        del cam
        cam_list.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()
        shm_manager.cleanup()
        click_manager.cleanup()

if __name__ == "__main__":
    main()
