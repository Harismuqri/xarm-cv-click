#!/usr/bin/env python3
"""
Standalone Detection Monitor with Live Display
- Uses FLIR camera 0
- Runs YOLO detection independently
- Shows live video feed with detection overlays
- Displays both pixel and real-world coordinates
- No dependencies on other scripts
"""

import cv2
import numpy as np
import PySpin
from ultralytics import YOLO
import json
import pickle
import os
import sys

# Configuration
CONFIG_FILE = "config.json"
YOLO_MODEL_PATH = "D:\\2. yolo\\train30\\weights\\best.pt"
HOMOGRAPHY_FILE = "homography_auto.pkl"
DETECTION_CONFIDENCE = 0.8

# Display colors
COLOR_BOX = (0, 255, 0)  # Green
COLOR_TEXT = (255, 255, 255)  # White
COLOR_BG = (0, 0, 0)  # Black
COLOR_VERTICAL = (255, 100, 100)  # Light blue for vertical
COLOR_HORIZONTAL = (100, 255, 100)  # Light green for horizontal

def load_config():
    """Load configuration from JSON file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        print(f"[Config] Loaded from {CONFIG_FILE}")
        return config
    except Exception as e:
        print(f"[Config] Error loading config: {e}")
        return None

def load_homography():
    """Load homography transformation matrix."""
    try:
        with open(HOMOGRAPHY_FILE, 'rb') as f:
            H = pickle.load(f)
        print(f"[Homography] Loaded from {HOMOGRAPHY_FILE}")
        return H
    except Exception as e:
        print(f"[Homography] Warning: Could not load homography matrix: {e}")
        print("[Homography] Will show pixel coordinates only")
        return None

def initialize_camera(camera_index=0):
    """Initialize FLIR camera."""
    try:
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        
        if cam_list.GetSize() == 0:
            print("[ERROR] No cameras detected!")
            cam_list.Clear()
            system.ReleaseInstance()
            return None, None
        
        cam = cam_list.GetByIndex(camera_index)
        cam.Init()
        
        # Configure camera
        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        cam.BeginAcquisition()
        
        print(f"[Camera] Initialized camera index {camera_index}")
        return system, cam
    except Exception as e:
        print(f"[ERROR] Camera initialization failed: {e}")
        return None, None

def convert_image_to_cv2(pyspin_image):
    """Convert PySpin image to OpenCV format."""
    try:
        if pyspin_image.IsIncomplete():
            return None
        
        image_converted = pyspin_image.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)
        img_array = image_converted.GetNDArray()
        return img_array
    except Exception as e:
        print(f"[ERROR] Image conversion failed: {e}")
        return None

def transform_to_real_world(points, H):
    """Transform pixel coordinates to real-world mm coordinates."""
    if H is None:
        return None
    try:
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, H)
        return transformed.reshape(-1, 2)
    except Exception as e:
        return None

def get_angle_from_obb(corners):
    """Calculate angle from oriented bounding box corners."""
    try:
        vec = corners[1] - corners[0]
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        # Normalize to 0-180 range
        if angle < 0:
            angle += 180
        return angle
    except:
        return 0

def determine_orientation(width, height):
    """Determine object orientation based on dimensions."""
    if height == 0:
        return "UNKNOWN", "N/A", (200, 200, 200)
    
    aspect_ratio = width / height
    
    if aspect_ratio > 1.5:
        return "HORIZONTAL", "BACKWARD (180°)", COLOR_HORIZONTAL
    elif aspect_ratio < 0.67:
        return "VERTICAL", "LEFT (0°)", COLOR_VERTICAL
    else:
        return "MIXED", "AUTO", (200, 200, 100)

def draw_text_with_background(img, text, pos, font_scale=0.6, thickness=2):
    """Draw text with black background for better visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = pos
    # Draw background rectangle
    cv2.rectangle(img, (x, y - text_height - 5), (x + text_width + 5, y + 5), COLOR_BG, -1)
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA)
    return text_height + 10

def main():
    """Main detection and display loop."""
    print("\n" + "="*80)
    print("STANDALONE DETECTION MONITOR")
    print("="*80)
    print("Press 'Q' to quit")
    print("="*80 + "\n")
    
    # Load configuration
    config = load_config()
    if config:
        model_path = config.get('yolo_model_path', YOLO_MODEL_PATH)
        confidence = config.get('detection_confidence', DETECTION_CONFIDENCE)
    else:
        model_path = YOLO_MODEL_PATH
        confidence = DETECTION_CONFIDENCE
    
    # Load YOLO model
    print(f"[YOLO] Loading model from {model_path}")
    try:
        model = YOLO(model_path)
        print("[YOLO] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        return
    
    # Load homography
    H = load_homography()
    
    # Initialize camera
    system, cam = initialize_camera(0)
    if cam is None:
        print("[ERROR] Cannot start without camera")
        return
    
    print("\n[INFO] Starting detection monitor...")
    print("[INFO] Detection overlay will be displayed on screen")
    
    try:
        while True:
            # Get camera frame
            image_result = cam.GetNextImage()
            if image_result.IsIncomplete():
                image_result.Release()
                continue
            
            frame = convert_image_to_cv2(image_result)
            image_result.Release()
            
            if frame is None:
                continue
            
            # Run YOLO detection
            results = model(frame, conf=confidence, verbose=False)
            
            # Create display frame
            display_frame = frame.copy()
            
            # Process detections
            if len(results) > 0 and hasattr(results[0], 'obb') and results[0].obb is not None:
                obb_preds = results[0].obb
                
                for i, obb in enumerate(obb_preds, 1):
                    try:
                        # Get bounding box corners
                        if hasattr(obb, "xyxyxyxy"):
                            corners = obb.xyxyxyxy.cpu().numpy().reshape(-1, 2)
                        elif hasattr(obb, "xyxy"):
                            corners = obb.xyxy.cpu().numpy().reshape(-1, 2)
                        else:
                            continue
                        
                        # Calculate properties
                        center_px = np.mean(corners, axis=0)
                        angle = get_angle_from_obb(corners)
                        
                        # Calculate dimensions in pixels
                        side1 = np.linalg.norm(corners[1] - corners[0])
                        side2 = np.linalg.norm(corners[2] - corners[1])
                        
                        # Determine width and height based on angle
                        angle_normalized = angle % 180
                        if 45 <= angle_normalized <= 135:
                            width_px = min(side1, side2)
                            height_px = max(side1, side2)
                        else:
                            width_px = max(side1, side2)
                            height_px = min(side1, side2)
                        
                        # Transform to real-world coordinates
                        if H is not None:
                            corners_real = transform_to_real_world(corners, H)
                            center_real = transform_to_real_world([center_px], H)
                            
                            if corners_real is not None and center_real is not None:
                                side1_real = np.linalg.norm(corners_real[1] - corners_real[0])
                                side2_real = np.linalg.norm(corners_real[2] - corners_real[1])
                                
                                if 45 <= angle_normalized <= 135:
                                    width_mm = min(side1_real, side2_real)
                                    height_mm = max(side1_real, side2_real)
                                else:
                                    width_mm = max(side1_real, side2_real)
                                    height_mm = min(side1_real, side2_real)
                                
                                x_mm, y_mm = center_real[0]
                            else:
                                width_mm = height_mm = x_mm = y_mm = 0
                        else:
                            width_mm = height_mm = x_mm = y_mm = 0
                        
                        # Determine orientation
                        if width_mm > 0 and height_mm > 0:
                            orientation, direction, color = determine_orientation(width_mm, height_mm)
                            aspect = width_mm / height_mm
                        else:
                            orientation, direction, color = determine_orientation(width_px, height_px)
                            aspect = width_px / height_px if height_px > 0 else 0
                        
                        # Draw bounding box
                        corners_int = corners.astype(int)
                        cv2.polylines(display_frame, [corners_int], isClosed=True, color=color, thickness=3)
                        
                        # Draw center point
                        center_int = center_px.astype(int)
                        cv2.circle(display_frame, tuple(center_int), 6, color, -1)
                        cv2.circle(display_frame, tuple(center_int), 6, COLOR_TEXT, 2)
                        
                        # Prepare text information
                        text_x = corners_int[0][0]
                        text_y = corners_int[0][1] - 10
                        
                        # Draw all information
                        text_y -= draw_text_with_background(display_frame, f"Object {i}", (text_x, text_y))
                        text_y -= draw_text_with_background(display_frame, f"Angle: {angle:.1f}°", (text_x, text_y))
                        
                        if H is not None and width_mm > 0:
                            text_y -= draw_text_with_background(display_frame, 
                                f"Pos: ({x_mm:.1f}, {y_mm:.1f}) mm", (text_x, text_y))
                            text_y -= draw_text_with_background(display_frame, 
                                f"Size: {width_mm:.1f}×{height_mm:.1f} mm", (text_x, text_y))
                        else:
                            text_y -= draw_text_with_background(display_frame, 
                                f"Pos: ({center_px[0]:.0f}, {center_px[1]:.0f}) px", (text_x, text_y))
                            text_y -= draw_text_with_background(display_frame, 
                                f"Size: {width_px:.0f}×{height_px:.0f} px", (text_x, text_y))
                        
                        text_y -= draw_text_with_background(display_frame, 
                            f"Aspect: {aspect:.2f}", (text_x, text_y))
                        text_y -= draw_text_with_background(display_frame, 
                            f"Orient: {orientation}", (text_x, text_y))
                        text_y -= draw_text_with_background(display_frame, 
                            f"Robot: {direction}", (text_x, text_y))
                    
                    except Exception as e:
                        print(f"[ERROR] Processing detection {i}: {e}")
                        continue
            
            # Display frame
            cv2.imshow("Detection Monitor - Press 'Q' to quit", display_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[INFO] Quit requested")
                break
    
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Runtime error: {e}")
    finally:
        # Cleanup
        print("\n[INFO] Cleaning up...")
        if cam is not None:
            cam.EndAcquisition()
            cam.DeInit()
            del cam
        if system is not None:
            cam_list = system.GetCameras()
            cam_list.Clear()
            system.ReleaseInstance()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown complete")

if __name__ == "__main__":
    main()
