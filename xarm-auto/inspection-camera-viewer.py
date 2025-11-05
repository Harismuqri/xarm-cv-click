"""
Inspection Camera Viewer - Separate from Detection System
Shows real-time feed from inspection camera mounted on gripper
"""
import cv2
import PySpin
import numpy as np
import time

# Configuration
INSPECTION_CAMERA_SERIAL = None  # Set to specific serial number, or None for first camera
WINDOW_NAME = "Inspection Camera"
SHOW_CROSSHAIR = True
SHOW_INFO = True  # Enable info overlay

def convert_pyspin_image_to_cv2(image):
    """Convert PySpin image to OpenCV format"""
    if not image.IsValid():
        return None
    img_array = image.GetNDArray()
    if len(img_array.shape) == 2:
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif len(img_array.shape) == 3:
        return img_array
    return None

def draw_crosshair(frame, color=(0, 0, 255), thickness=3):
    """Draw minimal crosshair at center of frame"""
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Longer lines
    line_length = 40
    
    # Horizontal line
    cv2.line(frame, (center_x - line_length, center_y), (center_x + line_length, center_y), color, thickness)
    # Vertical line
    cv2.line(frame, (center_x, center_y - line_length), (center_x, center_y + line_length), color, thickness)
    # Center dot
    cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), 2)

def draw_info_overlay(frame):
    """Draw information overlay on frame"""
    h, w = frame.shape[:2]
    
    # Semi-transparent background for info
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (500, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw info text with larger font
    font = cv2.FONT_HERSHEY_SIMPLEX
    info_lines = [
        "INSPECTION CAMERA",
        f"Resolution: {w} x {h}",
        "Press 'Q' to quit"
    ]
    
    y_offset = 50
    for line in info_lines:
        cv2.putText(frame, line, (25, y_offset), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        y_offset += 40

def get_inspection_camera(system, cam_list):
    """Get inspection camera (second camera or by serial number)"""
    if INSPECTION_CAMERA_SERIAL:
        # Find camera by serial number
        for i, cam in enumerate(cam_list):
            cam.Init()
            serial = cam.TLDevice.DeviceSerialNumber.GetValue()
            if serial == INSPECTION_CAMERA_SERIAL:
                print(f"[INFO] Found inspection camera by serial: {serial}")
                return cam
            cam.DeInit()
        raise RuntimeError(f"Camera with serial {INSPECTION_CAMERA_SERIAL} not found")
    else:
        # Use first camera (gripper-mounted inspection camera)
        if cam_list.GetSize() < 1:
            raise RuntimeError("No camera found.")
        cam = cam_list.GetByIndex(0)  # Index 0 = first camera (inspection on gripper)
        cam.Init()
        serial = cam.TLDevice.DeviceSerialNumber.GetValue()
        print(f"[INFO] Using first camera (index 0), Serial: {serial}")
        return cam

def main():
    """Main inspection camera viewer"""
    print("="*60)
    print("INSPECTION CAMERA VIEWER")
    print("="*60)
    print("Starting inspection camera...")
    print("Press 'Q' to quit")
    print("="*60 + "\n")
    
    # Initialize PySpin system
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    
    num_cameras = cam_list.GetSize()
    print(f"[INFO] Number of cameras detected: {num_cameras}")
    
    if num_cameras == 0:
        print("[ERROR] No cameras detected!")
        cam_list.Clear()
        system.ReleaseInstance()
        return
    
    try:
        # Get inspection camera
        cam = get_inspection_camera(system, cam_list)
        
        # Configure camera
        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        
        # Start acquisition
        cam.BeginAcquisition()
        print("[INFO] Inspection camera started successfully\n")
        
        # Create window with better visual settings
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, 960, 720)
        cv2.moveWindow(WINDOW_NAME, 100, 50)  # Position window nicely on screen
        
        while True:
            # Get image
            image = cam.GetNextImage(1000)
            
            if image.IsIncomplete():
                print(f"[WARNING] Image incomplete: {image.GetImageStatus()}")
                image.Release()
                continue
            
            # Convert to OpenCV format
            frame = convert_pyspin_image_to_cv2(image)
            if frame is None:
                image.Release()
                continue
            
            # Make writable copy
            frame = frame.copy()
            
            # Calculate FPS
            
            # Draw overlays
            if SHOW_CROSSHAIR:
                draw_crosshair(frame)
            
            if SHOW_INFO:
                draw_info_overlay(frame)
            
            # Display frame
            cv2.imshow(WINDOW_NAME, frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[INFO] Quitting...")
                break
            
            image.Release()
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n[INFO] Cleaning up...")
        try:
            cam.EndAcquisition()
            cam.DeInit()
        except:
            pass
        del cam
        cam_list.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()
        print("[INFO] Inspection camera viewer closed")

if __name__ == "__main__":
    main()