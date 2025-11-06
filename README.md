# xArm CV Click - Intelligent Robot Click Detection System

Advanced click-based control system for xArm robot using dual FLIR cameras, YOLO object detection, and intelligent angle-aware pick/place operations with precision inspection capabilities.

## System Overview

This system provides:
- **Dual Camera Setup**: Separate detection and inspection cameras
- **YOLO OBB Detection**: Oriented bounding box detection for object angle awareness
- **Two-Step Homography**: Camera → Workspace → Robot coordinate transformation
- **Auto-Calibration**: Automatic homography calibration using circular markers
- **Intelligent Gripper**: Auto-calculates optimal pick angle based on object dimensions
- **Precision Inspection**: Click-to-inspect at exact clicked position, not just object center
- **Workspace Safety**: Boundary validation prevents robot from exceeding limits

## System Components

### 1. Detection System (`yolo-mouse-v2.py`)
**Latest Version - Dual Camera with Auto-Calibration**

- Runs YOLO v8 OBB (Oriented Bounding Box) detection
- Manages two FLIR cameras:
  - **Detection Camera** (Index 0): Main workspace view for YOLO detection
  - **Inspection Camera** (Index 1): Gripper-mounted for close-up inspection
- Auto-calibrates homography transformation on every startup
- Handles mouse clicks with three buttons (left/middle/right)
- Tracks clicked position for precision inspection
- Displays real-time object info: position, angle, dimensions
- Writes commands to shared memory for robot controller

**Key Features:**
- Always performs fresh calibration (overwrites existing files)
- Camera index validation with auto-fallback
- Dual homography transformation:
  - `homography_auto.pkl`: Camera pixels → Workspace [0-300mm]
  - `homography_det_to_robot.pkl`: Workspace → Robot coordinates
- Stores clicked position separately from object center

### 2. Robot Controller (`xarm-motion-v1.6.py`)
**Intelligent Movement with Angle Awareness**

- Monitors shared memory for click and inspection commands
- Implements intelligent gripper angle calculation
- Three movement modes with angle support:
  - **Left Click**: Move to position
  - **Middle Click**: Pick with optimal angle (grips narrower dimension)
  - **Right Click**: Place maintaining original angle
- **Inspection Mode**: Position camera to view clicked location
- Workspace boundary validation
- Error recovery with automatic retry logic

**Intelligent Features:**
- `calculate_optimal_pick_angle()`: Analyzes width/height to grip narrower side
- `calculate_optimal_inspect_angle()`: Positions camera perpendicular to object
- `is_robot_position_safe()`: Validates all movements within workspace bounds
- File modification monitoring for calibration updates

### 3. Calibration Helper (`calibration-helper.py`)
**Live Camera Preview Tool**

- Shows real-time camera feed from detection camera
- Visualizes detected circles with highlighting
- Displays circle count (needs 4/4 for calibration)
- Interactive controls:
  - **'c' key**: Capture calibration when 4 circles detected
  - **'q' key**: Quit without calibrating

**Purpose:** Helps position calibration circles correctly before running main system

### 4. Configuration (`config.json`)
**Centralized System Configuration**

```json
{
  "robot_ip": "192.168.1.151",
  "tcp_speed": 300,
  "tcp_acc": 1000,
  "angle_speed": 20,
  "angle_acc": 500,
  "calibration_position": {
    "x": -95.3, "y": 211.6, "z": 172.1,
    "roll": -179.6, "pitch": -1.2, "yaw": -1.6
  },
  "click_control": {
    "safe_height": 150,
    "pick_height": -5,
    "workspace_min_x": 0,
    "workspace_max_x": 300,
    "workspace_min_y": 0,
    "workspace_max_y": 300
  },
  "camera_config": {
    "detection_camera_index": 0,
    "inspection_camera_index": 1
  },
  "camera_offset": {
    "offset_x": 92.9,
    "offset_y": -1.35,
    "offset_error": 0.4
  },
  "workspace": {
    "width": 300,
    "height": 300
  }
}
```

## Usage

### Initial Setup

1. **Position Calibration Circles** (4 white circles in workspace)
   ```bash
   python calibration-helper.py
   ```
   - Adjust circle positions until 4/4 detected (shown in green)
   - Press 'c' to capture calibration data
   - Press 'q' to exit

2. **Start Robot Controller** (creates shared memory):
   ```bash
   python xarm-motion-v1.6.py
   ```
   - Robot moves to calibration position
   - Waits for YOLO to complete auto-calibration
   - Monitors calibration file modification time
   - Returns to home position when ready

3. **Start Detection System**:
   ```bash
   python yolo-mouse-v2.py
   ```
   - Performs auto-calibration (overwrites existing files)
   - Opens two camera windows:
     - Detection window (right): YOLO detections + mouse control
     - Inspection window (left): Gripper camera view
   - Starts YOLO detection loop

### Mouse Controls

**Detection Window:**
- **Left Click**: Move robot to clicked position
- **Right Click**: Pick/Place **TOGGLE** (single button for both operations)
  - **First right-click**: PICK object
    - Auto-calculates optimal gripper angle
    - Grips narrower dimension (width or height)
    - Must complete before next pick
  - **Second right-click**: PLACE object
    - Places straight down without rotation
    - Resets state for next pick
  - **Third right-click**: PICK again (cycle repeats)
- **T Key**: Inspect at clicked position
  - Positions inspection camera to view exact clicked location
  - Uses object angle for camera orientation

**Keyboard:**
- **Q Key**: Quit both camera windows and exit system

**Note**: You **MUST** place before you can pick again! The system enforces the pick/place cycle.

### Workflow Example

1. **Click on object** (any button)
   - Info panel shows:
     - Object ID
     - Center position
     - Clicked position (different from center!)
     - Object angle, width, height
     - "Press 'T' to inspect clicked position"

2. **Right-click object to PICK**
   - Robot analyzes: width=50mm, height=30mm
   - Decision: Height < Width → Rotate 90° RIGHT from object angle
   - Grips along the 30mm dimension for better control
   - Console: `[STATE] ✅ Object picked! Next right-click will PLACE.`

3. **Press 'T' to inspect** (optional)
   - Robot moves inspection camera to view clicked position
   - NOT the object center - views exactly where you clicked
   - Camera rotates perpendicular to object angle

4. **Right-click destination to PLACE**
   - Robot places straight down without rotation
   - Simple drop at destination location
   - Console: `[STATE] ✅ Object placed! Next right-click will PICK.`

5. **Right-click another object to PICK** (cycle repeats)
   - Ready to pick the next object

## Coordinate System

### Two-Step Transformation

```
Camera Pixels → Workspace [0-300mm] → Robot Coordinates [88.9-382.0mm]
     ↓                    ↓                        ↓
homography_auto.pkl  homography_det_to_robot.pkl  Robot Movement
```

### Workspace Mapping

```
Detection workspace (mm)  →  Robot coordinates (mm)
TR: (300, 300)           →  (382.0, 14.7)
TL: (0, 300)             →  (382.0, 312.0)
BR: (300, 0)             →  (88.9, 14.7)
BL: (0, 0)               →  (88.9, 312.0)
```

### Workspace Boundaries

**Robot Coordinate Limits:**
- X-axis: 88.9mm to 382.0mm
- Y-axis: 14.7mm to 312.0mm

All movements are validated before execution. Out-of-bounds commands are rejected with clear error messages.

## Shared Memory Architecture

### 1. DetectionData
- **Name**: `"DetectionData"`
- **Size**: 4096 bytes
- **Purpose**: YOLO detection results
- **Contains**: Object positions, angles, dimensions from all detected objects

### 2. ClickData
- **Name**: `"ClickData"`
- **Size**: 512 bytes
- **Structure**:
  ```json
  {
    "click_x": 150.5,
    "click_y": 200.3,
    "button": "left",
    "angle": 45.0,
    "width": 50.0,
    "height": 30.0,
    "timestamp": 1234567890.123,
    "processed": false
  }
  ```

### 3. InspectData
- **Name**: `"InspectData"`
- **Size**: 512 bytes
- **Structure**:
  ```json
  {
    "inspect": true,
    "target_x": 150.5,
    "target_y": 200.3,
    "angle": 45.0,
    "offset_x": 92.9,
    "offset_y": -1.35,
    "offset_error": 0.4,
    "timestamp": 1234567890.123,
    "processed": false
  }
  ```

## Intelligent Movement Logic

### Pick Sequence with Auto-Angle

```python
def calculate_optimal_pick_angle(object_angle, object_width, object_height):
    """
    Analyzes object dimensions to determine optimal gripper angle.
    Always grips the narrower dimension for better control.
    """
    if object_width < object_height:
        # Width is narrower → align gripper with object angle
        gripper_angle = object_angle
    else:
        # Height is narrower → rotate gripper 90° from object
        gripper_angle = (object_angle + 90) % 180
    return gripper_angle
```

**Example:**
- Object: 50mm wide × 30mm tall at 45°
- Robot calculates: 30mm < 50mm → Rotate 90°
- Gripper angle: 45° + 90° = 135°
- Result: Grips along the 30mm (narrower) dimension

### Inspection Sequence with Click Position

```python
def inspect_sequence(target_det_x, target_det_y, object_angle):
    """
    Moves inspection camera to view exact clicked position.
    Camera rotates perpendicular to object angle for optimal viewing.
    """
    # Calculate gripper position accounting for camera offset
    gripper_x = target_det_x - CAMERA_OFFSET_X
    gripper_y = target_det_y - CAMERA_OFFSET_Y

    # Calculate perpendicular viewing angle
    camera_angle = (object_angle + 90) % 180

    # Transform to robot coordinates and validate bounds
    # Move gripper to position camera over clicked location
```

**Key Feature:**
- Inspects **clicked position**, not object center
- Example: Click on object's edge → camera views that edge
- Stores `last_click_x_mm`, `last_click_y_mm` separately from object center

### Movement Sequence Details

**1. Left Click (Move)**
```
1. Transform click → robot coordinates
2. Validate workspace boundaries
3. Move to safe height (150mm)
4. Move XY to target position
5. Mark as processed
```

**2. Right Click - First Time (Pick)**
```
1. Check pick/place state (must not be holding object)
2. Transform click → robot coordinates
3. Validate workspace boundaries
4. Calculate optimal gripper angle (grips narrower dimension)
5. Move to safe height above target
6. Rotate to gripper angle (90° RIGHT if height < width)
7. Descend to pick height (-5mm)
7. Close gripper
8. Update TCP load (object weight)
9. Lift to safe height
10. Store angle for place operation
```

**3. Right Click - Second Time (Place)**
```
1. Check pick/place state (must be holding object)
2. Transform click → robot coordinates
3. Validate workspace boundaries
4. Move to safe height above target (yaw=0, straight down)
5. Descend to place height (-5mm)
6. Open gripper
7. Reset TCP load (empty gripper weight)
8. Retract to safe height
9. Reset pick/place state (ready for next pick)
```

**Note**: Place does NOT rotate - places straight down without maintaining pick angle

**4. T Key (Inspect)**
```
1. Use stored clicked position (not object center)
2. Calculate gripper position = target - camera_offset
3. Transform gripper position → robot coordinates
4. Validate workspace boundaries
5. Calculate perpendicular camera angle
6. Move to safe height
7. Move XY to gripper position (camera over clicked spot)
8. Rotate to camera angle
9. Descend to inspection height (130.2mm)
```

## Calibration System

### Auto-Calibration Process

**Performed on every startup of yolo-mouse-v2.py:**

1. **Circle Detection**
   - Detects 4 white calibration circles in camera view
   - Uses `cv2.HoughCircles()` with parameters:
     - dp=1.2, minDist=100
     - param1=100, param2=30
     - radius: 10-50 pixels

2. **Homography Calculation**
   - Maps detected circles to known real-world positions
   - Creates `homography_auto.pkl`: Camera pixels → Workspace [0-300mm]
   - Auto-generates `homography_det_to_robot.pkl`: Workspace → Robot coordinates

3. **File Management**
   - Always overwrites existing calibration files
   - Robot detects updates by monitoring file modification time
   - No manual deletion required

### Calibration Troubleshooting

**Not enough circles detected:**
1. Run `calibration-helper.py` to see live camera view
2. Adjust lighting conditions
3. Ensure circles are white and clearly visible
4. Position circles within camera field of view
5. Check circle size (should be 10-50 pixels radius)

**Camera not showing:**
- Verify camera index in config.json
- Check camera connection (USB/ethernet)
- System auto-falls back to index 0 if configured index fails

## Requirements

### Python Packages
```bash
pip install opencv-python numpy ultralytics xarm-python-sdk
```

**Additional:**
- **PySpin**: FLIR camera SDK (download from FLIR website)
- **YOLO Model**: Trained OBB model (configured in config.json)

### Hardware
- **FLIR Cameras**: 2x cameras (detection + inspection)
- **xArm Robot**: 6-axis arm with gripper
- **Calibration Setup**: 4 white circular markers
- **Workspace**: 300mm × 300mm detection area

### Software Versions
- Python 3.8+
- OpenCV 4.x
- YOLO v8 (Ultralytics)
- xArm Python SDK 1.11+

## Configuration Parameters

### Robot Settings
- `robot_ip`: xArm IP address (default: 192.168.1.151)
- `tcp_speed`: Linear movement speed in mm/s (default: 300)
- `tcp_acc`: Linear acceleration in mm/s² (default: 1000)
- `angle_speed`: Rotational speed in °/s (default: 20)
- `angle_acc`: Rotational acceleration in °/s² (default: 500)

### Height Settings
- `safe_height`: Safe movement height in mm (default: 150)
- `pick_height`: Pick/place height in mm (default: -5)
- `inspection_height`: Inspection camera height in mm (default: 130.2)

### Camera Settings
- `detection_camera_index`: Detection camera index (default: 0)
- `inspection_camera_index`: Inspection camera index (default: 1)
- `offset_x`: Camera X offset from gripper in mm (default: 92.9)
- `offset_y`: Camera Y offset from gripper in mm (default: -1.35)
- `offset_error`: Camera offset measurement error in mm (default: 0.4)

### Workspace Settings
- `workspace_min_x/max_x`: Detection workspace X bounds (0-300mm)
- `workspace_min_y/max_y`: Detection workspace Y bounds (0-300mm)
- `width/height`: Workspace dimensions in mm (300×300)

### YOLO Settings
- `yolo_model_path`: Path to trained YOLO model weights
- `detection_confidence`: Minimum confidence threshold (default: 0.8)

## Troubleshooting

### Robot Not Responding to Clicks
1. Check console for `[CLICK-MOVE/PICK/PLACE]` messages from detection system
2. Check console for click processing messages from robot controller
3. Verify both programs are running (robot controller first!)
4. Ensure clicks are within workspace bounds (0-300mm)
5. Check shared memory connection

### Calibration Issues
- **No circles detected**: Run `calibration-helper.py` for live preview
- **Calibration file not updating**: Check file permissions in script directory
- **Robot stuck at calibration position**: Verify YOLO is running and detecting circles

### Inspection Camera Not Working
- Verify inspection camera index in config.json
- Check camera connection and power
- Ensure camera offset values are correct
- System continues with detection-only if inspection camera fails

### Gripper Angle Issues
- Verify object dimensions are being detected correctly
- Check console output for angle calculation logic
- Object width/height from YOLO OBB should be in mm (transformed)
- Angle should be 0-180° from YOLO detection

### Workspace Boundary Violations
```
[Pick] ❌ POSITION OUT OF BOUNDS
[Pick] Robot position: (400.0, 250.0) mm
[Pick] Workspace limits: X=[88.9-382.0], Y=[14.7-312.0]
```
**Solution:**
- Click closer to workspace center
- Check coordinate transformation is correct
- Verify homography calibration is accurate
- Ensure calibration circles are properly positioned

## Safety Features

### Workspace Validation
- All movements validated against robot coordinate boundaries
- Pre-movement position checking prevents out-of-bounds operations
- Clear error messages with actual position vs limits

### Error Recovery
- Automatic error detection and recovery
- Retry logic for transient failures
- Safe heights maintained during movement
- Home position on startup and shutdown

### Physical Safety
- Safe height movement (150mm) before XY positioning
- Gradual descent to pick/place height
- Gripper force limits (configured in xArm)
- Emergency stop capability (robot's physical button)

## File Structure

```
xarm-cv-click/
├── xarm-auto/
│   ├── yolo-mouse-v2.py           # Detection system (latest)
│   ├── xarm-motion-v1.6.py        # Robot controller (latest)
│   ├── calibration-helper.py      # Calibration preview tool
│   ├── config.json                # System configuration
│   ├── homography_auto.pkl        # Camera→Workspace transform
│   └── homography_det_to_robot.pkl # Workspace→Robot transform
├── README.md                       # This file
└── [legacy files...]
```

## Version History

### Latest Updates (v1.6 / v2.0) - Current Version
- **Pick/Place Toggle**: Single button (right-click) for both pick and place operations
  - First right-click: PICK object
  - Second right-click: PLACE object
  - State enforcement prevents double-pick
- **Place Straight Down**: Place operation no longer rotates (yaw=0)
- **Lite6 Gripper Fix**: Fixed gripper commands to use Lite6 API instead of modbus
- **Rotation Direction Fix**: Gripper now rotates RIGHT (-90°) instead of LEFT (+90°)
- **Clean Ctrl+C Shutdown**: No more traceback when pressing Ctrl+C
- **T Key Inspection**: Inspection via 'T' key (middle click unused/reserved)
- **Click-to-Inspect**: Inspect at clicked position instead of object center
- **Updated Calibration Position**: Tuned for optimal camera view
- **Config Updates**: Refined heights and camera indices
- **Enhanced Info Panel**: Shows both center and clicked positions

### v1.6 / v2.0 Core Features
- Dual camera system with inspection support
- Two-step homography transformation
- Auto-calibration with file monitoring
- Intelligent angle calculation for pick/place
- Workspace boundary validation
- Inspection mode with 'T' key
- Click position tracking
- Enhanced error messages and debugging

### v1.5
- YOLO OBB detection integration
- Object angle detection
- Auto-calculated gripper opening
- Angle-following for pick/place

### v1.2-1.4
- Basic click control implementation
- Shared memory architecture
- Three-button mouse support
- Movement/pick/place sequences

## Contributing

When modifying the system:
1. Test calibration thoroughly after changes
2. Validate workspace boundaries with test movements
3. Check both cameras work correctly
4. Verify angle calculations with objects of different sizes
5. Test inspection at various clicked positions
6. Update config.json documentation if adding new parameters

## License

This project is part of xArm robot control research and development.

## Support

For issues or questions:
- Check console output for detailed error messages
- Verify config.json parameters match your hardware setup
- Use calibration-helper.py to diagnose camera/calibration issues
- Review coordinate transformation mathematics if position accuracy is off
