# xArm CV Click - Robot Click Detection System

This system enables click-based control of an xArm robot using computer vision and mouse interaction.

## Critical Bug Fix

### Issue Identified
The original `yolo-mouse-v1.6.py` file contained **duplicate class method definitions** for `ClickDataManager` (around line 357-450). These duplicate methods:
- Overwrote the correct implementation
- Used wrong shared memory name (`SHARED_MEMORY_NAME` instead of `CLICK_MEMORY_NAME`)
- **Removed the critical `write_click()` method**

This caused silent failures when clicking - the method simply didn't exist!

### Solution Applied
- Removed all duplicate method definitions from `ClickDataManager`
- Ensured proper shared memory initialization with correct memory names
- Verified `write_click()` method is present and functional

## System Components

### 1. Detection System (`yolo-mouse-v1.6.py`)
- Uses YOLO for object detection
- Handles mouse click events (Left/Middle/Right buttons)
- Writes click coordinates to shared memory
- Requires FLIR camera and calibration

### 2. Robot Controller (`xarm-motion-v1.2.py`)
- Monitors shared memory for click events
- Controls xArm robot movements
- Three modes based on mouse button:
  - **Left Click**: Move to position
  - **Middle Click**: Pick sequence
  - **Right Click**: Place sequence

### 3. Configuration (`config.json`)
- Robot IP address
- Speed and acceleration settings
- Workspace boundaries
- Coordinate offsets for camera-to-robot transformation

## Usage

### Starting the System

1. **Start Robot Controller First** (creates shared memory):
   ```bash
   python xarm-motion-v1.2.py
   ```

2. **Start Detection System**:
   ```bash
   python yolo-mouse-v1.6.py
   ```

### Mouse Controls
- **Left Click**: Move robot to clicked position
- **Middle Click**: Execute pick sequence at clicked position
- **Right Click**: Execute place sequence at clicked position
- **Q Key**: Quit detection system

## Shared Memory Architecture

### DetectionData
- Name: `"DetectionData"`
- Size: 4096 bytes
- Contains: Object positions, angles, dimensions

### ClickData (Fixed!)
- Name: `"ClickData"`
- Size: 512 bytes
- Structure:
  ```json
  {
    "click_x": 150.5,
    "click_y": 200.3,
    "button": "left",
    "timestamp": 1234567890.123,
    "processed": false
  }
  ```

## Troubleshooting

### Robot Not Responding to Clicks
- **FIXED**: Ensure you're using the corrected `yolo-mouse-v1.6.py` without duplicate methods
- Check console for `[CLICK-MOVE/PICK/PLACE]` messages from detection system
- Check console for `[DEBUG] Found unprocessed click` messages from robot controller
- Verify both programs are running
- Ensure clicks are within workspace bounds (0-300mm x 0-300mm)

### Shared Memory Issues
- If "shared memory already exists" errors occur, restart both programs
- Robot controller should start first to create clean shared memory

### Workspace Boundaries
- Camera coordinates: 0-300mm x 0-300mm
- Robot coordinates: Transformed using offsets in config.json
- Default offset: X=0mm, Y=-150mm

## Requirements

### Python Packages
- OpenCV (`cv2`)
- NumPy
- Ultralytics YOLO
- PySpin (FLIR camera SDK)
- xArm-Python-SDK

### Hardware
- FLIR camera
- xArm robot (with Lite6 gripper)
- Calibration markers (4 circles)

## Configuration Parameters

Edit `config.json` to adjust:
- `robot_ip`: xArm IP address
- `tcp_speed`: Movement speed (mm/s)
- `safe_height`: Height for safe movement (mm)
- `pick_height`: Height for picking/placing (mm)
- `coordinate_offset_x/y`: Camera-to-robot coordinate transformation

## Technical Details

### Coordinate Transformation
```
Robot X = Camera X + offset_x
Robot Y = Camera Y + offset_y
```

### Movement Sequence (Left Click)
1. Read click coordinates from shared memory
2. Transform to robot coordinates
3. Move to position at safe height
4. Mark click as processed

### Pick Sequence (Middle Click)
1. Move above object (safe height)
2. Move down (pick height)
3. Close gripper
4. Update TCP load
5. Lift object

### Place Sequence (Right Click)
1. Move above target (safe height)
2. Move down (pick height)
3. Open gripper
4. Reset TCP load
5. Retract to safe height

## Safety Features
- Workspace boundary checking
- Position validation before movement
- Error recovery and retry logic
- Safe heights for movement
- Home position on startup/shutdown
