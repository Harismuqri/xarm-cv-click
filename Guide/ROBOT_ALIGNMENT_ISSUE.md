# Robot Movement Alignment Issue

## Problem Report

**User reports:** Robot moves but does NOT align with the clicked target position on camera.

## System Components

The system has 3 parts working together:
1. **yolo-mouse-v1.6.py** - Camera detection and click handling
2. **xarm-motion-v1.2.py** - Robot movement control
3. **Shared memory** - Communication between them

## Current Status

✅ **Detection system working:**
- Camera detects objects correctly
- Click coordinates captured
- Coordinates written to shared memory

✅ **Robot control working:**
- Robot receives click coordinates
- Robot moves in response to clicks
- Movement commands execute successfully

❌ **Alignment problem:**
- Robot moves to WRONG position
- Click position and robot final position don't match
- Coordinate transformation issue suspected

## Likely Causes

### 1. Coordinate Offset Mismatch

**In config.json:**
```json
"coordinate_offset_x": 0,
"coordinate_offset_y": -150
```

**In xarm-motion-v1.2.py:**
```python
robot_x = x + offset_x  # Camera X + 0 = robot X
robot_y = y + offset_y  # Camera Y + (-150) = robot Y
```

**Problem:** These offsets might not be calibrated correctly for your setup.

### 2. Coordinate System Rotation

From previous diagnostics (coordinate_mapping.txt):
- Robot X+ → Visual UP
- Robot Y+ → Visual LEFT

But the click system assumes:
- Camera X+ → Right
- Camera Y+ → Up

**There might be a 90° rotation that's not accounted for in the click-to-robot transform.**

### 3. Homography Calibration Issue

The homography matrix (homography_auto.pkl) transforms:
- Pixel coordinates → Real-world mm coordinates

If this calibration is off, clicked positions will be wrong.

## Diagnostic Steps

### Step 1: Test Known Position

1. **Note robot's current position:**
   ```python
   # In xarm-motion-v1.2.py, add print:
   position = self._arm.get_position()
   print(f"Robot at: X={position[1][0]}, Y={position[1][1]}, Z={position[1][2]}")
   ```

2. **Click on camera where robot currently is**
3. **See if robot stays in place or moves**

**If robot moves:** Coordinate transformation is wrong

### Step 2: Test Simple Offset

1. **Click at workspace center (150, 150)**
2. **Check where robot goes**
3. **Measure the error:**
   - How far off in X? (mm)
   - How far off in Y? (mm)

**This tells you the offset error.**

### Step 3: Test Direction Mapping

1. **Robot at center position**
2. **Click 50mm to the RIGHT of robot**
3. **Does robot move:**
   - ✓ Right? → X mapping correct
   - ✗ Left? → X inverted
   - ✗ Up/Down? → X mapped to Y (rotation issue)

4. **Repeat for UP, DOWN, LEFT directions**

## Quick Fixes to Test

### Fix 1: Adjust Offsets in config.json

Try different offset values:

```json
"coordinate_offset_x": 0,    // Try: -50, 50, 100
"coordinate_offset_y": -150  // Try: -100, -200, 0
```

### Fix 2: Add Debug Visualization

In `yolo-mouse-v1.6.py`, when you click, draw where robot SHOULD go:

```python
# After writing click to shared memory
robot_x_visual = click_x_mm + offset_x
robot_y_visual = click_y_mm + offset_y
# Draw a blue circle at expected robot position
cv2.circle(frame, robot_pos_pixel, 15, (255, 0, 0), 3)
```

### Fix 3: Test Coordinate Transform

In `xarm-motion-v1.2.py`, add logging:

```python
def process_click(self, click_data):
    x = click_data.get("click_x", 0)  # Camera coords
    y = click_data.get("click_y", 0)

    # Transform to robot coords
    robot_x = x + self.offset_x
    robot_y = y + self.offset_y

    print(f"[TRANSFORM]")
    print(f"  Click (camera): ({x:.1f}, {y:.1f}) mm")
    print(f"  Target (robot): ({robot_x:.1f}, {robot_y:.1f}) mm")
    print(f"  Offset applied: ({self.offset_x:.1f}, {self.offset_y:.1f}) mm")

    # Move robot
    self.arm.move_to_position(x, y)
```

## Testing Procedure

### Test Script

Create `test_alignment.py`:

```python
"""Test if click coordinates align with robot movement"""
import numpy as np

# Test points in camera space (mm)
test_points = [
    (150, 150, "Center"),
    (100, 150, "Left of center"),
    (200, 150, "Right of center"),
    (150, 100, "Below center"),
    (150, 200, "Above center"),
]

print("TEST ALIGNMENT")
print("="*60)
print("Instructions:")
print("1. Move robot to CENTER (150, 150) manually")
print("2. For each test point:")
print("   - Click at the specified position")
print("   - Measure where robot actually goes")
print("   - Record the error")
print("="*60)

for x, y, desc in test_points:
    print(f"\nTest: {desc}")
    print(f"  Click at: ({x}, {y}) mm")
    print(f"  Expected robot: ???  (depends on offset)")
    print(f"  Actual robot: ___ mm (YOU FILL IN)")
    print(f"  Error: ___ mm")
```

### Data Collection

| Click Position | Expected Robot | Actual Robot | Error X | Error Y |
|---------------|----------------|--------------|---------|---------|
| (150, 150) | (?, ?) | (?, ?) | ? mm | ? mm |
| (100, 150) | (?, ?) | (?, ?) | ? mm | ? mm |
| (200, 150) | (?, ?) | (?, ?) | ? mm | ? mm |

## Next Steps

1. **Run diagnostic tests** (Steps 1-3 above)
2. **Collect error data** (Testing Procedure)
3. **Identify pattern:**
   - Constant offset? → Adjust config offsets
   - Rotation? → Need coordinate transformation fix
   - Scale wrong? → Homography re-calibration needed

4. **Share findings:**
   - What's the error pattern?
   - How far off (mm)?
   - Which direction is wrong?

## Files to Check

- `config.json` - Offset values
- `xarm-motion-v1.2.py` - Coordinate transformation
- `homography_auto.pkl` - Camera calibration
- `coordinate_mapping.txt` - Known robot axis mapping

## Questions to Answer

1. When you click at (150, 150), where does robot go?
2. If you click 50mm right, does robot move right or wrong direction?
3. Is the error consistent (same offset always)?
4. Or does error change based on position (scaling issue)?

---

**Status:** Issue documented, needs diagnostic testing to determine root cause.
