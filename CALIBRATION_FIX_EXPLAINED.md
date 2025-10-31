# Calibration Synchronization Fix

## Problem Description

The red dot showing robot position on the camera view was NOT synchronized with the actual robot arm movements. When the robot moved, the dot would appear in the wrong location or not move correctly.

## Root Cause

**90-degree coordinate system rotation** was not being accounted for in the visualization code!

### Coordinate System Mapping

From diagnostic tests (`coordinate_mapping.txt`):
- **Robot X+** → Moves visually **UP** on camera
- **Robot Y+** → Moves visually **LEFT** on camera

This means:
- Robot X-axis corresponds to Camera/Workspace Y-axis (vertical)
- Robot Y-axis corresponds to Camera/Workspace X-axis (horizontal, inverted)

### The Bug

The old code in `robot_keyboard_calibration.py` used a **simple offset subtraction** that assumed axes were aligned:

```python
# ❌ WRONG - Assumes X→X and Y→Y (no rotation)
camera_x = robot_x - offset_x
camera_y = robot_y - offset_y
```

This ignores the 90-degree rotation between coordinate systems!

## The Fix

### New Coordinate Transformation Function

Added `robot_to_camera_coords()` method that properly accounts for rotation:

```python
def robot_to_camera_coords(self, robot_x, robot_y):
    """
    Convert robot coordinates to camera/workspace coordinates.
    Accounts for 90-degree rotation between coordinate systems.

    From diagnostic tests:
    - Robot X+ → Visual UP → Camera Y+
    - Robot Y+ → Visual LEFT → Camera X-

    Therefore:
    - camera_y = robot_x - offset_x
    - camera_x = offset_y - robot_y
    """
    camera_x = self.offset_y - robot_y
    camera_y = robot_x - self.offset_x
    return camera_x, camera_y
```

### Mathematical Explanation

With offsets: `offset_x = -208`, `offset_y = -148`

#### Old (Wrong) Transform:
```
camera_x = robot_x - (-208) = robot_x + 208
camera_y = robot_y - (-148) = robot_y + 148
```
This treats X→X and Y→Y, which is incorrect!

#### New (Correct) Transform:
```
camera_x = offset_y - robot_y = -148 - robot_y
camera_y = robot_x - offset_x = robot_x - (-208) = robot_x + 208
```

This properly maps:
- Robot X → Camera Y (vertical axis)
- Robot Y → Camera X (horizontal axis, inverted)

### Visualization Example

If robot is at `(robot_x=50, robot_y=100)`:

**Old (wrong):**
- camera_x = 50 + 208 = 258
- camera_y = 100 + 148 = 248
- Dot appears at (258, 248) in workspace - **WRONG LOCATION**

**New (correct):**
- camera_x = -148 - 100 = -248 (off-screen, robot outside workspace)
- camera_y = 50 + 208 = 258
- Correctly shows robot position accounting for rotation

## Changes Made

Updated **7 locations** in `robot_keyboard_calibration.py` where coordinate transformation occurs:

1. **`draw_robot_position()`** (line ~302) - Visual dot on camera
2. **`draw_saved_points()`** (line ~376) - Saved calibration points visualization
3. **`camera_display_thread()`** (line ~418) - Info text overlay
4. **`save_calibration_point()`** (line ~526) - Console output when saving
5. **`display_current_position()`** (line ~595) - Console position display
6. **`view_calibration_points()`** (line ~634) - Viewing saved points
7. **`export_calibration()`** (line ~671) - JSON export

All now use the centralized `robot_to_camera_coords()` method.

## Testing the Fix

### Before Fix:
- Move robot up → Dot moves in wrong direction or stays still
- Move robot left → Dot moves incorrectly
- Dot position doesn't match actual robot location

### After Fix:
- Move robot up (X+) → Dot moves UP on screen ✓
- Move robot down (X-) → Dot moves DOWN on screen ✓
- Move robot left (Y+) → Dot moves LEFT on screen ✓
- Move robot right (Y-) → Dot moves RIGHT on screen ✓
- Dot accurately tracks robot TCP position

## How to Use

1. **Run the calibration tool:**
   ```bash
   python robot_keyboard_calibration.py
   ```

2. **Use arrow keys** to move robot:
   - ↑ = Move up (robot X+)
   - ↓ = Move down (robot X-)
   - ← = Move left (robot Y+)
   - → = Move right (robot Y-)

3. **Watch the camera window** - The red dot should now perfectly track the robot's end effector position!

4. **Align robot to center marker** (yellow crosshair at 150, 150) for calibration

## Why Movement Controls Still Work

The keyboard movement controls were **already correct** and didn't need changes:

```python
if key == 'up':
    self.move_relative(dx=self.step_size)  # Robot X+
elif key == 'left':
    self.move_relative(dy=self.step_size)  # Robot Y+
```

These directly command the robot in **robot coordinates**, which is correct. The bug was ONLY in the **visualization** (converting robot position to camera position for display).

## Related Files

- `coordinate_mapping.txt` - Documents the axis mapping from diagnostic tests
- `find_coordinate_mapping.py` - Tool used to discover the mapping
- `debug_dot_position.py` - Tool to debug dot visibility issues
- `config.json` - Contains offset values (-208, -148)

## Technical Notes

### Why the Rotation Exists

The robot's coordinate system and the camera/workspace coordinate system have different orientations:

- **Robot:** X-axis points forward (toward workspace top), Y-axis points left
- **Camera/Workspace:** X-axis points right, Y-axis points up (standard image coords)

This 90° rotation is common in robotics when mounting cameras at different angles.

### Alternative Formulation

The transformation can also be expressed as a rotation matrix:

```
[camera_x]   [  0  -1 ] [robot_y]   [offset_y]
[camera_y] = [  1   0 ] [robot_x] + [offset_x]
```

This is a 90° counter-clockwise rotation plus translation.
