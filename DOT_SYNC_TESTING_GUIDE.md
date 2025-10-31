# Robot Dot Synchronization Testing Guide

## Quick Fix Summary

### Changes Made

1. **✅ Made center marker smaller** - Now shows as small cyan circle with crosshair
2. **✅ Added transformation mode cycling** - Press `m` to test 5 different coordinate transformations
3. **✅ Enhanced debug output** - Shows robot, camera, and pixel coordinates

## Problem: Dot Still Not Syncing

If the robot position dot is still not syncing with actual arm movement, the coordinate transformation formula might need adjustment. The system now has **5 different transformation modes** you can test.

## Testing Procedure

### Step 1: Start Calibration Tool

```bash
python robot_keyboard_calibration.py
```

### Step 2: Move Robot to Known Position

Use arrow keys to move the robot somewhere clearly visible on camera.

**Watch two things:**
1. The **red dot** on the camera view
2. The **actual robot arm position**

### Step 3: Test Arrow Key Movements

Press each arrow key and verify:

| Key Press | Robot Should | Dot Should |
|-----------|-------------|------------|
| ↑ (UP) | Move UP (away from you) | Move UP on screen |
| ↓ (DOWN) | Move DOWN (toward you) | Move DOWN on screen |
| ← (LEFT) | Move LEFT | Move LEFT on screen |
| → (RIGHT) | Move RIGHT | Move RIGHT on screen |

**If dot moves in WRONG direction**, proceed to Step 4.

### Step 4: Cycle Through Transformation Modes

Press **`m`** to cycle through 5 different coordinate transformation modes:

#### **Mode 0** (Default - Expected to work)
```
camera_x = -robot_y - offset_y
camera_y = robot_x - offset_x
```
- Based on diagnostic: Robot X+ → Visual UP, Robot Y+ → Visual LEFT
- Accounts for 90° rotation

#### **Mode 1** (Alternative rotation direction)
```
camera_x = robot_y + offset_y
camera_y = robot_x - offset_x
```
- Different rotation direction

#### **Mode 2** (No rotation - simple offset)
```
camera_x = robot_x - offset_x
camera_y = robot_y - offset_y
```
- If coordinate systems are actually aligned

#### **Mode 3** (Rotation variant 3)
```
camera_x = robot_x - offset_x
camera_y = -robot_y - offset_y
```
- Y-axis inverted

#### **Mode 4** (Rotation variant 4)
```
camera_x = -robot_y - offset_y
camera_y = -robot_x + offset_x
```
- Both axes rotated and inverted

### Step 5: Find the Correct Mode

For **each mode** (press `m` to cycle):

1. **Move robot UP** (↑ key)
   - Does red dot move UP? ✓ or ✗

2. **Move robot DOWN** (↓ key)
   - Does red dot move DOWN? ✓ or ✗

3. **Move robot LEFT** (← key)
   - Does red dot move LEFT? ✓ or ✗

4. **Move robot RIGHT** (→ key)
   - Does red dot move RIGHT? ✓ or ✗

**When you find a mode where ALL 4 directions match**, that's the correct one!

### Step 6: Record the Correct Mode

Once you find the working mode, note it down:

```
CORRECT TRANSFORMATION MODE: ___

Test results:
- UP arrow → Dot moves UP: ✓
- DOWN arrow → Dot moves DOWN: ✓
- LEFT arrow → Dot moves LEFT: ✓
- RIGHT arrow → Dot moves RIGHT: ✓
```

## Debug Output Analysis

Watch the console output when moving the robot. For each movement you'll see:

```
[DEBUG - Mode 0]
  Robot:  X=  150.0, Y= -300.0, Z=  200.0 mm
  Camera: X=  150.0, Y=  150.0 mm
  Pixel:  X= 640, Y= 480 ✓ IN FRAME
  Offset: X=    0.0, Y= -150.0 mm
```

### What to Check:

1. **Robot coords change** when you press arrow keys? ✓
2. **Camera coords make sense?** (should be 0-300 range for workspace)
3. **Pixel coords visible?** (should be within camera frame dimensions)
4. **Dot appears where pixel coords indicate?** ✓

## Common Issues & Solutions

### Issue 1: Dot is Off-Screen

**Symptom:** Console shows "✗ OFF-SCREEN"

**Possible causes:**
- Robot is outside the calibrated 300x300mm workspace
- Wrong transformation mode
- Homography calibration is incorrect

**Solution:**
- Move robot to workspace center (150, 150) first
- Try different transformation modes with `m`
- Re-run homography calibration if needed

### Issue 2: Dot Moves in Opposite Direction

**Symptom:** UP arrow makes dot move DOWN

**Solution:**
- Press `m` to cycle through modes
- Mode with axis inversions (Mode 3 or 4) might work

### Issue 3: Dot Moves But At Wrong Angle

**Symptom:** UP arrow makes dot move diagonally or sideways

**Solution:**
- This indicates wrong rotation
- Try Mode 0, 1, 3, or 4 (not Mode 2)
- One of the rotational modes should match

### Issue 4: Dot Appears But Doesn't Move

**Symptom:** Dot visible but stays still when robot moves

**Possible causes:**
- `update_current_position()` not being called
- Robot position not updating
- Previous position tracking issue

**Solution:**
- Press `p` to force position update
- Check console debug output shows changing robot coords

## Camera View Info

On the camera window you'll see:

```
Robot Calibration - Live View
Transform Mode: 0 (press 'm' to cycle)    ← Current mode
Robot: (150.0, -300.0) mm                 ← Robot coordinates
Camera: (150.0, 150.0) mm                 ← Workspace coordinates
Step: 10mm                                ← Movement step size
```

**Key indicators:**
- **Cyan circle + "C"**: Workspace center (150, 150) marker (now small!)
- **Red dot**: Current robot TCP position
- **White box**: Workspace boundary (300x300mm)
- **Green dots**: Saved calibration points

## After Finding Correct Mode

Once you've identified the working transformation mode, update the code:

1. Edit `robot_keyboard_calibration.py`
2. Set the default mode in `__init__`:
   ```python
   self.transform_mode = X  # Replace X with correct mode number
   ```
3. Or better: Update the `robot_to_camera_coords()` function to permanently use the correct formula

## Need Help?

### Collect This Information:

1. **Current transform mode when it works:** _____
2. **Your offset values:**
   - offset_x = _____
   - offset_y = _____
3. **Sample robot position:** (X, Y, Z) = _____, _____, _____
4. **Corresponding camera position:** (X, Y) = _____, _____
5. **Does robot move match dot movement?** Yes/No per direction

### Commands for Testing:

- `m` - Cycle transformation mode
- `p` - Print current position
- `↑↓←→` - Move robot to test sync
- `?` - Show all controls
- `h` - Go to home position
