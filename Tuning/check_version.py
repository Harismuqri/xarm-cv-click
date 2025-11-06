"""
Version Checker - Verify you're running the latest robot_keyboard_calibration.py
"""

import os

print("\n" + "="*70)
print("ROBOT KEYBOARD CALIBRATION - VERSION CHECKER")
print("="*70)

# Check if file exists
file_path = "robot_keyboard_calibration.py"
if not os.path.exists(file_path):
    print(f"\n❌ ERROR: {file_path} not found in current directory!")
    print(f"Current directory: {os.getcwd()}")
    exit(1)

print(f"\n✓ File found: {file_path}")
print(f"  Location: {os.path.abspath(file_path)}")

# Check file size
file_size = os.path.getsize(file_path)
print(f"  Size: {file_size} bytes")

# Check for version markers
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

print("\n" + "-"*70)
print("CHECKING FOR LATEST VERSION MARKERS")
print("-"*70)

# Check 1: Small dot marker
if "small, simple dot for easy visual tracking" in content:
    print("✓ FOUND: Small dot code (latest version)")
    has_small_dot = True
else:
    print("✗ MISSING: Small dot code")
    has_small_dot = False

# Check 2: Transform mode cycling
if "def robot_to_camera_coords" in content and "self.transform_mode" in content:
    print("✓ FOUND: Transform mode cycling (latest version)")
    has_transform_modes = True
else:
    print("✗ MISSING: Transform mode cycling")
    has_transform_modes = False

# Check 3: Small center marker
if 'cv2.circle(frame, center_pt, 8, (0, 255, 255), 2)' in content:
    print("✓ FOUND: Small center marker (latest version)")
    has_small_center = True
else:
    print("✗ MISSING: Small center marker")
    has_small_center = False

# Check 4: Old large markers (should NOT be present)
if 'cv2.circle(frame, pos_pt, 10, (0, 0, 255), -1)' in content:
    print("✗ FOUND OLD CODE: Large robot dot (10px) - OUTDATED VERSION!")
    has_old_code = True
else:
    print("✓ Good: No old large robot dot code")
    has_old_code = False

if 'cv2.circle(frame, center_pt, 30, (0, 255, 255), 3)' in content:
    print("✗ FOUND OLD CODE: Large center marker (30px) - OUTDATED VERSION!")
    has_old_center = True
else:
    print("✓ Good: No old large center marker code")
    has_old_center = False

print("\n" + "-"*70)
print("VERSION SUMMARY")
print("-"*70)

if has_small_dot and has_transform_modes and has_small_center and not has_old_code and not has_old_center:
    print("\n🎉 SUCCESS! You have the LATEST VERSION!")
    print("\nYou should see:")
    print("  • Small 5px red dot for robot position")
    print("  • Small 8px cyan circle for center marker")
    print("  • 'Transform Mode: X' at top of camera view")
    print("  • Press 'm' to cycle through modes")

elif has_old_code or has_old_center:
    print("\n⚠️  WARNING! You have an OUTDATED VERSION!")
    print("\nThis file still contains old code for large markers.")
    print("\nACTION REQUIRED:")
    print("1. Make sure you're in the correct directory")
    print("2. Run: git pull origin claude/robot-click-detection-011CUd1sBD9MMPs4dYBL7xoK")
    print("3. Verify with: git log --oneline -1")
    print("   Should show: 9f7afef Simplify robot position marker...")

else:
    print("\n⚠️  PARTIAL VERSION - Some features missing")
    print("\nMissing features:")
    if not has_small_dot:
        print("  ✗ Small dot code")
    if not has_transform_modes:
        print("  ✗ Transform mode cycling")
    if not has_small_center:
        print("  ✗ Small center marker")

print("\n" + "="*70)

# Additional checks
print("\nFILE DETAILS:")
print(f"  Total lines: {len(content.splitlines())}")
print(f"  Contains 'transform_mode': {('transform_mode' in content)}")
m_key_check = "key.lower() == 'm'" in content
print(f"  Contains 'm' key handler: {m_key_check}")

# Check for .pyc files
pyc_files = [f for f in os.listdir('.') if f.endswith('.pyc')]
if pyc_files:
    print(f"\n⚠️  WARNING: Found {len(pyc_files)} .pyc cache files!")
    print("  These might be causing Python to run old code.")
    print("  Run: find . -name '*.pyc' -delete")

print("\n" + "="*70 + "\n")
