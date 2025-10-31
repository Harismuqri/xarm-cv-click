"""
Coordinate Mapping Diagnostic Tool
Helps determine the exact coordinate transformation needed
"""

import json
import os
from xarm.wrapper import XArmAPI
import time

class CoordinateMappingTool:
    """Interactive tool to find correct coordinate mapping."""
    
    def __init__(self):
        self.config = self.load_config()
        self.robot_ip = self.config.get("robot_ip", "192.168.1.151")
        self._arm = None
        
        # Store test results
        self.test_results = {
            'initial_pos': None,
            'after_x_plus': None,
            'after_y_plus': None
        }
    
    def load_config(self, path="config.json"):
        """Load configuration."""
        try:
            if not os.path.isabs(path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(base_dir, path)
            with open(path, "r") as f:
                return json.load(f)
        except:
            return {"robot_ip": "192.168.1.151"}
    
    def connect_robot(self):
        """Connect to robot."""
        try:
            print(f"[Robot] Connecting to {self.robot_ip}...")
            self._arm = XArmAPI(self.robot_ip, baud_checkset=False)
            self._arm.clean_warn()
            self._arm.clean_error()
            self._arm.motion_enable(True)
            self._arm.set_mode(0)
            self._arm.set_state(0)
            time.sleep(0.5)
            print("[Robot] ✅ Connected\n")
            return True
        except Exception as e:
            print(f"[Robot] ❌ Failed: {e}")
            return False
    
    def get_position(self):
        """Get current robot position."""
        try:
            code, pos = self._arm.get_position()
            if code == 0:
                return {'x': pos[0], 'y': pos[1], 'z': pos[2]}
        except:
            pass
        return None
    
    def move_relative(self, dx=0, dy=0, dz=0):
        """Move robot relative to current position."""
        pos = self.get_position()
        if not pos:
            return False
        
        new_x = pos['x'] + dx
        new_y = pos['y'] + dy
        new_z = pos['z'] + dz
        
        try:
            code = self._arm.set_position(
                new_x, new_y, new_z, 180, 0, 0,
                speed=100, mvacc=1000, wait=True
            )
            return code == 0
        except:
            return False
    
    def run_diagnostic(self):
        """Run interactive diagnostic."""
        print("="*70)
        print("COORDINATE MAPPING DIAGNOSTIC")
        print("="*70)
        print("\nThis tool will help find the correct keyboard mapping.")
        print("\nThe test procedure:")
        print("  1. Record initial position")
        print("  2. Move robot X+ (forward in robot coordinates)")
        print("  3. You tell us which direction it moved on camera")
        print("  4. Move robot Y+ (right in robot coordinates)")
        print("  5. You tell us which direction it moved on camera")
        print("  6. Tool calculates correct mapping")
        print("\nPress ENTER to start...")
        input()
        
        # Step 1: Record initial position
        print("\n" + "-"*70)
        print("STEP 1: Recording initial position")
        print("-"*70)
        self.test_results['initial_pos'] = self.get_position()
        if not self.test_results['initial_pos']:
            print("❌ Cannot get robot position")
            return
        
        print(f"✅ Initial position: X={self.test_results['initial_pos']['x']:.1f}, "
              f"Y={self.test_results['initial_pos']['y']:.1f}, "
              f"Z={self.test_results['initial_pos']['z']:.1f}")
        
        # Step 2: Move X+
        print("\n" + "-"*70)
        print("STEP 2: Moving robot X+ by 20mm")
        print("-"*70)
        print("Watch your camera carefully!")
        input("Press ENTER to move robot X+ ...")
        
        if not self.move_relative(dx=20):
            print("❌ Movement failed")
            return
        
        time.sleep(1)
        self.test_results['after_x_plus'] = self.get_position()
        
        print(f"\n✅ Robot moved X+20mm")
        print(f"   New position: X={self.test_results['after_x_plus']['x']:.1f}, "
              f"Y={self.test_results['after_x_plus']['y']:.1f}")
        
        print("\n📹 QUESTION: Which direction did the robot move on your CAMERA view?")
        print("   1 = UP (toward top of screen)")
        print("   2 = RIGHT (toward right of screen)")
        print("   3 = DOWN (toward bottom of screen)")
        print("   4 = LEFT (toward left of screen)")
        
        x_direction = input("Enter 1, 2, 3, or 4: ").strip()
        direction_map = {'1': 'UP', '2': 'RIGHT', '3': 'DOWN', '4': 'LEFT'}
        x_visual = direction_map.get(x_direction, 'UNKNOWN')
        
        print(f"✅ Robot X+ moves visually {x_visual}")
        
        # Step 3: Return to start
        print("\n" + "-"*70)
        print("STEP 3: Returning to initial position")
        print("-"*70)
        input("Press ENTER to return...")
        
        if not self.move_relative(dx=-20):
            print("❌ Return failed")
            return
        
        time.sleep(1)
        print("✅ Returned to start")
        
        # Step 4: Move Y+
        print("\n" + "-"*70)
        print("STEP 4: Moving robot Y+ by 20mm")
        print("-"*70)
        print("Watch your camera carefully!")
        input("Press ENTER to move robot Y+ ...")
        
        if not self.move_relative(dy=20):
            print("❌ Movement failed")
            return
        
        time.sleep(1)
        self.test_results['after_y_plus'] = self.get_position()
        
        print(f"\n✅ Robot moved Y+20mm")
        print(f"   New position: X={self.test_results['after_y_plus']['x']:.1f}, "
              f"Y={self.test_results['after_y_plus']['y']:.1f}")
        
        print("\n📹 QUESTION: Which direction did the robot move on your CAMERA view?")
        print("   1 = UP (toward top of screen)")
        print("   2 = RIGHT (toward right of screen)")
        print("   3 = DOWN (toward bottom of screen)")
        print("   4 = LEFT (toward left of screen)")
        
        y_direction = input("Enter 1, 2, 3, or 4: ").strip()
        y_visual = direction_map.get(y_direction, 'UNKNOWN')
        
        print(f"✅ Robot Y+ moves visually {y_visual}")
        
        # Step 5: Return to start
        print("\n" + "-"*70)
        print("STEP 5: Returning to initial position")
        print("-"*70)
        input("Press ENTER to return...")
        
        if not self.move_relative(dy=-20):
            print("❌ Return failed")
            return
        
        time.sleep(1)
        print("✅ Returned to start")
        
        # Calculate mapping
        print("\n" + "="*70)
        print("DIAGNOSTIC RESULTS")
        print("="*70)
        print(f"\nRobot X+ moves visually: {x_visual}")
        print(f"Robot Y+ moves visually: {y_visual}")
        
        # Determine code mapping
        mapping = self.calculate_mapping(x_visual, y_visual)
        
        print("\n" + "="*70)
        print("RECOMMENDED CODE MAPPING")
        print("="*70)
        print("\nIn robot_keyboard_calibration.py, use this mapping:\n")
        print("```python")
        print("# Movement controls (CORRECTED MAPPING)")
        print("if key == 'up':")
        print(f"    {mapping['up']}")
        print()
        print("elif key == 'down':")
        print(f"    {mapping['down']}")
        print()
        print("elif key == 'left':")
        print(f"    {mapping['left']}")
        print()
        print("elif key == 'right':")
        print(f"    {mapping['right']}")
        print("```")
        print("\n" + "="*70)
        
        # Save to file
        with open("coordinate_mapping.txt", "w") as f:
            f.write("COORDINATE MAPPING RESULTS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Robot X+ moves visually: {x_visual}\n")
            f.write(f"Robot Y+ moves visually: {y_visual}\n\n")
            f.write("RECOMMENDED CODE:\n\n")
            f.write("if key == 'up':\n")
            f.write(f"    {mapping['up']}\n\n")
            f.write("elif key == 'down':\n")
            f.write(f"    {mapping['down']}\n\n")
            f.write("elif key == 'left':\n")
            f.write(f"    {mapping['left']}\n\n")
            f.write("elif key == 'right':\n")
            f.write(f"    {mapping['right']}\n")
        
        print("✅ Results saved to: coordinate_mapping.txt")
    
    def calculate_mapping(self, x_visual, y_visual):
        """Calculate the correct keyboard mapping based on visual directions."""
        mapping = {}
        
        # Map visual directions to robot axes
        # Robot X+ goes in x_visual direction
        # Robot Y+ goes in y_visual direction
        
        # For UP key (want to move visually UP)
        if x_visual == 'UP':
            mapping['up'] = "self.move_relative(dx=self.step_size)"
        elif x_visual == 'DOWN':
            mapping['up'] = "self.move_relative(dx=-self.step_size)"
        elif y_visual == 'UP':
            mapping['up'] = "self.move_relative(dy=self.step_size)"
        elif y_visual == 'DOWN':
            mapping['up'] = "self.move_relative(dy=-self.step_size)"
        
        # For DOWN key (want to move visually DOWN)
        if x_visual == 'DOWN':
            mapping['down'] = "self.move_relative(dx=self.step_size)"
        elif x_visual == 'UP':
            mapping['down'] = "self.move_relative(dx=-self.step_size)"
        elif y_visual == 'DOWN':
            mapping['down'] = "self.move_relative(dy=self.step_size)"
        elif y_visual == 'UP':
            mapping['down'] = "self.move_relative(dy=-self.step_size)"
        
        # For LEFT key (want to move visually LEFT)
        if x_visual == 'LEFT':
            mapping['left'] = "self.move_relative(dx=self.step_size)"
        elif x_visual == 'RIGHT':
            mapping['left'] = "self.move_relative(dx=-self.step_size)"
        elif y_visual == 'LEFT':
            mapping['left'] = "self.move_relative(dy=self.step_size)"
        elif y_visual == 'RIGHT':
            mapping['left'] = "self.move_relative(dy=-self.step_size)"
        
        # For RIGHT key (want to move visually RIGHT)
        if x_visual == 'RIGHT':
            mapping['right'] = "self.move_relative(dx=self.step_size)"
        elif x_visual == 'LEFT':
            mapping['right'] = "self.move_relative(dx=-self.step_size)"
        elif y_visual == 'RIGHT':
            mapping['right'] = "self.move_relative(dy=self.step_size)"
        elif y_visual == 'LEFT':
            mapping['right'] = "self.move_relative(dy=-self.step_size)"
        
        return mapping
    
    def cleanup(self):
        """Cleanup resources."""
        print("\n[Cleanup] Done")


def main():
    """Main function."""
    print("\n" + "="*70)
    print("COORDINATE MAPPING DIAGNOSTIC TOOL")
    print("="*70)
    print("\nThis tool will determine the correct keyboard-to-robot mapping")
    print("for your specific setup.")
    print("="*70 + "\n")
    
    tool = CoordinateMappingTool()
    
    try:
        if not tool.connect_robot():
            return
        
        tool.run_diagnostic()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        tool.cleanup()


if __name__ == "__main__":
    main()
