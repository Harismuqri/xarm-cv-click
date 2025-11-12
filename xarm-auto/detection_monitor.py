#!/usr/bin/env python3
"""
Lightweight Detection Display Monitor
Reads detection data from yolo-mouse shared memory and displays in GUI
No camera initialization, no YOLO processing - just displays data
"""

import tkinter as tk
from tkinter import ttk
from multiprocessing import shared_memory
import struct
import json
import time
import threading

# Shared memory configuration
DETECTION_SHM_NAME = "DetectionData"
DETECTION_SHM_SIZE = 4096

class DetectionDisplayMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Detection Monitor - Live Display")
        self.root.geometry("1000x600")
        
        self.shm = None
        self.running = False
        
        self.setup_ui()
        self.connect_shared_memory()
        self.start_updates()
        
    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#2c3e50", height=60)
        header.pack(fill=tk.X)
        tk.Label(header, text="🔍 Real-Time Detection Monitor", 
                font=("Arial", 18, "bold"), bg="#2c3e50", fg="white").pack(pady=15)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg="#34495e", height=40)
        status_frame.pack(fill=tk.X)
        
        self.status_label = tk.Label(status_frame, text="Status: Connecting...", 
                                     font=("Arial", 11), bg="#34495e", fg="white")
        self.status_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.fps_label = tk.Label(status_frame, text="Update: 0 Hz", 
                                  font=("Arial", 11), bg="#34495e", fg="white")
        self.fps_label.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Main content frame
        content = tk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Object count display
        count_frame = tk.LabelFrame(content, text="Detection Summary", 
                                    font=("Arial", 12, "bold"), padx=10, pady=10)
        count_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.object_count_label = tk.Label(count_frame, text="Objects Detected: 0", 
                                           font=("Arial", 16, "bold"), fg="#2c3e50")
        self.object_count_label.pack()
        
        # Objects table
        table_frame = tk.LabelFrame(content, text="Detected Objects", 
                                    font=("Arial", 12, "bold"))
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview
        columns = ("ID", "Position (mm)", "Angle (°)", "Size (mm)", "Aspect", "Orientation", "Robot Direction")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        widths = [50, 150, 80, 120, 80, 120, 150]
        for col, width in zip(columns, widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor=tk.CENTER)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
        # Configure row colors
        self.tree.tag_configure('vertical', background='#e3f2fd')
        self.tree.tag_configure('horizontal', background='#e8f5e9')
        self.tree.tag_configure('mixed', background='#fff9c4')
        
        # Control buttons
        button_frame = tk.Frame(self.root, bg="#ecf0f1")
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Button(button_frame, text="🔄 Refresh", command=self.manual_refresh,
                 bg="#3498db", fg="white", font=("Arial", 10, "bold"), 
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="🗑️ Clear", command=self.clear_display,
                 bg="#95a5a6", fg="white", font=("Arial", 10, "bold"), 
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="❌ Exit", command=self.on_closing,
                 bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), 
                 padx=20, pady=5).pack(side=tk.RIGHT, padx=5)
        
        # Instructions
        info_label = tk.Label(button_frame, 
                             text="📌 Make sure yolo-mouse-v2.py is running first!", 
                             font=("Arial", 9), bg="#ecf0f1", fg="#7f8c8d")
        info_label.pack(side=tk.LEFT, padx=20)
        
    def connect_shared_memory(self):
        """Connect to shared memory."""
        try:
            self.shm = shared_memory.SharedMemory(name=DETECTION_SHM_NAME)
            self.status_label.config(text="Status: Connected ✓", fg="#27ae60")
            print(f"[✓] Connected to {DETECTION_SHM_NAME}")
        except FileNotFoundError:
            self.status_label.config(text="Status: Waiting for yolo-mouse...", fg="#e67e22")
            print(f"[!] Waiting for {DETECTION_SHM_NAME}...")
            # Retry connection in background
            self.root.after(1000, self.connect_shared_memory)
        except Exception as e:
            self.status_label.config(text=f"Status: Error - {e}", fg="#c0392b")
            print(f"[✗] Connection error: {e}")
    
    def read_detection_data(self):
        """Read detection data from shared memory."""
        if not self.shm:
            return None
        
        try:
            # Read 4-byte length prefix
            length = struct.unpack('I', bytes(self.shm.buf[:4]))[0]
            
            if length == 0 or length > DETECTION_SHM_SIZE - 4:
                return None
            
            # Read JSON data
            json_bytes = bytes(self.shm.buf[4:4+length])
            json_str = json_bytes.decode('utf-8')
            
            data = json.loads(json_str)
            return data
        except Exception as e:
            return None
    
    def determine_orientation(self, width, height):
        """Determine object orientation based on dimensions."""
        if height == 0:
            return "UNKNOWN", "N/A", "mixed"
        
        aspect_ratio = width / height
        
        if aspect_ratio > 1.5:
            return "HORIZONTAL", "BACKWARD (180°)", "horizontal"
        elif aspect_ratio < 0.67:
            return "VERTICAL", "LEFT (0°)", "vertical"
        else:
            return "MIXED", "AUTO", "mixed"
    
    def update_display(self):
        """Update display with current detection data."""
        data = self.read_detection_data()
        
        if not data:
            return
        
        # Update object count
        objects = data.get("objects", {})
        count = len(objects)
        self.object_count_label.config(text=f"Objects Detected: {count}")
        
        # Clear existing items
        self.tree.delete(*self.tree.get_children())
        
        # Populate tree with objects
        for obj_id, obj in sorted(objects.items()):
            try:
                x = obj.get('x', 0)
                y = obj.get('y', 0)
                angle = obj.get('angle', 0)
                width = obj.get('width', 0)
                height = obj.get('height', 0)
                
                # Skip empty objects
                if x == 0 and y == 0:
                    continue
                
                # Determine orientation
                orientation, direction, tag = self.determine_orientation(width, height)
                aspect = width / height if height > 0 else 0
                
                # Insert into tree
                self.tree.insert("", tk.END, 
                               values=(
                                   obj_id,
                                   f"({x:.1f}, {y:.1f})",
                                   f"{angle:.1f}",
                                   f"{width:.1f} × {height:.1f}",
                                   f"{aspect:.2f}",
                                   orientation,
                                   direction
                               ),
                               tags=(tag,))
            except Exception as e:
                continue
    
    def manual_refresh(self):
        """Manual refresh button handler."""
        self.update_display()
    
    def clear_display(self):
        """Clear the display."""
        self.tree.delete(*self.tree.get_children())
        self.object_count_label.config(text="Objects Detected: 0")
    
    def start_updates(self):
        """Start automatic updates."""
        self.running = True
        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()
    
    def update_loop(self):
        """Background update loop."""
        last_update = time.time()
        update_count = 0
        
        while self.running:
            try:
                self.root.after(0, self.update_display)
                time.sleep(0.1)  # 10 Hz
                
                # Calculate FPS
                update_count += 1
                if time.time() - last_update >= 1.0:
                    fps = update_count / (time.time() - last_update)
                    self.fps_label.config(text=f"Update: {fps:.1f} Hz")
                    update_count = 0
                    last_update = time.time()
            except:
                break
    
    def on_closing(self):
        """Clean up on exit."""
        self.running = False
        if self.shm:
            try:
                self.shm.close()
                print("[✓] Disconnected from shared memory")
            except:
                pass
        self.root.destroy()

def main():
    print("\n" + "="*70)
    print("LIGHTWEIGHT DETECTION DISPLAY MONITOR")
    print("="*70)
    print("This monitor displays detection data from yolo-mouse-v2.py")
    print("No camera access, no YOLO processing - just displays data")
    print("="*70 + "\n")
    
    root = tk.Tk()
    app = DetectionDisplayMonitor(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()