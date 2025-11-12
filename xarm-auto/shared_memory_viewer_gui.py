#!/usr/bin/env python3
"""
All Shared Memory Viewer
Shows real-time data from DetectionData, ClickData, and InspectData
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from multiprocessing import shared_memory
import json
import time
import threading

# Shared memory configurations
SHARED_MEMORIES = {
    "DetectionData": 4096,
    "ClickData": 256,
    "InspectData": 256
}

class SharedMemoryViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("All Shared Memory Viewer")
        self.root.geometry("1000x800")
        
        self.shm_connections = {}
        self.running = False
        
        self.setup_ui()
        self.connect_all()
        self.start_updates()
        
    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#2c3e50", height=50)
        header.pack(fill=tk.X)
        tk.Label(header, text="🔍 All Shared Memory Viewer", 
                font=("Arial", 16, "bold"), bg="#2c3e50", fg="white").pack(pady=10)
        
        # Notebook (tabs) for each shared memory
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tab for each shared memory
        self.text_widgets = {}
        self.status_labels = {}
        
        for name in SHARED_MEMORIES.keys():
            tab = tk.Frame(self.notebook)
            self.notebook.add(tab, text=name)
            
            # Status label
            status_frame = tk.Frame(tab)
            status_frame.pack(fill=tk.X, padx=10, pady=5)
            
            tk.Label(status_frame, text="Status:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
            status_label = tk.Label(status_frame, text="Not Connected", fg="red", font=("Arial", 10))
            status_label.pack(side=tk.LEFT, padx=10)
            self.status_labels[name] = status_label
            
            # Text area for data display
            text_area = scrolledtext.ScrolledText(tab, font=("Courier", 10), wrap=tk.WORD)
            text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            self.text_widgets[name] = text_area
        
        # Control buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(button_frame, text="🔄 Refresh Now", command=self.manual_refresh,
                 bg="#3498db", fg="white", font=("Arial", 10), padx=20).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="❌ Exit", command=self.on_closing,
                 bg="#e74c3c", fg="white", font=("Arial", 10), padx=20).pack(side=tk.RIGHT, padx=5)
        
        # Auto-update status
        self.update_label = tk.Label(button_frame, text="Auto-update: ON (10 Hz)", 
                                     font=("Arial", 10), fg="green")
        self.update_label.pack(side=tk.LEFT, padx=20)
    
    def connect_all(self):
        """Connect to all shared memories."""
        for name, size in SHARED_MEMORIES.items():
            try:
                shm = shared_memory.SharedMemory(name=name)
                self.shm_connections[name] = shm
                self.status_labels[name].config(text="Connected ✓", fg="green")
                print(f"[✓] Connected to {name}")
            except FileNotFoundError:
                self.status_labels[name].config(text="Not Found ✗", fg="red")
                print(f"[✗] {name} not found")
            except Exception as e:
                self.status_labels[name].config(text=f"Error: {e}", fg="orange")
                print(f"[!] Error connecting to {name}: {e}")
    
    def read_shared_memory(self, name):
        """Read data from specific shared memory."""
        if name not in self.shm_connections:
            return None
        
        try:
            import struct
            shm = self.shm_connections[name]
            
            # Read 4-byte length prefix
            length = struct.unpack('I', bytes(shm.buf[:4]))[0]
            
            if length == 0 or length > SHARED_MEMORIES[name] - 4:
                return None
            
            # Read JSON data
            json_bytes = bytes(shm.buf[4:4+length])
            json_str = json_bytes.decode('utf-8')
            
            data = json.loads(json_str)
            return data
        except Exception as e:
            return {"error": str(e)}
    
    def update_display(self):
        """Update all displays with current data."""
        for name in SHARED_MEMORIES.keys():
            data = self.read_shared_memory(name)
            
            text_widget = self.text_widgets[name]
            text_widget.delete(1.0, tk.END)
            
            if data is None:
                text_widget.insert(1.0, "No data available or shared memory not connected")
            elif "error" in data:
                text_widget.insert(1.0, f"Error reading data: {data['error']}")
            else:
                # Format the JSON nicely
                formatted = json.dumps(data, indent=2)
                text_widget.insert(1.0, formatted)
                
                # Add summary at the top
                summary = f"=== {name} Summary ===\n"
                summary += f"Timestamp: {time.strftime('%H:%M:%S', time.localtime(data.get('timestamp', 0)))}\n"
                
                if name == "DetectionData" and 'objects' in data:
                    summary += f"Objects detected: {len(data['objects'])}\n"
                elif name == "ClickData":
                    summary += f"Button: {data.get('button', 'none')}\n"
                    summary += f"Position: ({data.get('click_x', 0):.1f}, {data.get('click_y', 0):.1f})\n"
                    summary += f"Angle: {data.get('angle', 0):.1f}°\n"
                elif name == "InspectData":
                    if data.get('inspect', False):
                        summary += f"Inspecting at: ({data.get('target_x', 0):.1f}, {data.get('target_y', 0):.1f})\n"
                        summary += f"Angle: {data.get('angle', 0):.1f}°\n"
                
                summary += "\n=== Raw JSON Data ===\n\n"
                text_widget.insert(1.0, summary)
    
    def manual_refresh(self):
        """Manual refresh button handler."""
        self.update_display()
    
    def start_updates(self):
        """Start automatic updates."""
        self.running = True
        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()
    
    def update_loop(self):
        """Background update loop."""
        while self.running:
            try:
                self.root.after(0, self.update_display)
                time.sleep(0.1)  # 10 Hz
            except:
                break
    
    def on_closing(self):
        """Clean up on exit."""
        self.running = False
        for name, shm in self.shm_connections.items():
            try:
                shm.close()
                print(f"[✓] Closed {name}")
            except:
                pass
        self.root.destroy()

def main():
    root = tk.Tk()
    app = SharedMemoryViewerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()