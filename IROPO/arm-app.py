"""
Robot Vision System - Setup & Monitoring GUI
PyQt6 application for calibration, dual camera visualization, and system monitoring
Communicates with xarm-motion via shared memory
"""

import sys
import cv2
import numpy as np
import pickle
import json
import os
from multiprocessing import shared_memory
import struct
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QTabWidget, QLabel, QLineEdit, 
                              QPushButton, QGroupBox, QGridLayout, QTextEdit,
                              QCheckBox, QComboBox, QMessageBox)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO
import PySpin


class ClickableLabel(QLabel):
    """Custom QLabel that emits click signals with scaled coordinates"""
    clicked = pyqtSignal(int, int)  # Signal emits x, y in original image coordinates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_image_size = None  # Store original image size for coordinate scaling
        self.displayed_image_size = None  # Store displayed (scaled) image size
        
    def mousePressEvent(self, event):
        """Handle mouse press events and emit scaled coordinates"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Get click position in widget coordinates
            click_pos = event.pos()
            widget_x = click_pos.x()
            widget_y = click_pos.y()
            
            # Scale coordinates to match original image size
            if self.original_image_size and self.displayed_image_size:
                # Calculate scaling factors
                scale_x = self.original_image_size[0] / self.displayed_image_size[0]
                scale_y = self.original_image_size[1] / self.displayed_image_size[1]
                
                # Scale click coordinates
                img_x = int(widget_x * scale_x)
                img_y = int(widget_y * scale_y)
                
                # Emit signal with scaled coordinates
                self.clicked.emit(img_x, img_y)
            else:
                # No scaling info, emit raw coordinates
                self.clicked.emit(widget_x, widget_y)


class RobotVisionGUI(QMainWindow):
    """Main GUI application for robot vision system"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Robot Positioning")
        self.setGeometry(100, 100, 900, 850)  # Horizontal layout: wide for 2 cameras side-by-side
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize variables
        self.detection_camera = None
        self.inspection_camera = None
        self.camera_system = None
        self.H_camera_to_workspace = None  # Camera → Workspace transformation
        self.H_workspace_to_robot = None   # Workspace → Robot transformation
        self.system_running = False
        
        # Object detection tracking
        self.detected_objects = []  # List of detected objects with their data
        self.selected_object = None  # Currently selected object
        self.mouse_click_x = 0  # Mouse click position in image coordinates
        self.mouse_click_y = 0
        self.show_info_panel = False  # Whether to show info panel
        
        # Shared memory
        self.click_shm = None
        self.inspect_shm = None
        
        # YOLO model
        model_path = self.config.get("yolo_model_path", "best.pt")
        self.model = YOLO(model_path)
        self.model.overrides['verbose'] = False
        
        # Setup UI
        self.init_ui()
        
        # Timer for camera updates
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_feeds)
        
        print("[GUI] Initialized")
    
    def load_config(self, path="config.json"):
        """Load configuration from JSON file"""
        try:
            if not os.path.isabs(path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(base_dir, path)
            
            with open(path, "r") as f:
                config = json.load(f)
                print(f"[Config] Loaded from {path}")
                return config
        except Exception as e:
            print(f"[Config] Failed to load: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            "yolo_model_path": "best.pt",
            "detection_confidence": 0.7,
            "camera_config": {
                "detection_camera_index": 0,
                "inspection_camera_index": 1
            },
            "workspace": {
                "width": 300,
                "height": 300
            }
        }
    
    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Add tabs
        tabs.addTab(self.create_calibration_tab(), "Calibration Setup")
        tabs.addTab(self.create_live_view_tab(), "Live Camera View")
        tabs.addTab(self.create_info_tab(), "System Info")
        
        # Add control panel at bottom
        main_layout.addWidget(self.create_control_panel())
        
        # Status bar
        self.statusBar().showMessage("Ready - Configure calibration and press START")
    
    def create_calibration_tab(self):
        """Create calibration configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # === Detection Camera Calibration ===
        detection_group = QGroupBox("Detection Camera Calibration (Camera → Workspace)")
        detection_layout = QVBoxLayout()
        
        # Auto calibration button
        auto_cal_btn = QPushButton("Auto Calibrate (Circle Detection)")
        auto_cal_btn.clicked.connect(self.auto_calibrate_detection)
        detection_layout.addWidget(auto_cal_btn)
        
        # Manual calibration option
        manual_detection_label = QLabel("Manual Backup (4 corners in pixels → workspace mm):")
        detection_layout.addWidget(manual_detection_label)
        
        self.detection_cal_inputs = {}
        corners = [
            ('BL', 'Bottom Left'),
            ('BR', 'Bottom Right'),
            ('TR', 'Top Right'),
            ('TL', 'Top Left')
        ]
        
        # Use grid layout for better alignment
        grid_layout = QGridLayout()
        grid_layout.setColumnStretch(1, 1)  # Pixel X
        grid_layout.setColumnStretch(2, 1)  # Pixel Y
        grid_layout.setColumnStretch(4, 1)  # Workspace X
        grid_layout.setColumnStretch(5, 1)  # Workspace Y
        
        for row, (corner_code, corner_name) in enumerate(corners):
            # Label
            grid_layout.addWidget(QLabel(f"{corner_name} ({corner_code}):"), row, 0)
            
            # Pixel inputs
            pixel_x = QLineEdit()
            pixel_x.setPlaceholderText("Pixel X")
            pixel_y = QLineEdit()
            pixel_y.setPlaceholderText("Pixel Y")
            grid_layout.addWidget(pixel_x, row, 1)
            grid_layout.addWidget(pixel_y, row, 2)
            
            # Arrow
            grid_layout.addWidget(QLabel("→"), row, 3)
            
            # Workspace inputs
            mm_x = QLineEdit()
            mm_x.setPlaceholderText("Workspace X (mm)")
            mm_y = QLineEdit()
            mm_y.setPlaceholderText("Workspace Y (mm)")
            grid_layout.addWidget(mm_x, row, 4)
            grid_layout.addWidget(mm_y, row, 5)
            
            self.detection_cal_inputs[corner_code] = {
                'pixel_x': pixel_x, 'pixel_y': pixel_y,
                'mm_x': mm_x, 'mm_y': mm_y
            }
        
        detection_layout.addLayout(grid_layout)
        
        manual_detection_btn = QPushButton("Apply Manual Detection Calibration")
        manual_detection_btn.clicked.connect(self.apply_manual_detection_calibration)
        detection_layout.addWidget(manual_detection_btn)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # === Robot Coordinate Transformation ===
        robot_group = QGroupBox("Robot Coordinate Transformation (Workspace → Robot)")
        robot_layout = QVBoxLayout()
        
        # Auto mode (2 points)
        auto_robot_label = QLabel("Auto Mode - Enter 2 corners (BL and TR):")
        robot_layout.addWidget(auto_robot_label)
        
        self.robot_auto_mode = QCheckBox("Use Auto Calculation (enter only BL and TR)")
        self.robot_auto_mode.setChecked(True)
        self.robot_auto_mode.stateChanged.connect(self.toggle_robot_calibration_mode)
        robot_layout.addWidget(self.robot_auto_mode)
        
        # Input fields for robot calibration
        self.robot_cal_inputs = {}
        robot_corners = [
            ('BL', 'Bottom Left'),
            ('BR', 'Bottom Right'),
            ('TR', 'Top Right'),
            ('TL', 'Top Left')
        ]
        
        # Use grid layout for better alignment
        robot_grid_layout = QGridLayout()
        robot_grid_layout.setColumnStretch(1, 1)  # Workspace X
        robot_grid_layout.setColumnStretch(2, 1)  # Workspace Y
        robot_grid_layout.setColumnStretch(4, 1)  # Robot X
        robot_grid_layout.setColumnStretch(5, 1)  # Robot Y
        
        for row, (corner_code, corner_name) in enumerate(robot_corners):
            # Label
            robot_grid_layout.addWidget(QLabel(f"{corner_name} ({corner_code}):"), row, 0)
            
            # Workspace inputs
            ws_x = QLineEdit()
            ws_x.setPlaceholderText("Workspace X (mm)")
            ws_y = QLineEdit()
            ws_y.setPlaceholderText("Workspace Y (mm)")
            robot_grid_layout.addWidget(ws_x, row, 1)
            robot_grid_layout.addWidget(ws_y, row, 2)
            
            # Arrow
            robot_grid_layout.addWidget(QLabel("→"), row, 3)
            
            # Robot inputs
            robot_x = QLineEdit()
            robot_x.setPlaceholderText("Robot X (mm)")
            robot_y = QLineEdit()
            robot_y.setPlaceholderText("Robot Y (mm)")
            robot_grid_layout.addWidget(robot_x, row, 4)
            robot_grid_layout.addWidget(robot_y, row, 5)
            
            self.robot_cal_inputs[corner_code] = {
                'ws_x': ws_x, 'ws_y': ws_y,
                'robot_x': robot_x, 'robot_y': robot_y
            }
            
            # Disable BR and TL in auto mode initially
            if corner_code in ['BR', 'TL']:
                ws_x.setEnabled(False)
                ws_y.setEnabled(False)
                robot_x.setEnabled(False)
                robot_y.setEnabled(False)
        
        robot_layout.addLayout(robot_grid_layout)
        
        # Pre-fill workspace coordinates
        self.robot_cal_inputs['BL']['ws_x'].setText("0")
        self.robot_cal_inputs['BL']['ws_y'].setText("0")
        self.robot_cal_inputs['TR']['ws_x'].setText("300")
        self.robot_cal_inputs['TR']['ws_y'].setText("300")
        
        calculate_btn = QPushButton("Calculate Robot Transformation")
        calculate_btn.clicked.connect(self.calculate_robot_transformation)
        robot_layout.addWidget(calculate_btn)
        
        robot_group.setLayout(robot_layout)
        layout.addWidget(robot_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return widget
    
    def create_live_view_tab(self):
        """Create live camera view tab"""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        
        # Create horizontal layout for side-by-side cameras
        camera_layout = QHBoxLayout()
        
        # Detection camera (left side)
        detection_container = QVBoxLayout()
        detection_title = QLabel("Detection Camera")
        detection_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detection_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        detection_container.addWidget(detection_title)
        
        self.detection_label = ClickableLabel()  # Use ClickableLabel for click detection
        self.detection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detection_label.setMinimumSize(640, 480)  # Minimum size
        self.detection_label.setStyleSheet("border: 2px solid black; background-color: #2b2b2b;")
        self.detection_label.clicked.connect(self.on_detection_click)  # Connect click handler
        detection_container.addWidget(self.detection_label)
        
        # Inspection camera (right side)
        inspection_container = QVBoxLayout()
        inspection_title = QLabel("Inspection Camera")
        inspection_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inspection_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        inspection_container.addWidget(inspection_title)
        
        self.inspection_label = QLabel()
        self.inspection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inspection_label.setMinimumSize(640, 480)  # Minimum size
        self.inspection_label.setStyleSheet("border: 2px solid black; background-color: #2b2b2b;")
        inspection_container.addWidget(self.inspection_label)
        
        # Add both cameras to horizontal layout
        camera_layout.addLayout(detection_container)
        camera_layout.addLayout(inspection_container)
        
        # Add camera layout to main layout
        main_layout.addLayout(camera_layout)
        
        # Detection info display at bottom
        self.detection_info = QTextEdit()
        self.detection_info.setReadOnly(True)
        self.detection_info.setMaximumHeight(100)
        main_layout.addWidget(self.detection_info)
        
        return widget
    
    def create_info_tab(self):
        """Create system information tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFont(QFont("Courier", 10))
        layout.addWidget(self.info_text)
        
        self.update_info_display()
        
        return widget
    
    def create_control_panel(self):
        """Create control panel with START/STOP buttons"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        self.start_btn = QPushButton("START System")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; padding: 10px;")
        self.start_btn.clicked.connect(self.start_system)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("STOP System")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-size: 16px; padding: 10px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_system)
        layout.addWidget(self.stop_btn)
        
        return widget
    
    def toggle_robot_calibration_mode(self, state):
        """Toggle between auto and manual robot calibration mode"""
        auto_mode = (state == Qt.CheckState.Checked.value)
        
        # Enable/disable BR and TL inputs
        for corner in ['BR', 'TL']:
            for key in ['ws_x', 'ws_y', 'robot_x', 'robot_y']:
                self.robot_cal_inputs[corner][key].setEnabled(not auto_mode)
    
    def auto_calibrate_detection(self):
        """Auto calibrate detection camera using circle detection"""
        try:
            self.statusBar().showMessage("Starting auto calibration...")
            
            # Initialize camera if not already done
            if self.detection_camera is None:
                if not self.initialize_cameras():
                    QMessageBox.warning(self, "Error", "Failed to initialize camera")
                    return
            
            # Get frame
            frame = self.get_detection_frame()
            if frame is None:
                QMessageBox.warning(self, "Error", "Failed to capture frame")
                return
            
            # Perform auto calibration (circle detection)
            H = self.perform_circle_calibration(frame)
            
            if H is not None:
                self.H_camera_to_workspace = H
                
                # Save homography
                script_dir = os.path.dirname(os.path.abspath(__file__))
                homography_file = os.path.join(script_dir, "homography_auto.pkl")
                with open(homography_file, "wb") as f:
                    pickle.dump(H, f)
                
                QMessageBox.information(self, "Success", "Auto calibration completed!")
                self.statusBar().showMessage("Auto calibration successful")
                self.update_info_display()
            else:
                QMessageBox.warning(self, "Failed", "Could not detect calibration circles")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Calibration failed: {str(e)}")
            print(f"[Error] Auto calibration: {e}")
    
    def perform_circle_calibration(self, frame):
        """Perform circle-based calibration (from your existing code)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=30
        )
        
        if circles is None:
            return None
        
        filtered = []
        for (x, y, r) in np.round(circles[0]).astype("int"):
            if 10 <= r <= 50:
                filtered.append((x, y, r))
        
        if len(filtered) < 4:
            return None
        
        # Get 4 corners
        image_points = np.array([[x, y] for (x, y, _) in filtered[:4]], dtype=np.float32)
        image_points = sorted(image_points, key=lambda pt: (pt[1], pt[0]))
        top = sorted(image_points[:2], key=lambda pt: pt[0])
        bottom = sorted(image_points[2:], key=lambda pt: pt[0])
        sorted_img_pts = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
        
        # Real world points (workspace)
        workspace_width = self.config.get("workspace", {}).get("width", 300)
        workspace_height = self.config.get("workspace", {}).get("height", 300)
        
        real_pts = np.array([
            [0, workspace_height],
            [workspace_width, workspace_height],
            [workspace_width, 0],
            [0, 0]
        ], dtype=np.float32)
        
        H, _ = cv2.findHomography(sorted_img_pts, real_pts)
        return H
    
    def apply_manual_detection_calibration(self):
        """Apply manual detection camera calibration"""
        try:
            # Collect input points
            image_points = []
            real_points = []
            
            corners = ['BL', 'BR', 'TR', 'TL']
            for corner in corners:
                inputs = self.detection_cal_inputs[corner]
                
                try:
                    px = float(inputs['pixel_x'].text())
                    py = float(inputs['pixel_y'].text())
                    mx = float(inputs['mm_x'].text())
                    my = float(inputs['mm_y'].text())
                    
                    image_points.append([px, py])
                    real_points.append([mx, my])
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", f"Please fill all fields for {corner}")
                    return
            
            # Calculate homography
            image_points = np.array(image_points, dtype=np.float32)
            real_points = np.array(real_points, dtype=np.float32)
            
            H, _ = cv2.findHomography(image_points, real_points)
            
            if H is not None:
                self.H_camera_to_workspace = H
                
                # Save homography
                script_dir = os.path.dirname(os.path.abspath(__file__))
                homography_file = os.path.join(script_dir, "homography_auto.pkl")
                with open(homography_file, "wb") as f:
                    pickle.dump(H, f)
                
                QMessageBox.information(self, "Success", "Manual calibration applied!")
                self.update_info_display()
            else:
                QMessageBox.warning(self, "Failed", "Could not calculate homography")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Manual calibration failed: {str(e)}")
    
    def calculate_robot_transformation(self):
        """Calculate workspace to robot coordinate transformation"""
        try:
            auto_mode = self.robot_auto_mode.isChecked()
            
            # Collect points
            workspace_points = []
            robot_points = []
            
            if auto_mode:
                # Get BL and TR only
                corners_to_use = ['BL', 'TR']
                
                for corner in corners_to_use:
                    inputs = self.robot_cal_inputs[corner]
                    try:
                        ws_x = float(inputs['ws_x'].text())
                        ws_y = float(inputs['ws_y'].text())
                        robot_x = float(inputs['robot_x'].text())
                        robot_y = float(inputs['robot_y'].text())
                        
                        workspace_points.append([ws_x, ws_y])
                        robot_points.append([robot_x, robot_y])
                    except ValueError:
                        QMessageBox.warning(self, "Invalid Input", f"Please fill all fields for {corner}")
                        return
                
                # Auto-calculate BR and TL using Option 2 (rotation-aware)
                bl_ws = np.array(workspace_points[0])
                tr_ws = np.array(workspace_points[1])
                bl_robot = np.array(robot_points[0])
                tr_robot = np.array(robot_points[1])
                
                # Calculate vectors
                vector_diag_ws = tr_ws - bl_ws  # Diagonal in workspace
                vector_diag_robot = tr_robot - bl_robot  # Diagonal in robot
                
                # Since it's a square, the other diagonal should be perpendicular and equal length
                # Rotate the diagonal vector by 90 degrees
                # For a 90° rotation: (x, y) → (-y, x)
                vector_perp_robot = np.array([-vector_diag_robot[1], vector_diag_robot[0]])
                
                # Calculate BR and TL
                # BR is at BL + right vector
                # In workspace: BR = (300, 0) = BL + (300, 0)
                # The workspace right vector is (300, 0)
                workspace_right = np.array([300, 0]) - bl_ws
                
                # Scale the perpendicular vector
                scale = np.linalg.norm(workspace_right) / np.linalg.norm(vector_diag_ws)
                
                # BR in robot coords
                br_robot = bl_robot + (tr_robot - bl_robot) * np.array([1, 0]) / np.linalg.norm(tr_ws - bl_ws) * workspace_right[0]
                
                # Actually, let's use a simpler geometric approach
                # The transformation is affine, so we can interpolate
                
                # For a square workspace (0,0)→(300,300), we know:
                # BL = (0, 0), BR = (300, 0), TR = (300, 300), TL = (0, 300)
                
                # Calculate BR: same Y as BL, same X as TR
                br_ws = np.array([300, 0])
                br_robot_calc = bl_robot + (tr_robot - bl_robot) * (br_ws - bl_ws) / (tr_ws - bl_ws)
                
                # Better approach: preserve the square geometry
                # Vector from BL to BR in workspace: (300, 0)
                # Vector from BL to TL in workspace: (0, 300)
                
                # Find how these vectors map to robot space
                # We have BL→TR mapped, which is (300, 300) → (tr_robot - bl_robot)
                
                # For a proper square transformation with rotation:
                # Split the diagonal into two perpendicular sides
                diag_vector = tr_robot - bl_robot
                half_diag = diag_vector / 2
                
                # Rotate 90 degrees for perpendicular
                perp_vector = np.array([-half_diag[1], half_diag[0]])
                
                # Calculate all corners from center
                center_ws = (bl_ws + tr_ws) / 2
                center_robot = (bl_robot + tr_robot) / 2
                
                br_robot = center_robot + half_diag - perp_vector
                tl_robot = center_robot - half_diag + perp_vector
                
                # Add to lists
                workspace_points.append([300, 0])  # BR workspace
                robot_points.append(br_robot.tolist())
                
                workspace_points.append([0, 300])  # TL workspace  
                robot_points.append(tl_robot.tolist())
                
                # Update display fields
                self.robot_cal_inputs['BR']['robot_x'].setText(f"{br_robot[0]:.1f}")
                self.robot_cal_inputs['BR']['robot_y'].setText(f"{br_robot[1]:.1f}")
                self.robot_cal_inputs['TL']['robot_x'].setText(f"{tl_robot[0]:.1f}")
                self.robot_cal_inputs['TL']['robot_y'].setText(f"{tl_robot[1]:.1f}")
                
            else:
                # Manual mode - use all 4 points
                corners = ['BL', 'BR', 'TR', 'TL']
                for corner in corners:
                    inputs = self.robot_cal_inputs[corner]
                    try:
                        ws_x = float(inputs['ws_x'].text())
                        ws_y = float(inputs['ws_y'].text())
                        robot_x = float(inputs['robot_x'].text())
                        robot_y = float(inputs['robot_y'].text())
                        
                        workspace_points.append([ws_x, ws_y])
                        robot_points.append([robot_x, robot_y])
                    except ValueError:
                        QMessageBox.warning(self, "Invalid Input", f"Please fill all fields for {corner}")
                        return
            
            # Calculate homography
            workspace_points = np.array(workspace_points, dtype=np.float32)
            robot_points = np.array(robot_points, dtype=np.float32)
            
            H, _ = cv2.findHomography(workspace_points, robot_points)
            
            if H is not None:
                self.H_workspace_to_robot = H
                
                # Save homography
                script_dir = os.path.dirname(os.path.abspath(__file__))
                homography_file = os.path.join(script_dir, "homography_det_to_robot.pkl")
                with open(homography_file, "wb") as f:
                    pickle.dump(H, f)
                
                QMessageBox.information(self, "Success", "Robot transformation calculated!")
                self.update_info_display()
            else:
                QMessageBox.warning(self, "Failed", "Could not calculate transformation")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Transformation calculation failed: {str(e)}")
            print(f"[Error] Robot transformation: {e}")
    
    def initialize_cameras(self):
        """Initialize both cameras"""
        try:
            print("[Camera] Initializing...")
            self.camera_system = PySpin.System.GetInstance()
            cam_list = self.camera_system.GetCameras()
            
            if cam_list.GetSize() < 1:
                print("[Camera] No cameras found")
                return False
            
            # Initialize detection camera
            self.detection_camera = cam_list.GetByIndex(0)
            self.detection_camera.Init()
            self.detection_camera.BeginAcquisition()
            print("[Camera] Detection camera initialized")
            
            # Initialize inspection camera if available
            if cam_list.GetSize() >= 2:
                self.inspection_camera = cam_list.GetByIndex(1)
                self.inspection_camera.Init()
                self.inspection_camera.BeginAcquisition()
                print("[Camera] Inspection camera initialized")
            
            return True
            
        except Exception as e:
            print(f"[Camera] Initialization error: {e}")
            return False
    
    def get_detection_frame(self):
        """Get frame from detection camera"""
        if self.detection_camera is None:
            return None
        
        try:
            image = self.detection_camera.GetNextImage()
            if image.IsIncomplete():
                image.Release()
                return None
            
            img_array = image.GetNDArray()
            image.Release()
            
            if len(img_array.shape) == 2:
                return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            return img_array
        except:
            return None
    
    def get_inspection_frame(self):
        """Get frame from inspection camera"""
        if self.inspection_camera is None:
            return None
        
        try:
            image = self.inspection_camera.GetNextImage()
            if image.IsIncomplete():
                image.Release()
                return None
            
            img_array = image.GetNDArray()
            image.Release()
            
            if len(img_array.shape) == 2:
                return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            return img_array
        except:
            return None
    
    def transform_points(self, points, H):
        """Transform points using homography matrix"""
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, H)
        return warped.reshape(-1, 2)
    
    def is_inside_workspace(self, pts, width=None, height=None):
        """Check if points are inside workspace boundaries"""
        if width is None:
            width = self.config.get("workspace", {}).get("width", 300)
        if height is None:
            height = self.config.get("workspace", {}).get("height", 300)
        
        x, y = pts[:, 0], pts[:, 1]
        return np.all((x >= 0) & (x <= width) & (y >= 0) & (y <= height))
    
    def get_angle(self, obb_pts):
        """Calculate angle from OBB points"""
        v1 = obb_pts[1] - obb_pts[0]
        v2 = obb_pts[2] - obb_pts[1]
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        long_vec = v1 if len1 >= len2 else v2
        angle_rad = np.arctan2(long_vec[1], long_vec[0])
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 180
        return angle_deg
    
    def update_camera_feeds(self):
        """Update camera display (called by timer)"""
        # Get detection frame
        detection_frame = self.get_detection_frame()
        if detection_frame is not None:
            # Run YOLO detection
            results = self.model(detection_frame, conf=self.config.get("detection_confidence", 0.7))
            
            # Draw detections and update display
            annotated = self.draw_detections(detection_frame, results)
            self.display_frame(annotated, self.detection_label)
        
        # Get inspection frame
        inspection_frame = self.get_inspection_frame()
        if inspection_frame is not None:
            # Draw crosshair on inspection camera (like yolo-mouse-v2.py)
            camera_offset_config = self.config.get("camera_offset", {})
            offset_error = camera_offset_config.get("offset_error", 0)
            
            inspection_frame = self.draw_inspection_crosshair(inspection_frame.copy(), offset_error)
            
            # Draw info overlay
            overlay = inspection_frame.copy()
            cv2.rectangle(overlay, (10, 10), (500, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, inspection_frame, 0.4, 0, inspection_frame)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            offset_x = camera_offset_config.get("offset_x", 7)
            offset_y = camera_offset_config.get("offset_y", 92.9)
            
            info_lines = [
                "INSPECTION CAMERA",
                f"Offset: ({offset_x:.1f}, {offset_y:.1f}) mm",
                f"Error: {offset_error:.1f} mm"
            ]
            
            y_offset = 50
            for line in info_lines:
                cv2.putText(inspection_frame, line, (25, y_offset), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
                y_offset += 40
            
            self.display_frame(inspection_frame, self.inspection_label)
    
    def draw_detections(self, frame, results):
        """Draw YOLO detections on frame with color based on workspace position"""
        annotated = frame.copy()
        
        # Clear previous detected objects
        self.detected_objects = []
        
        # Draw workspace boundary if calibrated
        if self.H_camera_to_workspace is not None:
            H_inv = np.linalg.inv(self.H_camera_to_workspace)
            self.draw_workspace_boundary(annotated, H_inv)
        
        # Draw detection results
        obb_preds = results[0].obb
        
        for i, obb in enumerate(obb_preds, 1):
            if hasattr(obb, "xyxyxyxy"):
                corners = obb.xyxyxyxy.cpu().numpy().reshape(-1, 2)
            elif hasattr(obb, "xyxy"):
                corners = obb.xyxy.cpu().numpy().reshape(-1, 2)
            else:
                continue
            
            # Determine color based on workspace position
            if self.H_camera_to_workspace is not None:
                # Transform corners to workspace coordinates
                transformed = self.transform_points(corners, self.H_camera_to_workspace)
                is_inside = self.is_inside_workspace(transformed)
                
                if is_inside:
                    color = (0, 255, 0)  # Green - inside workspace
                    
                    # Calculate object properties
                    center = np.mean(transformed, axis=0)
                    angle = self.get_angle(transformed)
                    
                    side1 = np.linalg.norm(transformed[1] - transformed[0])
                    side2 = np.linalg.norm(transformed[2] - transformed[1])
                    width_mm = round(max(side1, side2), 1)
                    height_mm = round(min(side1, side2), 1)
                    
                    x_mm, y_mm = round(center[0], 1), round(center[1], 1)
                    angle_deg = round(angle, 2)
                    
                    # Store object data for click detection
                    self.detected_objects.append({
                        'id': i,
                        'corners': corners.astype(int),
                        'x_mm': x_mm,
                        'y_mm': y_mm,
                        'angle': angle_deg,
                        'width': width_mm,
                        'height': height_mm
                    })
                    
                    # Draw center point
                    center_img = np.mean(corners, axis=0).astype(int)
                    cv2.circle(annotated, tuple(center_img), 5, (0, 0, 255), -1)
                    cv2.circle(annotated, tuple(center_img), 5, (255, 255, 255), 1)
                    cv2.drawMarker(annotated, tuple(center_img), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
                else:
                    color = (0, 0, 255)  # Red - outside workspace
            else:
                color = (128, 128, 128)  # Gray - no calibration
            
            # Draw border
            corners_int = corners.astype(int)
            cv2.polylines(annotated, [corners_int], isClosed=True, color=color, thickness=2)
            
            # Draw object label (like yolo-mouse-v2.py)
            label_pos = tuple(corners_int[0] - [0, 10])
            cv2.putText(annotated, f"Object {i}", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        # Draw info panel if object is selected
        if self.show_info_panel and self.selected_object:
            # Highlight selected object with cyan border
            cv2.polylines(annotated, [self.selected_object['corners']], 
                         isClosed=True, color=(255, 255, 0), thickness=3)
            
            # Draw info panel near click position
            info_lines = [
                f"Object ID: {self.selected_object['id']}",
                f"Position: ({self.selected_object['x_mm']:.1f}, {self.selected_object['y_mm']:.1f}) mm",
                f"Angle: {self.selected_object['angle']:.1f} degrees",
                f"Width: {self.selected_object['width']:.1f} mm",
                f"Height: {self.selected_object['height']:.1f} mm"
            ]
            
            annotated = self.draw_info_panel_on_frame(
                annotated, 
                self.mouse_click_x + 10, 
                self.mouse_click_y + 10, 
                info_lines, 
                f"Object {self.selected_object['id']}"
            )
        
        return annotated
    
    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon"""
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def draw_info_panel_on_frame(self, frame, x, y, info_lines, title="Info"):
        """Draw an information panel at specified position (like yolo-mouse-v2.py)"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        padding = 10
        line_height = 20

        max_width = 0
        for line in info_lines:
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, w)

        panel_width = max_width + 2 * padding
        panel_height = len(info_lines) * line_height + 2 * padding + 25

        # Adjust position if panel goes off screen
        if x + panel_width > frame.shape[1]:
            x = frame.shape[1] - panel_width - 10
        if y + panel_height > frame.shape[0]:
            y = frame.shape[0] - panel_height - 10

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw border
        cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), (0, 255, 255), 2)

        # Draw title
        cv2.rectangle(frame, (x, y), (x + panel_width, y + 25), (0, 255, 255), -1)
        cv2.putText(frame, title, (x + padding, y + 18), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw info lines
        y_offset = y + 40
        for line in info_lines:
            cv2.putText(frame, line, (x + padding, y_offset), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y_offset += line_height

        return frame
    
    def draw_inspection_crosshair(self, frame, offset_error=0):
        """Draw inspection crosshair with error tolerance circle (like yolo-mouse-v2.py)"""
        h, w = frame.shape[:2]
        x, y = w // 2, h // 2  # Center of frame
        
        # Red crosshair
        line_length = 40
        thickness = 3
        color = (0, 0, 255)
        
        cv2.line(frame, (x - line_length, y), (x + line_length, y), color, thickness)
        cv2.line(frame, (x, y - line_length), (x, y + line_length), color, thickness)
        
        # Green center circle
        cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
        
        # Error tolerance circle (yellow, dashed appearance)
        error_radius_px = 15
        for angle in range(0, 360, 30):
            angle_rad = np.radians(angle)
            x1 = int(x + error_radius_px * np.cos(angle_rad))
            y1 = int(y + error_radius_px * np.sin(angle_rad))
            x2 = int(x + error_radius_px * np.cos(angle_rad + np.radians(15)))
            y2 = int(y + error_radius_px * np.sin(angle_rad + np.radians(15)))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
        
        return frame
    
    def on_detection_click(self, x, y):
        """Handle click on detection camera - find which object was clicked"""
        self.mouse_click_x = x
        self.mouse_click_y = y
        
        print(f"[DEBUG] Click at pixel ({x}, {y})")
        print(f"[DEBUG] Number of detected objects: {len(self.detected_objects)}")
        
        # Check if clicked on any detected object
        clicked_on_object = False
        for obj_data in self.detected_objects:
            # Debug: print object corners
            print(f"[DEBUG] Object {obj_data['id']} corners: {obj_data['corners']}")
            
            # Test if point is in polygon
            result = self.point_in_polygon((x, y), obj_data['corners'])
            print(f"[DEBUG] Object {obj_data['id']} point_in_polygon result: {result}")
            
            if result:
                self.selected_object = obj_data
                self.show_info_panel = True
                clicked_on_object = True
                print(f"[GUI CLICK] Object {obj_data['id']}: ({obj_data['x_mm']:.1f}, {obj_data['y_mm']:.1f}) mm")
                
                # Update detection info text
                info_text = f"Selected Object {obj_data['id']}:\n"
                info_text += f"Position: ({obj_data['x_mm']:.1f}, {obj_data['y_mm']:.1f}) mm\n"
                info_text += f"Angle: {obj_data['angle']:.1f}°, "
                info_text += f"Size: {obj_data['width']:.1f}x{obj_data['height']:.1f} mm"
                self.detection_info.setText(info_text)
                break
        
        # If clicked on empty space
        if not clicked_on_object:
            self.selected_object = None
            self.show_info_panel = False
            self.detection_info.clear()
            print(f"[GUI CLICK] Empty space at pixel ({x}, {y})")
    
    def draw_workspace_boundary(self, frame, H_inv):
        """Draw workspace boundary box"""
        workspace_width = self.config.get("workspace", {}).get("width", 300)
        workspace_height = self.config.get("workspace", {}).get("height", 300)
        
        box_real = np.array([
            [0, 0],
            [workspace_width, 0],
            [workspace_width, workspace_height],
            [0, workspace_height]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        box_img = cv2.perspectiveTransform(box_real, H_inv).reshape(-1, 2).astype(int)
        cv2.polylines(frame, [box_img], isClosed=True, color=(255, 255, 255), thickness=3)
    
    def display_frame(self, frame, label):
        """Display OpenCV frame in QLabel"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Store original image size for click coordinate scaling
        h, w, ch = rgb_frame.shape
        if isinstance(label, ClickableLabel):
            label.original_image_size = (w, h)
        
        # Resize to fit label
        label_w = label.width()
        label_h = label.height()
        
        # Calculate scaling
        scale = min(label_w / w, label_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Store displayed image size for click coordinate scaling
        if isinstance(label, ClickableLabel):
            label.displayed_image_size = (new_w, new_h)
        
        resized = cv2.resize(rgb_frame, (new_w, new_h))
        
        # Convert to QImage
        bytes_per_line = ch * new_w
        qt_image = QImage(resized.data, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Display
        label.setPixmap(QPixmap.fromImage(qt_image))
    
    def update_info_display(self):
        """Update system information display"""
        info = "=== System Status ===\n\n"
        
        info += f"Detection Camera: {'✓ Initialized' if self.detection_camera else '✗ Not initialized'}\n"
        info += f"Inspection Camera: {'✓ Initialized' if self.inspection_camera else '✗ Not initialized'}\n\n"
        
        info += f"Camera → Workspace Calibration: {'✓ Loaded' if self.H_camera_to_workspace is not None else '✗ Not calibrated'}\n"
        info += f"Workspace → Robot Calibration: {'✓ Loaded' if self.H_workspace_to_robot is not None else '✗ Not calibrated'}\n\n"
        
        info += f"System Status: {'🟢 RUNNING' if self.system_running else '🔴 STOPPED'}\n"
        
        self.info_text.setText(info)
    
    def start_system(self):
        """Start the vision system"""
        try:
            # Check calibrations
            if self.H_camera_to_workspace is None:
                QMessageBox.warning(self, "Not Ready", "Please calibrate detection camera first!")
                return
            
            if self.H_workspace_to_robot is None:
                QMessageBox.warning(self, "Not Ready", "Please configure robot transformation first!")
                return
            
            # Initialize cameras
            if self.detection_camera is None:
                if not self.initialize_cameras():
                    QMessageBox.critical(self, "Error", "Failed to initialize cameras")
                    return
            
            # Initialize shared memory
            self.initialize_shared_memory()
            
            # Start camera timer
            self.camera_timer.start(33)  # ~30 FPS
            
            # Update UI
            self.system_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.statusBar().showMessage("System RUNNING")
            self.update_info_display()
            
            print("[GUI] System started")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start system: {str(e)}")
            print(f"[Error] Start system: {e}")
    
    def stop_system(self):
        """Stop the vision system"""
        try:
            # Stop camera timer
            self.camera_timer.stop()
            
            # Cleanup shared memory
            self.cleanup_shared_memory()
            
            # Update UI
            self.system_running = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.statusBar().showMessage("System STOPPED")
            self.update_info_display()
            
            print("[GUI] System stopped")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to stop system: {str(e)}")
    
    def initialize_shared_memory(self):
        """Initialize shared memory for communication with xarm-motion"""
        try:
            # Click data shared memory
            try:
                self.click_shm = shared_memory.SharedMemory(name="ClickData", create=True, size=512)
            except FileExistsError:
                self.click_shm = shared_memory.SharedMemory(name="ClickData")
            
            # Inspect data shared memory
            try:
                self.inspect_shm = shared_memory.SharedMemory(name="InspectData", create=True, size=512)
            except FileExistsError:
                self.inspect_shm = shared_memory.SharedMemory(name="InspectData")
            
            print("[SharedMemory] Initialized")
            
        except Exception as e:
            print(f"[SharedMemory] Error: {e}")
    
    def cleanup_shared_memory(self):
        """Cleanup shared memory"""
        try:
            if self.click_shm:
                self.click_shm.close()
            if self.inspect_shm:
                self.inspect_shm.close()
            print("[SharedMemory] Cleaned up")
        except:
            pass
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop system if running
        if self.system_running:
            self.stop_system()
        
        # Cleanup cameras
        if self.detection_camera:
            try:
                self.detection_camera.EndAcquisition()
                self.detection_camera.DeInit()
            except:
                pass
        
        if self.inspection_camera:
            try:
                self.inspection_camera.EndAcquisition()
                self.inspection_camera.DeInit()
            except:
                pass
        
        if self.camera_system:
            try:
                cam_list = self.camera_system.GetCameras()
                cam_list.Clear()
                self.camera_system.ReleaseInstance()
            except:
                pass
        
        event.accept()
    
    def changeEvent(self, event):
        """Handle window state changes (maximize/restore)"""
        if event.type() == event.Type.WindowStateChange:
            if self.windowState() & Qt.WindowState.WindowMaximized:
                # When window is MAXIMIZED (shows restore icon ⧉) - use BIGGER size (850x720)
                self.detection_label.setFixedSize(850, 720)
                self.inspection_label.setFixedSize(850, 720)
            else:
                # When window is RESTORED/Normal (shows maximize icon ⬜) - use SMALLER size (640x480)
                self.detection_label.setMinimumSize(640, 480)
                self.detection_label.setFixedSize(640, 480)
                
                self.inspection_label.setMinimumSize(640, 480)
                self.inspection_label.setFixedSize(640, 480)
                
                # Resize window back to 900x850 when restored
                self.resize(900, 850)
        
        super().changeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = RobotVisionGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()