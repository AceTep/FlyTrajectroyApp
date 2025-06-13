import pandas as pd
import cv2
import numpy as np
import os
import tomllib
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout,
    QFileDialog, QMessageBox, QProgressDialog,
    QHBoxLayout, QSlider, QLabel, QDialog,
    QCheckBox, QDialogButtonBox, QTabWidget, QComboBox, 
    QLineEdit, QFormLayout, QFrame, QScrollArea
)
from app.video_processor import VideoProcessingThread
from app.video_player import VideoPlayerWindow
from utils.helpers import generate_fly_colors
from utils.theme import (
    DARK_GRAY,
    MEDIUM_GRAY,
    LIGHT_GRAY,
    ACCENT_COLOR,
    TEXT_COLOR,
)
from app.widgets import create_button, create_section_title



class CSVFilterApp(QWidget):
    def __init__(self):
        super().__init__()

        # Set main window title and initial size
        self.setWindowTitle("Fly Trajectory Visualizer")
        self.resize(900, 700)

        # Main vertical layout for the entire window content
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)  # Outer spacing inside window edges
        self.layout.setSpacing(15)  # Spacing between child widgets in layout

        # Apply a modern dark-themed stylesheet with specific colors for various widgets
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {DARK_GRAY};
                color: {TEXT_COLOR};
                font-family: Segoe UI, Arial;
            }}
            QTextEdit {{
                background-color: {MEDIUM_GRAY};
                border: 1px solid {LIGHT_GRAY};
                border-radius: 4px;
                padding: 8px;
            }}
            QTabWidget::pane {{
                border: 1px solid {LIGHT_GRAY};
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background: {MEDIUM_GRAY};
                color: {TEXT_COLOR};
                padding: 8px 16px;
                border: 1px solid {LIGHT_GRAY};
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background: {DARK_GRAY};
                border-bottom: 2px solid {ACCENT_COLOR};
            }}
            QSlider::groove:horizontal {{
                height: 8px;
                background: {LIGHT_GRAY};
                margin: 2px 0;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT_COLOR};
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT_COLOR};
            }}
            QCheckBox {{
                spacing: 8px;
            }}
            QLineEdit {{
                background-color: {MEDIUM_GRAY};
                border: 1px solid {LIGHT_GRAY};
                border-radius: 4px;
                padding: 5px;
            }}
            QComboBox {{
                background-color: {MEDIUM_GRAY};
                border: 1px solid {LIGHT_GRAY};
                border-radius: 4px;
                padding: 5px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {MEDIUM_GRAY};
                selection-background-color: {ACCENT_COLOR};
            }}
        """)

        # Title label with custom styling for emphasis
        title = QLabel("Fly Trajectory Visualizer")
        title.setStyleSheet(f"""
            QLabel {{
                font-size: 24px;
                font-weight: bold;
                color: {ACCENT_COLOR};
                padding-bottom: 10px;
            }}
        """)
        self.layout.addWidget(title)

        # Horizontal layout dividing main content area into left and right panels
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left panel: holds input file loading buttons and action buttons
        left_panel = QFrame()
        left_panel.setStyleSheet(f"background-color: {MEDIUM_GRAY}; border-radius: 5px;")
        left_panel.setFixedWidth(300)  # Fixed width for consistent layout
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(15)

        # Input Files section header
        left_layout.addWidget(create_section_title("Input Files"))

        # Buttons for loading various required input files
        self.load_button = create_button("Load Fly CSVs")
        self.load_button.clicked.connect(self.load_csv)
        left_layout.addWidget(self.load_button)

        self.load_video_button = create_button("Load Background Video")
        self.load_video_button.clicked.connect(self.load_video)
        left_layout.addWidget(self.load_video_button)

        self.load_edgelist_button = create_button("Load Edgelist CSV")
        self.load_edgelist_button.clicked.connect(self.load_edgelist)
        left_layout.addWidget(self.load_edgelist_button)

        self.load_calibration_button = create_button("Load Calibration File")
        self.load_calibration_button.clicked.connect(self.load_calibration)
        left_layout.addWidget(self.load_calibration_button)

        # Actions section header
        left_layout.addWidget(create_section_title("Actions"))

        # Options dialog button
        self.options_button = create_button("Options")
        self.options_button.clicked.connect(self.open_options_dialog)
        left_layout.addWidget(self.options_button)

        # Generate video button (initially disabled until inputs are loaded)
        self.video_button = create_button("Generate Video")
        self.video_button.clicked.connect(self.generate_video)
        self.video_button.setEnabled(False)
        left_layout.addWidget(self.video_button)

        # Play generated video button (disabled initially)
        self.play_button = create_button("Play Video")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_embedded_video)
        left_layout.addWidget(self.play_button)

        # Play loaded raw video button with prompt (always enabled)
        self.play_loaded_video_button = create_button("Play Loaded Video")
        self.play_loaded_video_button.clicked.connect(self.play_loaded_video_with_prompt)
        left_layout.addWidget(self.play_loaded_video_button)

        # Add stretch at the bottom to push buttons to top
        left_layout.addStretch()

        # Add left panel to main content horizontal layout
        content_layout.addWidget(left_panel)

        # Right panel: used for preview, status messages, and file lists
        right_panel = QFrame()
        right_panel.setStyleSheet(f"background-color: {MEDIUM_GRAY}; border-radius: 5px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(15)

        # Add the right panel to the content layout with stretch factor 1 (expandable)
        content_layout.addWidget(right_panel, 1)

        # Add the horizontal content layout to the main vertical layout with stretch 1
        self.layout.addLayout(content_layout, 1)

        # Preview & Status section title in right panel
        right_layout.addWidget(create_section_title("Preview & Status"))

        # Scrollable area to display loaded files as checkboxes or status info
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)  # Allow scroll area to resize with window
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {LIGHT_GRAY};
                border-radius: 4px;
                background-color: {DARK_GRAY};
            }}
            QScrollBar:vertical {{
                width: 12px;
                background: {MEDIUM_GRAY};
            }}
            QScrollBar::handle:vertical {{
                background: {LIGHT_GRAY};
                min-height: 20px;
            }}
        """)

        # Container widget inside scroll area that holds dynamically created file checkboxes
        self.file_list_container = QWidget()
        self.file_list_layout = QVBoxLayout(self.file_list_container)
        self.file_list_layout.setAlignment(Qt.AlignTop)  # Align checkboxes to top
        self.scroll_area.setWidget(self.file_list_container)

        # Add the scroll area widget to the right panel layout
        right_layout.addWidget(self.scroll_area)

        # Initialize internal data structures and state variables

        self.all_flies_df = None  # Pandas DataFrame holding all fly trajectory data
        self.video_path = None  # Path to the loaded background video
        self.edgelist_path = None  # Path to the loaded edgelist CSV file
        self.generated_video_path = None  # Path to the output generated video
        self.fly_colors = {}  # Dictionary mapping fly IDs to RGB color tuples

        # Visualization options with default values
        self.use_blank_background = False
        self.draw_boxes = True
        self.show_labels = True
        self.draw_arrows = True
        self.show_frame_counter = True
        self.scale_factor = 1.0
        self.edge_persistence_seconds = 0
        self.enable_screenshot_saving = False
        self.screenshot_interval_min = 1
        self.start_time_min = 0
        self.end_time_min = None
        self.draw_petri_circle = False
        self.min_edge_duration = 0
        self.color_code_edges = False

        self.calibration_path = None  # Path to calibration file
        # Default calibration values, used if no file is loaded
        self.calibration_values = {
            'min_x': 553.023338607595,
            'min_y': 167.17559769167354,
            'x_px_ratio': 31.839003077183513,
            'y_px_ratio': 32.18860843823452
        }

        # Dictionaries to keep track of checkboxes and file paths loaded by user
        self.file_checkboxes = {}
        self.video_checkbox = None
        self.edgelist_checkbox = None
        self.calibration_checkbox = None

        self.file_paths = {
            'video': None,
            'edgelist': None,
            'calibration': None
        }

        # Progress dialog shown during video generation with cancel support and styling
        self.progress_dialog = QProgressDialog("Processing video...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Video Progress")
        self.progress_dialog.setWindowModality(Qt.WindowModal)  # Modal to block interaction during processing
        self.progress_dialog.canceled.connect(self.cancel_video_processing)  # Cancel signal handler
        self.progress_dialog.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {LIGHT_GRAY};
                border-radius: 3px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {ACCENT_COLOR};
                width: 10px;
            }}
        """)
        self.progress_dialog.reset()  # Reset progress dialog to initial state


    def open_options_dialog(self):
        """
        Opens an Options dialog window with multiple tabs allowing the user to configure
        various settings related to video visualization, timing, and output scaling.

        The dialog contains three tabs:

        1. Visuals Tab:
            - Checkbox options for toggling visual elements such as:
                * Use blank background instead of video
                * Draw bounding boxes around flies
                * Show fly labels
                * Draw arrows for flies (or circles if off)
                * Show frame counter
                * Draw petri dish circle
                * Use smaller zoom for the fly grid view
            - Initializes checkbox states from current instance variables.

        2. Timing Tab:
            - Controls related to timing and persistence of overlays:
                * Start and End time inputs in minutes to filter the video processing time range
                * Slider for edge persistence duration (how long to keep interaction edges visible)
                * Option to enable saving screenshots every X minutes, with slider to set interval
                * Input for minimum interaction duration in frames to filter short edges
                * Checkbox to color-code edges based on interaction duration
            - Inputs and sliders are initialized from current instance variables.
            - Includes validation and safe conversion of text inputs to appropriate types.

        3. Scale Tab:
            - Dropdown combo box to select output video scale factor (100%, 75%, 50%, 25%)
            - Initializes selection based on current scale_factor

        Dialog Layout:
            - Uses QTabWidget to organize tabs
            - Includes OK and Cancel buttons to accept or reject changes
            - On acceptance, updates instance variables with user selections, including type conversions and error handling for numeric fields.
        """
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Options")
        tabs = QTabWidget()

        # --- Visuals Tab ---
        visuals_tab = QWidget()
        visuals_layout = QVBoxLayout()

        # Checkboxes for visuals options with initial states
        chk_blank = QCheckBox("Use blank background (no video)")
        chk_blank.setChecked(self.use_blank_background)
        chk_boxes = QCheckBox("Draw bounding boxes")
        chk_boxes.setChecked(self.draw_boxes)
        chk_labels = QCheckBox("Show fly labels")
        chk_labels.setChecked(self.show_labels)
        chk_arrows = QCheckBox("Draw fly arrows (off = circles)")
        chk_arrows.setChecked(self.draw_arrows)
        chk_frame_counter = QCheckBox("Show frame counter")
        chk_frame_counter.setChecked(self.show_frame_counter)
        chk_draw_circle = QCheckBox("Draw petri dish circle")
        chk_draw_circle.setChecked(self.draw_petri_circle)
        chk_small_grid_zoom = QCheckBox("Use smaller zoom in Fly Grid (50x50)")
        chk_small_grid_zoom.setChecked(getattr(self, 'use_small_grid_zoom', False))

        # Add all visual checkboxes to layout
        for chk in [chk_blank, chk_boxes, chk_labels, chk_arrows, chk_frame_counter, chk_small_grid_zoom, chk_draw_circle]:
            visuals_layout.addWidget(chk)

        visuals_tab.setLayout(visuals_layout)
        tabs.addTab(visuals_tab, "Visuals")

        # --- Timing Tab ---
        timing_tab = QWidget()
        timing_layout = QVBoxLayout()

        # Time range input controls grouped together
        time_range_group = QWidget()
        time_range_layout = QFormLayout()
        self.start_time_edit = QLineEdit(str(self.start_time_min))  
        self.end_time_edit = QLineEdit(str(self.end_time_min) if self.end_time_min is not None else "")  
        time_range_layout.addRow("Start Time (min):", self.start_time_edit)
        time_range_layout.addRow("End Time (min):", self.end_time_edit)
        time_range_group.setLayout(time_range_layout)

        timing_layout.addWidget(QLabel("Time Range:"))
        timing_layout.addWidget(time_range_group)

        # Edge persistence slider and label
        timing_label = QLabel("Edge Persistence (seconds):")
        self.edge_time_slider = QSlider(Qt.Horizontal)
        self.edge_time_slider.setRange(0, 60)
        self.edge_time_slider.setValue(self.edge_persistence_seconds)
        self.edge_time_label = QLabel(f"{self.edge_persistence_seconds} s")
        self.edge_time_slider.valueChanged.connect(lambda v: self.edge_time_label.setText(f"{v} s"))

        timing_layout.addWidget(timing_label)
        timing_layout.addWidget(self.edge_time_slider)
        timing_layout.addWidget(self.edge_time_label)

        # Screenshot saving options
        self.save_screenshot_checkbox = QCheckBox("Save screenshot every X minutes")
        self.save_screenshot_checkbox.setChecked(self.enable_screenshot_saving)

        self.screenshot_interval_slider = QSlider(Qt.Horizontal)
        self.screenshot_interval_slider.setRange(1, 30)
        self.screenshot_interval_slider.setValue(self.screenshot_interval_min)
        self.screenshot_interval_slider.setEnabled(self.enable_screenshot_saving)

        self.screenshot_interval_label = QLabel(f"{self.screenshot_interval_min} min")
        self.screenshot_interval_slider.valueChanged.connect(
            lambda v: self.screenshot_interval_label.setText(f"{v} min")
        )
        self.save_screenshot_checkbox.toggled.connect(self.screenshot_interval_slider.setEnabled)

        timing_layout.addWidget(self.save_screenshot_checkbox)
        timing_layout.addWidget(self.screenshot_interval_slider)
        timing_layout.addWidget(self.screenshot_interval_label)

        # Minimum interaction duration input
        self.min_duration_label = QLabel("Min Interaction Duration (frames):")
        self.min_duration_input = QLineEdit(str(self.min_edge_duration))
        timing_layout.addWidget(self.min_duration_label)
        timing_layout.addWidget(self.min_duration_input)

        # Color coding edges checkbox
        self.color_code_edges_checkbox = QCheckBox("Color-code edges by duration")
        self.color_code_edges_checkbox.setChecked(self.color_code_edges)
        timing_layout.addWidget(self.color_code_edges_checkbox)

        timing_tab.setLayout(timing_layout)
        tabs.addTab(timing_tab, "Timing")

        # --- Scale Tab ---
        scale_tab = QWidget()
        scale_layout = QVBoxLayout()
        scale_layout.setContentsMargins(10, 10, 10, 10)
        scale_layout.setSpacing(10)

        scale_label = QLabel("Scale Output Video:")
        scale_label.setStyleSheet(f"font-weight: bold; color: {ACCENT_COLOR};")
        scale_layout.addWidget(scale_label)

        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["100%", "75%", "50%", "25%"])
        current_scale_idx = ["100%", "75%", "50%", "25%"].index(f"{int(self.scale_factor * 100)}%")
        self.scale_combo.setCurrentIndex(current_scale_idx)
        scale_layout.addWidget(self.scale_combo)

        scale_layout.addStretch()
        scale_tab.setLayout(scale_layout)
        tabs.addTab(scale_tab, "Scale")

        # Dialog layout with OK/Cancel buttons
        layout = QVBoxLayout()
        layout.addWidget(tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        # Execute dialog modally, update instance variables if accepted
        if dialog.exec_():
            self.use_blank_background = chk_blank.isChecked()
            self.draw_boxes = chk_boxes.isChecked()
            self.show_labels = chk_labels.isChecked()
            self.draw_arrows = chk_arrows.isChecked()
            self.show_frame_counter = chk_frame_counter.isChecked()
            self.edge_persistence_seconds = self.edge_time_slider.value()
            scale_text = self.scale_combo.currentText()
            self.scale_factor = int(scale_text.strip('%')) / 100.0
            self.enable_screenshot_saving = self.save_screenshot_checkbox.isChecked()
            self.screenshot_interval_min = self.screenshot_interval_slider.value()
            self.draw_petri_circle = chk_draw_circle.isChecked()
            self.use_small_grid_zoom = chk_small_grid_zoom.isChecked()

            # Safely convert min interaction duration to int; default to 0 if invalid
            try:
                self.min_edge_duration = int(self.min_duration_input.text())
            except ValueError:
                self.min_edge_duration = 0

            self.color_code_edges = self.color_code_edges_checkbox.isChecked()

            # Safely parse start and end times as float; default to 0 or None if invalid
            try:
                self.start_time_min = float(self.start_time_edit.text()) if self.start_time_edit.text() else 0
            except ValueError:
                self.start_time_min = 0
            try:
                end_text = self.end_time_edit.text()
                self.end_time_min = float(end_text) if end_text else None
            except ValueError:
                self.end_time_min = None


    def load_csv(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Fly CSV Files", "", "CSV Files (*.csv)")
        if file_paths:
            try:
                all_data = []
                self.fly_colors.clear()

                # Clear old checkboxes
                for fly_id in list(self.file_checkboxes.keys()):
                    widget = self.file_checkboxes[fly_id]
                    if widget:
                        widget.setParent(None)
                    del self.file_checkboxes[fly_id]

                fly_ids = []

                for file_path in file_paths:
                    try:
                        # Read fly CSV with needed columns
                        df = pd.read_csv(file_path, usecols=["pos x", "pos y", "ori"])
                        fly_id = os.path.splitext(os.path.basename(file_path))[0]
                        df["fly_id"] = fly_id
                        all_data.append(df)
                        fly_ids.append(fly_id)

                        # Create and add checkbox for each fly
                        chk = QCheckBox(fly_id)
                        chk.setChecked(True)
                        chk.setStyleSheet(f"""
                            QCheckBox {{
                                color: {TEXT_COLOR};
                                spacing: 8px;
                            }}
                            QCheckBox::indicator {{
                                width: 16px;
                                height: 16px;
                            }}
                        """)
                        self.file_list_layout.addWidget(chk)
                        self.file_checkboxes[fly_id] = chk

                    except Exception as e:
                        print(f"Error loading file {file_path}: {str(e)}")
                        continue

                # Generate and assign distinct colors to flies
                self.fly_colors = generate_fly_colors(fly_ids)

                # Combine all fly data into one DataFrame
                self.all_flies_df = pd.concat(all_data, ignore_index=True)

                # Enable video button only if edgelist is already loaded
                if self.edgelist_path:
                    self.video_button.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSVs:\n{str(e)}")

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Background Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.file_paths['video'] = video_path
            video_name = os.path.basename(video_path)

            # Remove old video checkbox if exists
            if self.video_checkbox:
                self.video_checkbox.setParent(None)

            # Create new video checkbox
            self.video_checkbox = QCheckBox(f"Video: {video_name}")
            self.video_checkbox.setChecked(True)
            self.video_checkbox.setStyleSheet(f"""
                QCheckBox {{
                    color: {TEXT_COLOR};
                    spacing: 8px;
                }}
                QCheckBox::indicator {{
                    width: 16px;
                    height: 16px;
                }}
            """)
            self.file_list_layout.addWidget(self.video_checkbox)

            # Enable video button only if flies and edgelist are loaded
            if self.all_flies_df is not None and self.file_paths['edgelist']:
                self.video_button.setEnabled(True)

    def load_edgelist(self):
        edgelist_path, _ = QFileDialog.getOpenFileName(self, "Select Edgelist CSV", "", "CSV Files (*.csv)")
        if edgelist_path:
            self.file_paths['edgelist'] = edgelist_path
            edgelist_name = os.path.basename(edgelist_path)

            # Remove old edgelist checkbox if exists
            if self.edgelist_checkbox:
                self.edgelist_checkbox.setParent(None)

            # Create new edgelist checkbox
            self.edgelist_checkbox = QCheckBox(f"Edgelist: {edgelist_name}")
            self.edgelist_checkbox.setChecked(True)
            self.edgelist_checkbox.setStyleSheet(f"""
                QCheckBox {{
                    color: {TEXT_COLOR};
                    spacing: 8px;
                }}
                QCheckBox::indicator {{
                    width: 16px;
                    height: 16px;
                }}
            """)
            self.file_list_layout.addWidget(self.edgelist_checkbox)

            # Enable video button only if fly data is loaded
            if self.all_flies_df is not None:
                self.video_button.setEnabled(True)

    def load_calibration(self):
        # Open file dialog to select calibration file
        path, _ = QFileDialog.getOpenFileName(self)
        if path:
            try:
                # Load calibration config from toml file
                with open(path, "rb") as f:
                    config = tomllib.load(f)
                
                # Update calibration values from config
                self.calibration_values.update({
                    'min_x': config['min_x'],
                    'min_y': config['min_y'],
                    'x_px_ratio': config['x_px_ratio'],
                    'y_px_ratio': config['y_px_ratio']
                })
                self.file_paths['calibration'] = path
                cal_name = os.path.basename(path)
                
                # Remove old calibration checkbox if it exists
                if self.calibration_checkbox:
                    self.calibration_checkbox.setParent(None)
                
                # Create and add new calibration checkbox
                self.calibration_checkbox = QCheckBox(f"Calibration: {cal_name}")
                self.calibration_checkbox.setChecked(True)
                self.calibration_checkbox.setStyleSheet(f"""
                    QCheckBox {{
                        color: {TEXT_COLOR};
                        spacing: 8px;
                    }}
                    QCheckBox::indicator {{
                        width: 16px;
                        height: 16px;
                    }}
                """)
                self.file_list_layout.addWidget(self.calibration_checkbox)
                
            except Exception as e:
                # Show error message if loading fails
                QMessageBox.critical(self, "Error", f"Failed to load calibration:\n{str(e)}")

    def generate_video(self):
        # Check if fly data is loaded before generating video
        if self.all_flies_df is None:
            QMessageBox.warning(self, "Missing Inputs", "Please load fly CSVs before generating the video.")
            return
        
        # Check if edgelist is loaded and selected
        if not self.file_paths['edgelist'] or (self.edgelist_checkbox and not self.edgelist_checkbox.isChecked()):
            QMessageBox.warning(self, "Missing Edgelist", "Edgelist is required but not selected or unchecked.")
            return
        
        # Get list of checked fly IDs
        checked_ids = [fly_id for fly_id, chk in self.file_checkboxes.items() if chk.isChecked()]
        if not checked_ids:
            QMessageBox.warning(self, "No Files Selected", "Please check at least one fly CSV to include in the video.")
            return
        
        # Filter the fly data for selected flies
        filtered_df = self.all_flies_df[self.all_flies_df['fly_id'].isin(checked_ids)].copy()
        
        use_blank = self.use_blank_background
        video_path = None if use_blank else self.file_paths.get('video')

        # Inform user if using blank background
        if use_blank:
            QMessageBox.information(self, "Blank Background", "Using blank white background as selected in options.")
        # Check if video is loaded and selected if not using blank background
        elif not video_path or (self.video_checkbox and not self.video_checkbox.isChecked()):
            QMessageBox.warning(self, "Missing Video", "No background video selected or checkbox is unchecked.")
            return

        # Set calibration values if calibration file loaded and checkbox checked or missing
        calibration_values = None
        if self.file_paths['calibration'] and (not self.calibration_checkbox or self.calibration_checkbox.isChecked()):
            calibration_values = self.calibration_values
        
        # Show progress dialog and reset progress
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        
        # Create and start video processing thread with all needed parameters
        self.video_thread = VideoProcessingThread(
            filtered_df,
            video_path,
            self.file_paths['edgelist'],
            self.fly_colors,
            use_blank=use_blank,
            draw_boxes=self.draw_boxes,
            draw_labels=self.show_labels,
            draw_arrows=self.draw_arrows,
            show_frame_counter=self.show_frame_counter,
            scale_factor=self.scale_factor,
            edge_persistence_seconds=self.edge_persistence_seconds,
            save_graphs=self.enable_screenshot_saving,
            graph_interval_min=self.screenshot_interval_min,
            start_time_min=self.start_time_min,
            end_time_min=self.end_time_min,
            fly_size=13,
            draw_petri_circle=self.draw_petri_circle,
            min_edge_duration=getattr(self, 'min_edge_duration', 0),
            color_code_edges=getattr(self, 'color_code_edges', False),
            calibration_values=calibration_values)

        # Connect thread signals to UI slots
        self.video_thread.update_progress.connect(self.show_progress)
        self.video_thread.update_progress_bar.connect(self.progress_dialog.setValue)
        self.video_thread.video_saved.connect(self.show_video_saved)
        self.video_thread.finished.connect(self.on_video_thread_finished)
        self.video_thread.start()

    def closeEvent(self, event):
        # This function is called when the main window is closing.
        # It checks if a video popup window is currently open.
        # If so, it attempts to clean up and close that popup gracefully.
        # If an error occurs during cleanup or closing, it silently ignores it.
        # Finally, it accepts the close event to allow the window to close.
        if hasattr(self, 'video_popup'):
            try:
                self.video_popup.cleanup()
                self.video_popup.close()
            except:
                pass
        event.accept()

    def cancel_video_processing(self):
        # Allows the user to cancel the ongoing video processing thread.
        # Checks if the video thread exists and is currently running.
        # If so, calls the thread's cancel method to stop processing.
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.cancel()

    def on_video_thread_finished(self):
        # Slot connected to the video thread's finished signal.
        # When video processing is done, hides the progress dialog.
        self.progress_dialog.hide()

    def show_progress(self, message):
        # Displays a message box to inform the user about the current progress.
        # This can be used for status updates during long operations.
        QMessageBox.information(self, "Progress", message)

    def show_video_saved(self, output_path, elapsed_time):
        # Called when video processing completes and the video is saved.
        # Sets the progress bar to 100%, hides the progress dialog,
        # saves the path of the generated video,
        # enables the play button for playback,
        # and shows an information message with the file path and time taken.
        self.progress_dialog.setValue(100)
        self.progress_dialog.hide()
        self.generated_video_path = output_path
        self.play_button.setEnabled(True)
        QMessageBox.information(self, "Success", f"Video saved to: {output_path}\nTime taken: {elapsed_time / 60:.2f} minutes")

    def play_embedded_video(self):
        # Plays the video that was just generated and saved.
        # First, it verifies that the generated video file exists.
        # If a previous video popup window exists, it tries to close and delete it.
        # Then, it opens the new video using OpenCV to get frame size info.
        # Calls the video thread's method to get transformed fly positions for overlay.
        # Finally, creates and shows a VideoPlayerWindow with the video and fly overlay data.
        # If any error occurs during opening the video player, shows an error dialog.
        if not hasattr(self, 'generated_video_path') or not os.path.exists(self.generated_video_path):
            QMessageBox.warning(self, "Error", "Generated video not found.")
            return

        if hasattr(self, 'video_popup'):
            try:
                self.video_popup.close()
                self.video_popup.deleteLater()
            except Exception as e:
                print(f"Error closing existing video window: {str(e)}")
            finally:
                if hasattr(self, 'video_popup'):
                    del self.video_popup

        cap = cv2.VideoCapture(self.generated_video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        fly_positions, _ = self.video_thread.transform_fly_positions(frame_width, frame_height)

        try:
            self.video_popup = VideoPlayerWindow(self.generated_video_path, fly_positions)
            self.video_popup.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open video player: {str(e)}")

    def play_loaded_video_with_prompt(self):
        # Allows the user to manually select a generated video to play.
        # Prompts the user with a dialog to select the video file.
        # If no file is selected, the function returns early.
        # Asks the user if they want to enable the Fly Grid View, which shows a grid of fly positions.
        # Attempts to automatically load an associated `.npz` file containing fly position data for grid view.
        # If loading `.npz` fails or doesn't exist, prompts user to load fly CSVs and calibration manually.
        # Loads fly CSV data, combines it into a single DataFrame, and generates fly colors.
        # Loads calibration data from a TOML file for accurate spatial transformations.
        # Reads video frame size to properly transform fly positions.
        # Uses VideoProcessingThread's transformation function to get fly positions in video frame coordinates.
        # Shows the video player window with the grid overlay if grid view enabled.
        # If grid view is not enabled, plays video normally without overlays.
        # Catches and displays errors if video or data loading fails at any step.
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Generated Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if not video_path:
            return

        use_grid = QMessageBox.question(
            self,
            "Enable Grid View",
            "Do you want to enable Fly Grid View?",
            QMessageBox.Yes | QMessageBox.No
        ) == QMessageBox.Yes

        # Try to auto-load .npz file with fly positions and calibration
        npz_path = f"{os.path.splitext(video_path)[0]}_data.npz"
        if use_grid and os.path.exists(npz_path):
            try:
                data = np.load(npz_path, allow_pickle=True)
                transformed_positions = data['fly_positions'].item()
                calibration = data['calibration'].item()
                self.video_popup = VideoPlayerWindow(video_path, transformed_positions)
                self.video_popup.show()
                return
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load fly data from {npz_path}:\n{str(e)}")

        # If no .npz file or failed to load, manual CSV loading for grid view
        if use_grid:
            # Prompt user to select fly CSV files for loading position data
            file_paths, _ = QFileDialog.getOpenFileNames(self, "Load Fly CSVs", "", "CSV Files (*.csv)")
            if not file_paths:
                QMessageBox.warning(self, "Missing Data", "No CSVs loaded. Grid View requires fly positions.")
                return

            all_data = []
            fly_ids = []
            # Load each CSV and add a fly_id column derived from the filename
            for path in file_paths:
                df = pd.read_csv(path, usecols=["pos x", "pos y", "ori"])
                fly_id = os.path.splitext(os.path.basename(path))[0]
                df["fly_id"] = fly_id
                all_data.append(df)
                fly_ids.append(fly_id)

            # Concatenate all fly data into a single DataFrame
            all_flies_df = pd.concat(all_data, ignore_index=True)
            # Generate colors for each fly id for visualization
            fly_colors = generate_fly_colors(fly_ids)

            # Prompt user to load calibration file (required for accurate positioning)
            cal_path, _ = QFileDialog.getOpenFileName(self, "Load Calibration (.toml)", "", "TOML Files (*.toml)")
            if not cal_path:
                QMessageBox.warning(self, "Missing Calibration", "Calibration file is required for accurate fly grid view.")
                return
            try:
                with open(cal_path, "rb") as f:
                    config = tomllib.load(f)
                calibration_values = {
                    'min_x': config['min_x'],
                    'min_y': config['min_y'],
                    'x_px_ratio': config['x_px_ratio'],
                    'y_px_ratio': config['y_px_ratio']
                }
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read calibration file:\n{str(e)}")
                return

            # Get frame dimensions from the selected video for correct fly position transformation
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Use VideoProcessingThread's transform function to map fly positions onto video frames
            transformed_positions, _ = VideoProcessingThread(
                all_flies_df, video_path, fly_colors=fly_colors, calibration_values=calibration_values
            ).transform_fly_positions(frame_width, frame_height)

            # Create and show video player window with grid overlay of fly positions
            self.video_popup = VideoPlayerWindow(video_path, transformed_positions)
            self.video_popup.show()
        else:
            # If grid view not selected, just open video playback without overlays
            try:
                cap = cv2.VideoCapture(video_path)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                self.video_popup = VideoPlayerWindow(video_path, {})
                self.video_popup.show()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open video:\n{str(e)}")
