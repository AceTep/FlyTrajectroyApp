import sys
import pandas as pd
import cv2
import numpy as np
import time
import csv
import os
import traceback
from collections import defaultdict, deque
import tomllib
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QTextEdit, QMessageBox, QProgressDialog,
    QHBoxLayout, QStyle, QSlider, QLabel, QDialog,
    QCheckBox, QDialogButtonBox, QTabWidget, QComboBox, 
    QLineEdit, QFormLayout, QFrame, QScrollArea
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QFont, QPalette, QColor
import hashlib

DARK_GRAY = "#2D2D2D"
MEDIUM_GRAY = "#3E3E3E"
LIGHT_GRAY = "#4F4F4F"
ACCENT_COLOR = "#5D9B9B"
WHITE = "#FFFFFF"
TEXT_COLOR = "#E0E0E0"

def generate_fly_color(fly_id):
    digest = hashlib.md5(fly_id.encode()).hexdigest()
    r = int(digest[0:2], 16)
    g = int(digest[2:4], 16)
    b = int(digest[4:6], 16)
    return (r, g, b)

def setup_dark_theme(app):
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(DARK_GRAY))
    palette.setColor(QPalette.WindowText, QColor(TEXT_COLOR))
    palette.setColor(QPalette.Base, QColor(MEDIUM_GRAY))
    palette.setColor(QPalette.AlternateBase, QColor(DARK_GRAY))
    palette.setColor(QPalette.ToolTipBase, QColor(WHITE))
    palette.setColor(QPalette.ToolTipText, QColor(WHITE))
    palette.setColor(QPalette.Text, QColor(TEXT_COLOR))
    palette.setColor(QPalette.Button, QColor(MEDIUM_GRAY))
    palette.setColor(QPalette.ButtonText, QColor(TEXT_COLOR))
    palette.setColor(QPalette.BrightText, QColor(WHITE))
    palette.setColor(QPalette.Highlight, QColor(ACCENT_COLOR))
    palette.setColor(QPalette.HighlightedText, QColor(WHITE))
    app.setPalette(palette)

def create_button(text, parent=None):
    button = QPushButton(text, parent)
    button.setStyleSheet(f"""
        QPushButton {{
            background-color: {MEDIUM_GRAY};
            color: {TEXT_COLOR};
            border: 1px solid {LIGHT_GRAY};
            border-radius: 4px;
            padding: 8px;
            min-width: 120px;
        }}
        QPushButton:hover {{
            background-color: {LIGHT_GRAY};
            border: 1px solid {ACCENT_COLOR};
        }}
        QPushButton:pressed {{
            background-color: {DARK_GRAY};
        }}
    """)
    return button

def create_section_title(text):
    label = QLabel(text)
    label.setStyleSheet(f"""
        QLabel {{
            color: {ACCENT_COLOR};
            font-weight: bold;
            font-size: 14px;
            padding: 5px 0;
        }}
    """)
    return label

class VideoProcessingThread(QThread):
    update_progress = pyqtSignal(str)
    update_progress_bar = pyqtSignal(int)
    video_saved = pyqtSignal(str, float)
    
    def __init__(self, all_flies_df, video_path, edgelist_path=None, fly_colors=None,
                 use_blank=False, draw_boxes=True, draw_labels=True, draw_arrows=True,
                 show_frame_counter=True, scale_factor=1.0, edge_persistence_seconds=0,
                 save_graphs=False, graph_interval_min=1, start_time_min=0, end_time_min=None, 
                 fly_size=10, draw_petri_circle=False, min_edge_duration=0, color_code_edges=False,
                 calibration_values=None): 
        super().__init__()
        self.fly_size = fly_size  
        self.all_flies_df = all_flies_df
        self.video_path = video_path
        self.edgelist_path = edgelist_path
        self.fly_colors = fly_colors or {}
        self.use_blank = use_blank
        self.draw_boxes = draw_boxes
        self.draw_labels = draw_labels
        self.draw_arrows = draw_arrows
        self.show_frame_counter = show_frame_counter
        self._is_cancelled = False
        self.scale_factor = scale_factor
        self.edge_persistence_seconds = edge_persistence_seconds
        self.save_graphs = save_graphs
        self.graph_interval_min = graph_interval_min
        self.start_time_min = start_time_min
        self.end_time_min = end_time_min    
        self.draw_petri_circle = draw_petri_circle
        self.min_edge_duration = min_edge_duration
        self.color_code_edges = color_code_edges
        self.calibration_values = calibration_values 
        

    def cancel(self):
        self._is_cancelled = True

    def parse_edgelist(self, path):
        interactions = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    start = int(row["start_of_interaction"])
                    end = int(row["end_of_interaction"])
                    if (end - start) >= self.min_edge_duration:
                        node_1 = row["node_1"]
                        node_2 = row["node_2"]
                        interactions.append((start, end, node_1, node_2))
                except Exception as e:
                    print(f"Skipping row: {e}")
        return sorted(interactions, key=lambda x: x[0])


    def get_video_info(self, cap):
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.scale_factor)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.scale_factor)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return width, height, fps, total_frames
    
    def draw_static_tip_arrow(self, frame, pt1, pt2, color, thickness=4, tip_length=20, duration=None):
        if self.color_code_edges and duration is not None:
            if duration < 50:
                color = (0, 255, 0)      # Green
            elif duration < 150:
                color = (0, 255, 255)    # Yellow
            else:
                color = (0, 0, 255)      # Red
        x1, y1 = pt1
        x2, y2 = pt2
        
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance == 0:
            return
        
        max_offset = 50
        min_offset = 10
        min_distance_for_max_offset = 200
        
        offset = min_offset + (max_offset - min_offset) * min(distance / min_distance_for_max_offset, 1)
        
        ux = dx / distance
        uy = dy / distance
        
        start_x = int(x1 + ux * offset)
        start_y = int(y1 + uy * offset)
        end_x = int(x2 - ux * offset)
        end_y = int(y2 - uy * offset)
        
        cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)
        
        base_x = end_x - int(ux * tip_length)
        base_y = end_y - int(uy * tip_length)
        
        nx = -uy
        ny = ux
        
        wing_size = tip_length // 2
        wing1 = (int(base_x + nx * wing_size), int(base_y + ny * wing_size))
        wing2 = (int(base_x - nx * wing_size), int(base_y - ny * wing_size))
        
        cv2.fillConvexPoly(frame, np.array([[end_x, end_y], wing1, wing2], dtype=np.int32), color)

    def draw_edges(self, frame, frame_idx, interactions, transformed_positions):
        for (start, end, fly1, fly2) in interactions:
            extended_end = end + int(self.edge_persistence_seconds * self.fps)
            if start <= frame_idx <= extended_end:
                if (fly1 in transformed_positions and
                    fly2 in transformed_positions and
                    frame_idx < len(transformed_positions[fly1]) and
                    frame_idx < len(transformed_positions[fly2])):

                    x1, y1, _ = transformed_positions[fly1][frame_idx]
                    x2, y2, _ = transformed_positions[fly2][frame_idx]
                    duration = end - start
                    self.draw_static_tip_arrow(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                            (0, 0, 0), thickness=4, tip_length=20, duration=duration)

    def transform_fly_positions(self, frame_width, frame_height):
        fly_data = {
            fly_id: df.reset_index(drop=True)
            for fly_id, df in self.all_flies_df.groupby("fly_id")
        }

        try:
            min_x = self.calibration_values['min_x']
            min_y = self.calibration_values['min_y']
            x_px_ratio = self.calibration_values['x_px_ratio']
            y_px_ratio = self.calibration_values['y_px_ratio']

            transformed = {}
            for fly_id, df in fly_data.items():
                # First apply calibration
                scaled_x = (df["pos x"].values * x_px_ratio) + min_x
                scaled_y = (df["pos y"].values * y_px_ratio) + min_y
                # Then apply scale factor
                scaled_x = scaled_x * self.scale_factor
                scaled_y = scaled_y * self.scale_factor
                ori = df["ori"].values
                transformed[fly_id] = np.stack([scaled_x, scaled_y, ori], axis=1)

            return transformed, min(len(df) for df in fly_data.values())

        except Exception as e:
            print("Calibration fallback activated:", e)

            all_x = self.all_flies_df["pos x"].values
            all_y = self.all_flies_df["pos y"].values
            min_x, max_x = all_x.min(), all_x.max()
            min_y, max_y = all_y.min(), all_y.max()

            scale = 0.9 * min(frame_width / (max_x - min_x), frame_height / (max_y - min_y))
            offset_x = (frame_width - scale * (max_x - min_x)) / 2
            offset_y = (frame_height - scale * (max_y - min_y)) / 2

            transformed = {}
            for fly_id, df in fly_data.items():
                x = (df["pos x"].values - min_x) * scale + offset_x
                y = (df["pos y"].values - min_y) * scale + offset_y
                ori = df["ori"].values
                transformed[fly_id] = np.stack([x, y, ori], axis=1)

            return transformed, min(len(df) for df in fly_data.values())


    def run(self):
        cap = None
        out = None
        try:
            start_time = time.time()
            
            if self.use_blank:
                cap = None
                frame_width, frame_height = 3052, 2304  
                frame_width = int(frame_width * self.scale_factor)
                frame_height = int(frame_height * self.scale_factor)
                self.fps = 24  
                transformed_positions, max_data_len = self.transform_fly_positions(frame_width, frame_height)
                total_frames = max_data_len
            else:
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    self.update_progress.emit("Failed to open video.")
                    return
                frame_width, frame_height, self.fps, total_frames = self.get_video_info(cap)
                transformed_positions, max_data_len = self.transform_fly_positions(frame_width, frame_height)

            start_frame = int(self.start_time_min * 60 * self.fps)
            end_frame = int(self.end_time_min * 60 * self.fps) if self.end_time_min is not None else None
        
            if end_frame is None or end_frame > max_data_len:
                end_frame = max_data_len
            if start_frame >= end_frame:
                self.update_progress.emit("Invalid time range selected.")
                return

            base_name = os.path.splitext(os.path.basename(self.video_path if self.video_path else "blank"))[0]
            time_suffix = f"_{self.start_time_min}to{self.end_time_min}min" if self.end_time_min else ""
            output_filename = f"{base_name}_overlay_fly{time_suffix}.mp4"
            out = cv2.VideoWriter(
                output_filename,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps, (frame_width, frame_height))
            
            if not out.isOpened():
                self.update_progress.emit("Failed to create output video file.")
                return

            interactions = self.parse_edgelist(self.edgelist_path) if self.edgelist_path else []
            frame_idx = start_frame
            last_screenshot_frame = -1

            if not self.use_blank:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            while frame_idx < end_frame:
                if self._is_cancelled:
                    break

                if self.use_blank:
                    frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
                    if self.use_blank and self.draw_petri_circle:
                        center = (frame.shape[1] // 2, frame.shape[0] // 2)
                        radius = min(frame.shape[1], frame.shape[0]) // 2 - 50
                        cv2.circle(frame, center, radius, (200, 200, 200), 8)
                    ret = True
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if self.scale_factor != 1.0:
                        frame = cv2.resize(frame, (frame_width, frame_height))

                self.draw_flies(frame, frame_idx, transformed_positions)
                self.draw_edges(frame, frame_idx, interactions, transformed_positions)
                if self.draw_boxes:
                    self.draw_interaction_groups(frame, frame_idx, interactions, transformed_positions)

                if self.show_frame_counter:
                    self.draw_frame_counter(frame, frame_idx)

                if self.save_graphs:
                    screenshot_every_frames = int(self.graph_interval_min * 60 * self.fps)
                    if frame_idx % screenshot_every_frames == 0 and frame_idx != last_screenshot_frame:
                        screenshot_path = f"screenshot_frame_{frame_idx}.png"
                        cv2.imwrite(screenshot_path, frame)
                        last_screenshot_frame = frame_idx

                out.write(frame)
                progress = int(((frame_idx - start_frame) / (end_frame - start_frame)) * 100)
                self.update_progress_bar.emit(progress)
                frame_idx += 1

            if self._is_cancelled:
                self.update_progress.emit("Video generation cancelled.")
                return

            elapsed = time.time() - start_time
            self.video_saved.emit(output_filename, elapsed)

        except Exception as e:
            error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.update_progress.emit(error_message)
        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()

    def draw_frame_counter(self, frame, frame_idx):
        text = f"FRAMES: {frame_idx}"
        position = (20, 40) 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (0, 0, 0) if self.use_blank else (255, 255, 255)
        thickness = 2
        cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    def draw_offset_edge(self, frame, pos1, pos2, color, thickness=2, offset=50):
        p1 = np.array(pos1, dtype=np.float32)
        p2 = np.array(pos2, dtype=np.float32)
        vec = p2 - p1
        norm = np.linalg.norm(vec)
        if norm == 0:
            return  
        unit_vec = vec / norm
        offset_vec = unit_vec * offset
        start = tuple(np.round(p1 + offset_vec).astype(int))
        end = tuple(np.round(p2 - offset_vec).astype(int))
        cv2.line(frame, start, end, color, thickness)


    def draw_flies(self, frame, frame_idx, transformed_positions):
        label_color = (0, 0, 0) if self.use_blank else (255, 255, 255)
        for fly_id, coords in transformed_positions.items():
            if frame_idx >= len(coords):
                continue
            x, y, ori = coords[frame_idx]
            x, y = int(x), int(y)
            color = self.fly_colors.get(fly_id, (255, 255, 255))
            circle_size = self.fly_size * 2 if self.use_blank else self.fly_size
            
            if self.draw_arrows:
                triangle_size = circle_size 
                
                front_x = x + int(triangle_size * 2 * np.cos(ori))
                front_y = y - int(triangle_size * 2 * np.sin(ori))
                
                back_left_x = x + int(triangle_size * np.cos(ori + np.pi/2))
                back_left_y = y - int(triangle_size * np.sin(ori + np.pi/2))
                
                back_right_x = x + int(triangle_size * np.cos(ori - np.pi/2))
                back_right_y = y - int(triangle_size * np.sin(ori - np.pi/2))
                
                triangle_pts = np.array([[front_x, front_y], [back_left_x, back_left_y], [back_right_x, back_right_y]])
                cv2.fillConvexPoly(frame, triangle_pts, color)
                
            else:
                cv2.circle(frame, (x, y), circle_size, color, -1) 
            
            if self.draw_labels:
                cv2.putText(frame, str(fly_id), (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, label_color, 2, cv2.LINE_AA)


    def draw_interaction_groups(self, frame, frame_idx, interactions, transformed_positions):
        adjacency = defaultdict(set)
        
        for (start, end, fly1, fly2) in interactions:
            extended_end = end + int(self.edge_persistence_seconds * self.fps)
            if start <= frame_idx <= extended_end:
                if (fly1 in transformed_positions and 
                    fly2 in transformed_positions and
                    frame_idx < len(transformed_positions[fly1]) and
                    frame_idx < len(transformed_positions[fly2])):
                    
                    x1, y1, _ = transformed_positions[fly1][frame_idx]
                    x2, y2, _ = transformed_positions[fly2][frame_idx]
                    
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    if distance <= 300:
                        adjacency[fly1].add(fly2)
                        adjacency[fly2].add(fly1)

        visited = set()
        components = []

        for fly in adjacency:
            if fly not in visited:
                group = set()
                queue = deque([fly])
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        group.add(current)
                        queue.extend(adjacency[current] - visited)
                components.append(group)

        for group in components:
            xs, ys = [], []
            group_positions = []
            
            for fly_id in group:
                if fly_id in transformed_positions and frame_idx < len(transformed_positions[fly_id]):
                    x, y, _ = transformed_positions[fly_id][frame_idx]
                    xs.append(x)
                    ys.append(y)
                    group_positions.append((x, y))
            
            if not group_positions:
                continue
                
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            
            max_distance = 0
            for i in range(len(group_positions)):
                for j in range(i+1, len(group_positions)):
                    x1, y1 = group_positions[i]
                    x2, y2 = group_positions[j]
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if distance > max_distance:
                        max_distance = distance
            
            if max_distance <= 300:
                margin = 60 
                min_x_box = max(0, int(min_x - margin))
                max_x_box = min(frame.shape[1], int(max_x + margin))
                min_y_box = max(0, int(min_y - margin))
                max_y_box = min(frame.shape[0], int(max_y + margin))
                
                group_size = len(group)
                if group_size == 2:
                    box_color = (0, 128, 0)  # Green
                elif group_size == 3:
                    box_color = (0, 255, 255)  # Yellow
                else:
                    box_color = (0, 0, 255)  # Red
                
                cv2.rectangle(frame, (min_x_box, min_y_box), (max_x_box, max_y_box), box_color, 3)
                

                
class VideoPlayerWindow(QWidget):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle("Video Player")
        self.resize(800, 600)
        self.setStyleSheet(f"background-color: {DARK_GRAY};")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet(f"background-color: {DARK_GRAY};")
        layout.addWidget(self.video_widget)

        # Control panel
        control_panel = QFrame()
        control_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {MEDIUM_GRAY};
                border-radius: 5px;
                padding: 10px;
            }}
        """)
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)
        
        # Control buttons
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.setStyleSheet("QPushButton { padding: 8px; }")
        self.play_button.clicked.connect(self.play_video)
        
        self.pause_button = QPushButton()
        self.pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pause_button.setStyleSheet("QPushButton { padding: 8px; }")
        self.pause_button.clicked.connect(self.pause_video)
        
        self.stop_button = QPushButton()
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.setStyleSheet("QPushButton { padding: 8px; }")
        self.stop_button.clicked.connect(self.stop_video)
        
        self.fullscreen_button = create_button("Fullscreen")
        self.skip_back_button = create_button("<< 5s")
        self.skip_forward_button = create_button(">> 5s")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.skip_back_button.clicked.connect(self.skip_back)
        self.skip_forward_button.clicked.connect(self.skip_forward)

        for btn in [self.play_button, self.pause_button, self.stop_button, 
                   self.skip_back_button, self.skip_forward_button, self.fullscreen_button]:
            control_layout.addWidget(btn)

        layout.addWidget(control_panel)

        # Progress slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setStyleSheet(f"""
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
        """)
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self.seek_video)
        layout.addWidget(self.slider)

        self.media_player = None
        
        try:
            self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
            self.media_player.setVideoOutput(self.video_widget)
            self.media_player.positionChanged.connect(self.update_slider_position)
            self.media_player.durationChanged.connect(self.update_slider_range)
            
            media_content = QMediaContent(QUrl.fromLocalFile(os.path.abspath(video_path)))
            self.media_player.setMedia(media_content)
            
            # Connect state changed signal
            self.media_player.stateChanged.connect(self.handle_state_change)
        except Exception as e:
            print(f"Error initializing media player: {str(e)}")
            self.media_player = None

    def handle_state_change(self, state):
        """Handle media player state changes"""
        if state == QMediaPlayer.StoppedState and hasattr(self, 'media_player') and self.media_player:
            self.cleanup()

    def cleanup(self):
        """Safely clean up media player resources"""
        try:
            if hasattr(self, 'media_player') and self.media_player:
                # Disconnect signals first
                try:
                    self.media_player.positionChanged.disconnect()
                    self.media_player.durationChanged.disconnect()
                    self.media_player.stateChanged.disconnect()
                except:
                    pass
                
                # Stop and clear media
                self.media_player.stop()
                self.media_player.setMedia(QMediaContent())
                
                # Delete the media player
                self.media_player.deleteLater()
                self.media_player = None
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def closeEvent(self, event):
        """Handle window closing"""
        try:
            self.cleanup()
        except Exception as e:
            print(f"Error during window close: {str(e)}")
        finally:
            event.accept()

    def play_video(self): self.media_player.play()
    def pause_video(self): self.media_player.pause()
    def stop_video(self): self.media_player.stop()
    def skip_back(self): self.media_player.setPosition(max(0, self.media_player.position() - 5000))
    def skip_forward(self): self.media_player.setPosition(min(self.media_player.duration(), self.media_player.position() + 5000))
    def seek_video(self, position): self.media_player.setPosition(position)
    def update_slider_position(self, position): self.slider.setValue(position)
    def update_slider_range(self, duration): self.slider.setRange(0, duration)
    def toggle_fullscreen(self): self.showFullScreen() if not self.isFullScreen() else self.showNormal()



class CSVFilterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fly Trajectory Visualizer")
        self.resize(900, 700)
        
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)
        
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
        
        # Title
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
        
        # Main content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Left panel - Input controls
        left_panel = QFrame()
        left_panel.setStyleSheet(f"background-color: {MEDIUM_GRAY}; border-radius: 5px;")
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(15)
        
        # Input section
        left_layout.addWidget(create_section_title("Input Files"))
        
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
        
        # Action section
        left_layout.addWidget(create_section_title("Actions"))
        
        self.options_button = create_button("Options")
        self.options_button.clicked.connect(self.open_options_dialog)
        left_layout.addWidget(self.options_button)
        
        self.video_button = create_button("Generate Video")
        self.video_button.clicked.connect(self.generate_video)
        self.video_button.setEnabled(False)
        left_layout.addWidget(self.video_button)
        
        self.play_button = create_button("Play Video")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_embedded_video)
        left_layout.addWidget(self.play_button)
        
        left_layout.addStretch()
        content_layout.addWidget(left_panel)
        
        # Right panel - Preview and status
        right_panel = QFrame()
        right_panel.setStyleSheet(f"background-color: {MEDIUM_GRAY}; border-radius: 5px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(15)
        
        
        
        content_layout.addWidget(right_panel, 1)
        self.layout.addLayout(content_layout, 1)
        
        right_layout.addWidget(create_section_title("Preview & Status"))
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
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
        
        # Container widget for file checkboxes
        self.file_list_container = QWidget()
        self.file_list_layout = QVBoxLayout(self.file_list_container)
        self.file_list_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.file_list_container)
        
        # Add scroll area to right layout
        right_layout.addWidget(self.scroll_area)
        


        self.all_flies_df = None
        self.video_path = None
        self.edgelist_path = None
        self.generated_video_path = None
        self.fly_colors = {}
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
        self.calibration_path = None
        self.calibration_values = {
            'min_x': 553.023338607595,
            'min_y': 167.17559769167354,
            'x_px_ratio': 31.839003077183513,
            'y_px_ratio': 32.18860843823452
        }
        self.file_checkboxes = {}

        # Progress dialog
        self.progress_dialog = QProgressDialog("Processing video...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Video Progress")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.canceled.connect(self.cancel_video_processing)
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
        self.progress_dialog.reset()

    def open_options_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Options")
        tabs = QTabWidget()

        # --- Visuals Tab ---
        visuals_tab = QWidget()
        visuals_layout = QVBoxLayout()
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
        chk_draw_circle.setChecked(getattr(self, 'draw_petri_circle', False))
        visuals_layout.addWidget(chk_draw_circle)

        for chk in [chk_blank, chk_boxes, chk_labels, chk_arrows, chk_frame_counter]:
            visuals_layout.addWidget(chk)

        visuals_tab.setLayout(visuals_layout)
        tabs.addTab(visuals_tab, "Visuals")

        # --- Timing Tab ---
        timing_tab = QWidget()
        timing_layout = QVBoxLayout()
        
        # Time range controls
        time_range_group = QWidget()
        time_range_layout = QFormLayout()
        self.start_time_edit = QLineEdit(str(self.start_time_min))  
        self.end_time_edit = QLineEdit(str(self.end_time_min) if self.end_time_min is not None else "")  
        time_range_layout.addRow("Start Time (min):", self.start_time_edit)
        time_range_layout.addRow("End Time (min):", self.end_time_edit)
        time_range_group.setLayout(time_range_layout)
        timing_layout.addWidget(QLabel("Time Range:"))
        timing_layout.addWidget(time_range_group)

        timing_label = QLabel("Edge Persistence (seconds):")
        self.edge_time_slider = QSlider(Qt.Horizontal)
        self.edge_time_slider.setRange(0, 60)
        self.edge_time_slider.setValue(self.edge_persistence_seconds)
        self.edge_time_label = QLabel(f"{self.edge_persistence_seconds} s")
        self.edge_time_slider.valueChanged.connect(lambda v: self.edge_time_label.setText(f"{v} s"))
        timing_layout.addWidget(timing_label)
        timing_layout.addWidget(self.edge_time_slider)
        timing_layout.addWidget(self.edge_time_label)
        
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


        self.min_duration_label = QLabel("Min Interaction Duration (frames):")
        self.min_duration_input = QLineEdit("0")
        timing_layout.addWidget(self.min_duration_label)
        timing_layout.addWidget(self.min_duration_input)

        self.color_code_edges_checkbox = QCheckBox("Color-code edges by duration")
        self.color_code_edges_checkbox.setChecked(False)
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

        layout = QVBoxLayout()
        layout.addWidget(tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

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


            try:
                self.min_edge_duration = int(self.min_duration_input.text())
            except ValueError:
                self.min_edge_duration = 0

            self.color_code_edges = self.color_code_edges_checkbox.isChecked()

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
                
                for i in reversed(range(self.file_list_layout.count())): 
                    self.file_list_layout.itemAt(i).widget().setParent(None)
                self.file_checkboxes.clear()
                
                for idx, file_path in enumerate(file_paths):
                    df = pd.read_csv(file_path, usecols=["pos x", "pos y", "ori"])
                    fly_id = os.path.splitext(os.path.basename(file_path))[0]
                    df["fly_id"] = fly_id
                    all_data.append(df)
                    self.fly_colors[fly_id] = generate_fly_color(fly_id)
                    
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
                    
                self.all_flies_df = pd.concat(all_data, ignore_index=True)
                if self.edgelist_path:
                    self.video_button.setEnabled(True)
                    
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSVs:\n{str(e)}")
                
    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Background Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.video_path = video_path
            video_name = os.path.basename(video_path)
            chk = QCheckBox(f"Video: {video_name}")
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
            
            if self.all_flies_df is not None and self.edgelist_path:
                self.video_button.setEnabled(True)
                
    def load_edgelist(self):
        edgelist_path, _ = QFileDialog.getOpenFileName(self, "Select Edgelist CSV", "", "CSV Files (*.csv)")
        if edgelist_path:
            self.edgelist_path = edgelist_path
            edgelist_name = os.path.basename(edgelist_path)
            chk = QCheckBox(f"Edgelist: {edgelist_name}")
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
            
            if self.all_flies_df is not None:
                self.video_button.setEnabled(True)
                
    def load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(self)
        if path:
            try:
                with open(path, "rb") as f:
                    config = tomllib.load(f)
                
                self.calibration_values.update({
                    'min_x': config['min_x'],
                    'min_y': config['min_y'],
                    'x_px_ratio': config['x_px_ratio'],
                    'y_px_ratio': config['y_px_ratio']
                })
                self.calibration_path = path
                
                cal_name = os.path.basename(path)
                chk = QCheckBox(f"Calibration: {cal_name}")
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
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load calibration:\n{str(e)}")

    def closeEvent(self, event):
        if hasattr(self, 'video_popup'):
            try:
                self.video_popup.cleanup()
                self.video_popup.close()
            except:
                pass
        event.accept()
    def generate_video(self):
        if self.all_flies_df is None or not self.edgelist_path:
            QMessageBox.warning(self, "Missing Inputs", "Please load fly CSVs and edgelist before generating the video.")
            return
        checked_ids = [fly_id for fly_id, chk in self.file_checkboxes.items() if chk.isChecked()]
        filtered_df = self.all_flies_df[self.all_flies_df['fly_id'].isin(checked_ids)]
        
        if len(filtered_df) == 0:
            QMessageBox.warning(self, "No Files Selected", "Please check at least one fly CSV to include in the video.")
            return
        
        if not self.video_path:
            use_blank = True
            QMessageBox.information(self, "No Video", "No background video loaded. Using blank white background.")
        else:
            use_blank = self.use_blank_background

        final_draw_boxes = True
        final_draw_arrows = self.draw_arrows
        final_draw_boxes = self.draw_boxes
        min_duration = getattr(self, 'min_edge_duration', 0)
        color_edges = getattr(self, 'color_code_edges', False)


        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        self.video_thread = VideoProcessingThread(
            self.all_flies_df, self.video_path, self.edgelist_path, self.fly_colors,
            use_blank=use_blank,
            draw_boxes=final_draw_boxes,
            draw_labels=self.show_labels,
            draw_arrows=final_draw_arrows,
            show_frame_counter=self.show_frame_counter,
            scale_factor=self.scale_factor,
            edge_persistence_seconds=self.edge_persistence_seconds,
            save_graphs=self.enable_screenshot_saving,
            graph_interval_min=self.screenshot_interval_min,
            start_time_min=self.start_time_min,
            end_time_min=self.end_time_min,
            fly_size=13,
            draw_petri_circle=self.draw_petri_circle,
            min_edge_duration=min_duration,
            color_code_edges=color_edges,
            calibration_values=self.calibration_values) 

        self.video_thread.update_progress.connect(self.show_progress)
        self.video_thread.update_progress_bar.connect(self.progress_dialog.setValue)
        self.video_thread.video_saved.connect(self.show_video_saved)
        self.video_thread.finished.connect(self.on_video_thread_finished)
        self.video_thread.start()

    def cancel_video_processing(self):
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.cancel()

    def on_video_thread_finished(self):
        self.progress_dialog.hide()

    def show_progress(self, message):
        QMessageBox.information(self, "Progress", message)

    def show_video_saved(self, output_path, elapsed_time):
        self.progress_dialog.setValue(100)
        self.progress_dialog.hide()
        self.generated_video_path = output_path
        self.play_button.setEnabled(True)
        QMessageBox.information(self, "Success", f"Video saved to: {output_path}\nTime taken: {elapsed_time / 60:.2f} minutes")

    def play_embedded_video(self):
        if not hasattr(self, 'generated_video_path') or not os.path.exists(self.generated_video_path):
            QMessageBox.warning(self, "Error", "Generated video not found.")
            return

        # Close existing window if open
        if hasattr(self, 'video_popup'):
            try:
                self.video_popup.cleanup()
                self.video_popup.close()
                self.video_popup.deleteLater()
            except Exception as e:
                print(f"Error closing existing video window: {str(e)}")
            finally:
                if hasattr(self, 'video_popup'):
                    del self.video_popup

        # Create new window
        try:
            self.video_popup = VideoPlayerWindow(self.generated_video_path)
            self.video_popup.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open video player: {str(e)}")


if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/path/to/your/qt/plugins/platforms"  
    app = QApplication(sys.argv)
    setup_dark_theme(app)
    app.setStyle("Fusion")
    window = CSVFilterApp()
    window.show()
    sys.exit(app.exec_())

