import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton,
    QHBoxLayout, QSlider, QLabel, QFrame, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from app.widgets import create_button
from utils.theme import (
    DARK_GRAY, MEDIUM_GRAY, LIGHT_GRAY, ACCENT_COLOR
)
from app.fly_grid import FlyGridWindow  


class VideoPlayerWindow(QWidget):
    def __init__(self, video_path, fly_positions):
        super().__init__()
        self.setWindowTitle("Video Player")
        self.resize(800, 600)
        self.setStyleSheet(f"background-color: {DARK_GRAY};")

        # Load video
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        # Extract basic video info
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_ms = int(self.total_frames * 1000 / self.fps)

        self.frame_idx = 0
        self.paused = True
        self.fly_positions = fly_positions

        # Layout setup
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Display area for video frames
        self.video_label = QLabel()
        self.video_label.setStyleSheet(f"background-color: {DARK_GRAY};")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label, stretch=1)

        # Playback control panel
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

        # Create media control buttons
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QPushButton().style().SP_MediaPlay))
        self.play_button.setStyleSheet("QPushButton { padding: 8px; }")
        self.play_button.clicked.connect(self.play)

        self.pause_button = QPushButton()
        self.pause_button.setIcon(self.style().standardIcon(QPushButton().style().SP_MediaPause))
        self.pause_button.setStyleSheet("QPushButton { padding: 8px; }")
        self.pause_button.clicked.connect(self.pause)

        self.stop_button = QPushButton()
        self.stop_button.setIcon(self.style().standardIcon(QPushButton().style().SP_MediaStop))
        self.stop_button.setStyleSheet("QPushButton { padding: 8px; }")
        self.stop_button.clicked.connect(self.stop)

        self.fullscreen_button = create_button("Fullscreen")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)

        # Button to toggle fly grid overlay window
        self.fly_grid_button = create_button("Show Fly Grid")
        self.fly_grid_button.setCheckable(True)
        self.fly_grid_button.clicked.connect(self.toggle_fly_grid)
        control_layout.addWidget(self.fly_grid_button)

        # Buttons to skip 5 seconds back and forward
        self.skip_back_button = create_button("<< 5s")
        self.skip_back_button.clicked.connect(self.skip_back)

        self.skip_forward_button = create_button(">> 5s")
        self.skip_forward_button.clicked.connect(self.skip_forward)

        # Add all buttons to layout
        for btn in [self.play_button, self.pause_button, self.stop_button,
                    self.skip_back_button, self.skip_forward_button, self.fullscreen_button]:
            control_layout.addWidget(btn)

        layout.addWidget(control_panel)

        # Timeline slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.duration_ms)
        self.slider.sliderReleased.connect(self.seek_video)
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
        layout.addWidget(self.slider)

        # Frame update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

    # Playback functions
    def play(self):
        if not self.timer.isActive():
            self.timer.start(int(1000 / self.fps))
        self.paused = False
        if hasattr(self, 'fly_grid_window'):
            self.fly_grid_window.start()

    def pause(self):
        self.timer.stop()
        self.paused = True
        if hasattr(self, 'fly_grid_window'):
            self.fly_grid_window.pause()

    def stop(self):
        self.pause()
        self.frame_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        self.next_frame()
        if hasattr(self, 'fly_grid_window'):
            self.fly_grid_window.stop()

    def skip_back(self):
        # Rewind by 5 seconds
        self.frame_idx = max(0, self.frame_idx - int(5 * self.fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        self.next_frame()
        if hasattr(self, 'fly_grid_window'):
            self.fly_grid_window.set_frame_index(self.frame_idx)

    def skip_forward(self):
        # Skip ahead 5 seconds
        self.frame_idx = min(self.total_frames - 1, self.frame_idx + int(5 * self.fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        self.next_frame()
        if hasattr(self, 'fly_grid_window'):
            self.fly_grid_window.set_frame_index(self.frame_idx)

    def seek_video(self):
        # Jump to selected time via slider
        ms = self.slider.value()
        self.frame_idx = int((ms / 1000) * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        self.next_frame()
        if hasattr(self, 'fly_grid_window'):
            self.fly_grid_window.set_frame_index(self.frame_idx)

    def toggle_fly_grid(self, checked):
        # Show/hide fly grid overlay
        if checked:
            self.fly_grid_button.setText("Hide Fly Grid")
            self.show_fly_grid()
        else:
            self.fly_grid_button.setText("Show Fly Grid")
            self.hide_fly_grid()

    def show_fly_grid(self):
        # Create and show fly grid window
        if not hasattr(self, 'fly_grid_window'):
            zoom = getattr(self.parent(), 'use_small_grid_zoom', False)
            self.fly_grid_window = FlyGridWindow(self.fly_positions, self.frame_idx, small_zoom=zoom)

        if hasattr(self, 'current_frame'):
            self.fly_grid_window.update_from_frame(self.current_frame, self.frame_idx)
        else:
            blank = np.ones((480, 640, 3), dtype=np.uint8) * 255
            self.fly_grid_window.update_from_frame(blank, self.frame_idx)

        self.fly_grid_window.show()

    def hide_fly_grid(self):
        # Close fly grid window
        if hasattr(self, 'fly_grid_window'):
            self.fly_grid_window.close()
            del self.fly_grid_window

    def closeEvent(self, event):
        # Cleanup on close
        self.pause()
        if self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'fly_grid_window'):
            self.fly_grid_window.close()
        event.accept()

    def next_frame(self):
        # Show the next frame
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        self.frame_idx += 1
        self.current_frame = frame
        current_ms = int((self.frame_idx / self.fps) * 1000)
        self.slider.setValue(current_ms)

        # Convert to Qt-compatible format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        if hasattr(self, 'fly_grid_window'):
            self.fly_grid_window.update_from_frame(frame, self.frame_idx)

    def toggle_fullscreen(self):
        # Toggle fullscreen mode
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def resizeEvent(self, event):
        # Update frame display on resize
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            self.cap.grab()
            ret, frame = self.cap.retrieve()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                    self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
        super().resizeEvent(event)
