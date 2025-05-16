import sys
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import csv
import os

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QTextEdit, QMessageBox, QProgressDialog,
    QHBoxLayout, QStyle, QSlider, QLabel
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from collections import defaultdict, deque


# ===== Constants =====
PREDEFINED_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128),
    (0, 128, 128), (0, 0, 128), (128, 128, 0), (128, 0, 0),
    (0, 128, 0), (199, 21, 133), (210, 105, 30), (255, 192, 203),
    (70, 130, 180), (154, 205, 50), (139, 69, 19), (75, 0, 130),
]


class VideoProcessingThread(QThread):
    update_progress = pyqtSignal(str)
    update_progress_bar = pyqtSignal(int)
    video_saved = pyqtSignal(str, float)

    def __init__(self, all_flies_df, video_path, edgelist_path=None, fly_colors=None):
        super().__init__()
        self.all_flies_df = all_flies_df
        self.video_path = video_path
        self.edgelist_path = edgelist_path
        self.fly_colors = fly_colors or {}

    def parse_edgelist(self, path):
        interactions = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    start = int(row["start_of_interaction"])
                    end = int(row["end_of_interaction"])
                    node_1 = row["node_1"]
                    node_2 = row["node_2"]
                    interactions.append((start, end, node_1, node_2))
                except Exception as e:
                    print(f"Skipping row due to error: {e}")
        return sorted(interactions, key=lambda x: x[0])

    def get_video_info(self, cap):
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return width, height, fps, total_frames

    def transform_fly_positions(self, frame_width, frame_height):
        fly_data = {
            fly_id: df.reset_index(drop=True)
            for fly_id, df in self.all_flies_df.groupby("fly_id")
        }

        all_x = self.all_flies_df["pos x"].values
        all_y = self.all_flies_df["pos y"].values
        min_x, max_x = all_x.min(), all_x.max()
        min_y, max_y = all_y.min(), all_y.max()

        scale = 0.9 * min(frame_width / (max_x - min_x), frame_height / (max_y - min_y))
        offset_x = (frame_width - scale * (max_x - min_x)) / 2
        offset_y = (frame_height - scale * (max_y - min_y)) / 2
        center_x, center_y = frame_width / 2, frame_height / 2

        transformed = {}
        for fly_id, df in fly_data.items():
            x = (df["pos x"].values - min_x) * scale + offset_x
            y = (df["pos y"].values - min_y) * scale + offset_y
            ori = df["ori"].values
            x += np.sign(center_x - x) * 35
            y += np.sign(center_y - y) * 35
            transformed[fly_id] = np.stack([x, y, ori], axis=1)

        return transformed, min(len(df) for df in fly_data.values())

    def draw_fly_arrows(self, frame, frame_idx, transformed_positions):
        for fly_id, coords in transformed_positions.items():
            if frame_idx >= len(coords):
                continue
            x, y, ori = coords[frame_idx]
            x, y = int(x), int(y)
            dx = int(40 * np.cos(ori))
            dy = int(-40 * np.sin(ori))
            color = self.fly_colors.get(fly_id, (255, 255, 255))
            cv2.arrowedLine(frame, (x, y), (x + dx, y + dy), color, 10, tipLength=0.3)
            cv2.putText(frame, str(fly_id), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


    def draw_interactions(self, frame, frame_idx, interactions, transformed_positions):
        adjacency = defaultdict(set)
        for (start, end, fly1, fly2) in interactions:
            if start <= frame_idx <= end:
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
            group_list = list(group)
            for i in range(len(group_list)):
                fly_id = group_list[i]
                if fly_id in transformed_positions and frame_idx < len(transformed_positions[fly_id]):
                    x1, y1, _ = transformed_positions[fly_id][frame_idx]
                    xs.append(x1)
                    ys.append(y1)
                    for j in range(i + 1, len(group_list)):
                        fly2 = group_list[j]
                        if fly2 in transformed_positions and frame_idx < len(transformed_positions[fly2]):
                            x2, y2, _ = transformed_positions[fly2][frame_idx]
                            pt1 = (int(x1), int(y1))
                            pt2 = (int(x2), int(y2))
                            cv2.arrowedLine(frame, pt1, pt2, (0, 0, 0), 4, tipLength=0.1)

            if xs and ys:
                margin = 60
                min_x_box = max(0, int(min(xs) - margin))
                max_x_box = min(frame.shape[1], int(max(xs) + margin))
                min_y_box = max(0, int(min(ys) - margin))
                max_y_box = min(frame.shape[0], int(max(ys) + margin))
                cv2.rectangle(frame, (min_x_box, min_y_box), (max_x_box, max_y_box), (0, 255, 255), 3)



    def run(self):
        try:
            start_time = time.time()
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.update_progress.emit("Failed to open video file.")
                return

            frame_width, frame_height, fps, total_frames = self.get_video_info(cap)
            out = cv2.VideoWriter("overlayed_fly_video_sorted.mp4",
                                  cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

            transformed_positions, max_data_len = self.transform_fly_positions(frame_width, frame_height)
            max_frames = min(max_data_len, total_frames)
            interactions = self.parse_edgelist(self.edgelist_path) if self.edgelist_path else []

            frame_idx = 0
            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                self.draw_fly_arrows(frame, frame_idx, transformed_positions)
                self.draw_interactions(frame, frame_idx, interactions, transformed_positions)
                out.write(frame)
                self.update_progress_bar.emit(int((frame_idx / max_frames) * 100))
                frame_idx += 1

            cap.release()
            out.release()
            self.video_saved.emit("overlayed_fly_video_sorted.mp4", time.time() - start_time)

        except Exception as e:
            self.update_progress.emit(f"Error: {str(e)}")

class VideoPlayerWindow(QWidget):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle("Video Player")
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Video widget
        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget)

        # Control layout for buttons and playback bar
        control_layout = QHBoxLayout()

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_video)

        self.pause_button = QPushButton()
        self.pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pause_button.clicked.connect(self.pause_video)

        self.stop_button = QPushButton()
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop_video)


        self.fullscreen_button = QPushButton("Fullscreen")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
    
        self.skip_back_button = QPushButton("<<")
        self.skip_back_button.clicked.connect(self.skip_back)

        self.skip_forward_button = QPushButton(">>")
        self.skip_forward_button.clicked.connect(self.skip_forward)

        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.skip_back_button)
        control_layout.addWidget(self.skip_forward_button)
        control_layout.addWidget(self.fullscreen_button)

        # Add control layout to the main layout
        layout.addLayout(control_layout)

        # Add playback slider (QSlider)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.seek_video)
        layout.addWidget(self.slider)

        # Media player
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)

        # Connect to media player signals
        self.media_player.positionChanged.connect(self.update_slider_position)
        self.media_player.durationChanged.connect(self.update_slider_range)

        # Load the video
        url = QUrl.fromLocalFile(os.path.abspath(video_path))
        self.media_player.setMedia(QMediaContent(url))

    def play_video(self):
        self.media_player.play()

    def pause_video(self):
        self.media_player.pause()

    def stop_video(self):
        self.media_player.stop()

    def skip_back(self):
        # Skip backward 5 seconds
        current_pos = self.media_player.position()
        new_pos = max(0, current_pos - 5000)  
        self.media_player.setPosition(new_pos)

    def skip_forward(self):
        # Skip forward 5 seconds
        current_pos = self.media_player.position()
        new_pos = min(self.media_player.duration(), current_pos + 5000)  
        self.media_player.setPosition(new_pos)

    def seek_video(self, position):
        # Jump to the selected position in the slider
        self.media_player.setPosition(position)

    def update_slider_position(self, position):
        # Update the slider position during playback
        self.slider.setValue(position)

    def update_slider_range(self, duration):
        # Update slider range when the video is loaded
        self.slider.setRange(0, duration)

    def toggle_fullscreen(self):
            if self.isFullScreen():
                self.showNormal()  # Go back to windowed mode
            else:
                self.showFullScreen()  # Switch to fullscreen mode
class CSVFilterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fly Trajectory Visualizer")
        self.resize(800, 600)

        self.layout = QVBoxLayout()

        # Horizontal layout for 2 columns
        button_table_layout = QHBoxLayout()

        # Left column (3 buttons)
        left_column = QVBoxLayout()
        self.load_button = QPushButton("Load Fly CSVs")
        self.load_button.clicked.connect(self.load_csv)
        left_column.addWidget(self.load_button)

        self.load_video_button = QPushButton("Load Background Video")
        self.load_video_button.clicked.connect(self.load_video)
        left_column.addWidget(self.load_video_button)

        self.load_edgelist_button = QPushButton("Load Edgelist CSV")
        self.load_edgelist_button.clicked.connect(self.load_edgelist)
        left_column.addWidget(self.load_edgelist_button)

        # Right column (3 buttons)
        right_column = QVBoxLayout()
        self.plot_button = QPushButton("Plot Fly Paths")
        self.plot_button.clicked.connect(self.plot_fly_paths)
        self.plot_button.setEnabled(False)
        right_column.addWidget(self.plot_button)

        self.video_button = QPushButton("Generate Video")
        self.video_button.clicked.connect(self.generate_video)
        self.video_button.setEnabled(False)
        right_column.addWidget(self.video_button)

        self.play_button = QPushButton("Play Video")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_embedded_video)
        right_column.addWidget(self.play_button)

        # Combine both columns in the horizontal layout
        button_table_layout.addLayout(left_column)
        button_table_layout.addLayout(right_column)

        self.layout.addLayout(button_table_layout)

        # Text area for data preview
        self.text_preview = QTextEdit()
        self.text_preview.setReadOnly(True)
        self.layout.addWidget(self.text_preview)

        self.setLayout(self.layout)

        # Internal state
        self.all_flies_df = None
        self.video_path = None
        self.edgelist_path = None
        self.generated_video_path = None
        self.fly_colors = {}

        # Progress dialog
        self.progress_dialog = QProgressDialog("Processing video...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Video Progress")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.reset()


    def load_csv(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Fly CSV Files", "", "CSV Files (*.csv)")
        if file_paths:
            try:
                all_data = []
                self.fly_colors.clear()
                for idx, file_path in enumerate(file_paths):
                    df = pd.read_csv(file_path, usecols=["pos x", "pos y", "ori"], nrows=400)
                    fly_id = os.path.splitext(os.path.basename(file_path))[0]
                    df["fly_id"] = fly_id
                    all_data.append(df)
                    self.fly_colors[fly_id] = PREDEFINED_COLORS[idx % len(PREDEFINED_COLORS)]

                self.all_flies_df = pd.concat(all_data, ignore_index=True)

                if self.video_path and self.edgelist_path:
                    self.video_button.setEnabled(True)
                self.plot_button.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSVs:\n{str(e)}")


    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Background Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.video_path = video_path
            if self.all_flies_df is not None and self.edgelist_path:
                self.video_button.setEnabled(True)

    def load_edgelist(self):
        edgelist_path, _ = QFileDialog.getOpenFileName(self, "Select Edgelist CSV", "", "CSV Files (*.csv)")
        if edgelist_path:
            self.edgelist_path = edgelist_path
            if self.all_flies_df is not None and self.video_path:
                self.video_button.setEnabled(True)

    def plot_fly_paths(self):
        if self.all_flies_df is None:
            return

        try:
            plt.figure(figsize=(8, 6))
            unique_flies = self.all_flies_df["fly_id"].unique()
            for fly_id in unique_flies:
                fly_df = self.all_flies_df[self.all_flies_df["fly_id"] == fly_id]
                color = np.array(self.fly_colors.get(fly_id, (0, 0, 0))) / 255.0
                plt.plot(fly_df["pos x"], fly_df["pos y"], marker="o", markersize=2,
                         linestyle='-', label=fly_id, color=color)

            plt.title("Fly Trajectories")
            plt.xlabel("Position X")
            plt.ylabel("Position Y")
            plt.gca().invert_yaxis()
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to plot fly paths:\n{str(e)}")

    def generate_video(self):
        if self.all_flies_df is None or not self.video_path or not self.edgelist_path:
            QMessageBox.warning(self, "Missing Inputs", "Please load fly CSVs, video, and edgelist before generating the video.")
            return

        self.progress_dialog.setValue(0)
        self.progress_dialog.show()

        self.video_thread = VideoProcessingThread(
            self.all_flies_df, self.video_path, self.edgelist_path, self.fly_colors
        )
        self.video_thread.update_progress.connect(self.show_progress)
        self.video_thread.update_progress_bar.connect(self.progress_dialog.setValue)
        self.video_thread.video_saved.connect(self.show_video_saved)
        self.video_thread.start()

    def show_progress(self, message):
        QMessageBox.information(self, "Progress", message)

    def show_video_saved(self, output_path, elapsed_time):
        self.progress_dialog.setValue(100)
        self.progress_dialog.hide()
        self.generated_video_path = output_path
        self.play_button.setEnabled(True)
        QMessageBox.information(self, "Success", f"Video saved to: {output_path}\nTime taken: {elapsed_time / 60:.2f} minutes")

    def play_embedded_video(self):
        if self.generated_video_path and os.path.exists(self.generated_video_path):
            self.video_popup = VideoPlayerWindow(self.generated_video_path)
            self.video_popup.show()
        else:
            QMessageBox.warning(self, "Error", "Generated video not found.")


if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/path/to/your/qt/plugins/platforms"  # update only if needed
    app = QApplication(sys.argv)
    window = CSVFilterApp()
    window.show()
    sys.exit(app.exec_())