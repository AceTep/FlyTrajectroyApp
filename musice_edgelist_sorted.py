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
    QHBoxLayout, QStyle
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget


# Define a fixed list of 20 RGB color tuples
PREDEFINED_COLORS = [
    (255, 0, 0),       # Red
    (0, 255, 0),       # Green
    (0, 0, 255),       # Blue
    (255, 255, 0),     # Yellow
    (255, 0, 255),     # Magenta
    (0, 255, 255),     # Cyan
    (255, 165, 0),     # Orange
    (128, 0, 128),     # Purple
    (0, 128, 128),     # Teal
    (0, 0, 128),       # Navy
    (128, 128, 0),     # Olive
    (128, 0, 0),       # Maroon
    (0, 128, 0),       # Dark green
    (75, 0, 130),      # Indigo
    (199, 21, 133),    # Medium Violet Red
    (210, 105, 30),    # Chocolate
    (255, 192, 203),   # Pink
    (70, 130, 180),    # Steel Blue
    (154, 205, 50),    # Yellow Green
    (139, 69, 19)      # Saddle Brown
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
                except Exception:
                    continue
        return sorted(interactions, key=lambda x: x[0])

    def run(self):
        try:
            start_time = time.time()

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.update_progress.emit("Failed to open video file.")
                return

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_path = "overlayed_fly_video_sorted.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            fly_data = {
                fly_id: df.reset_index(drop=True)
                for fly_id, df in self.all_flies_df.groupby("fly_id")
            }

            min_rows = min(len(df) for df in fly_data.values())
            max_frames = min(min_rows, total_frames)

            all_x = self.all_flies_df["pos x"].values
            all_y = self.all_flies_df["pos y"].values
            min_x, max_x = all_x.min(), all_x.max()
            min_y, max_y = all_y.min(), all_y.max()

            bbox_width = max_x - min_x
            bbox_height = max_y - min_y

            scale = 0.9 * min(frame_width / bbox_width, frame_height / bbox_height)
            offset_x = (frame_width - scale * bbox_width) / 2
            offset_y = (frame_height - scale * bbox_height) / 2

            transformed_positions = {
                fly_id: np.stack([
                    ((df["pos x"].values - min_x) * scale + offset_x),
                    ((df["pos y"].values - min_y) * scale + offset_y),
                    df["ori"].values
                ], axis=1)
                for fly_id, df in fly_data.items()
            }

            interactions = self.parse_edgelist(self.edgelist_path) if self.edgelist_path else []

            frame_idx = 0
            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                for fly_id, coords in transformed_positions.items():
                    if frame_idx >= len(coords):
                        continue
                    x, y, ori = coords[frame_idx]
                    x = int(x)
                    y = int(y)
                    dx = int(15 * np.cos(ori))
                    dy = int(15 * np.sin(ori))
                    color = self.fly_colors.get(fly_id, (255, 255, 255))

                    cv2.circle(frame, (x, y), 25, color, -1)
                    cv2.line(frame, (x, y), (x + dx, y + dy), (255, 255, 0), 2)
                    cv2.putText(frame, str(fly_id), (x + 10, y - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                for (start, end, fly1, fly2) in interactions:
                    if start <= frame_idx <= end:
                        if fly1 in transformed_positions and fly2 in transformed_positions:
                            if frame_idx < len(transformed_positions[fly1]) and frame_idx < len(transformed_positions[fly2]):
                                x1, y1, _ = transformed_positions[fly1][frame_idx]
                                x2, y2, _ = transformed_positions[fly2][frame_idx]
                                pt1 = (int(x1), int(y1))
                                pt2 = (int(x2), int(y2))
                                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

                out.write(frame)
                progress = int((frame_idx / max_frames) * 100)
                self.update_progress_bar.emit(progress)
                frame_idx += 1

            cap.release()
            out.release()
            elapsed_time = time.time() - start_time
            self.video_saved.emit(output_path, elapsed_time)

        except Exception as e:
            self.update_progress.emit(f"Error: {str(e)}")


class VideoPlayerWindow(QWidget):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle("Video Player")
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget)

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

        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.stop_button)

        layout.addLayout(control_layout)

        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)

        url = QUrl.fromLocalFile(os.path.abspath(video_path))
        self.media_player.setMedia(QMediaContent(url))

    def play_video(self):
        self.media_player.play()

    def pause_video(self):
        self.media_player.pause()

    def stop_video(self):
        self.media_player.stop()


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
                    df = pd.read_csv(file_path, usecols=["pos x", "pos y", "ori"], nrows=1000)
                    fly_id = os.path.splitext(os.path.basename(file_path))[0]
                    df["fly_id"] = fly_id
                    all_data.append(df)
                    self.fly_colors[fly_id] = PREDEFINED_COLORS[idx % len(PREDEFINED_COLORS)]

                self.all_flies_df = pd.concat(all_data, ignore_index=True)
                self.text_preview.setPlainText(str(self.all_flies_df.head()))

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
