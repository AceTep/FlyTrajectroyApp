import sys
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import csv
import os
import traceback
from collections import defaultdict, deque

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QTextEdit, QMessageBox, QProgressDialog,
    QHBoxLayout, QStyle, QSlider, QLabel, QDialog,
    QCheckBox, QDialogButtonBox
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import hashlib

def generate_fly_color(fly_id):
    # Hash the fly_id to a consistent 6-digit hex
    digest = hashlib.md5(fly_id.encode()).hexdigest()
    r = int(digest[0:2], 16)
    g = int(digest[2:4], 16)
    b = int(digest[4:6], 16)
    return (r, g, b)

class VideoProcessingThread(QThread):
    update_progress = pyqtSignal(str)
    update_progress_bar = pyqtSignal(int)
    video_saved = pyqtSignal(str, float)

    def __init__(self, all_flies_df, video_path, edgelist_path=None, fly_colors=None,
             use_blank=False, draw_boxes=True, draw_labels=True, draw_arrows=True,
             show_frame_counter=True):

        super().__init__()
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
                    node_1 = row["node_1"]
                    node_2 = row["node_2"]
                    interactions.append((start, end, node_1, node_2))
                except Exception as e:
                    print(f"Skipping row: {e}")
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
                frame_width, frame_height = 3052,2304  # Default size for blank background
                fps = 30
                # Estimate frame count by minimum fly data length
                transformed_positions, max_data_len = self.transform_fly_positions(frame_width, frame_height)
                total_frames = max_data_len
            else:
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    self.update_progress.emit("Failed to open video.")
                    return
                frame_width, frame_height, fps, total_frames = self.get_video_info(cap)
                transformed_positions, max_data_len = self.transform_fly_positions(frame_width, frame_height)

            base_name = os.path.splitext(os.path.basename(self.video_path if self.video_path else "blank"))[0]
            output_filename = f"{base_name}_overlay_fly.mp4"
            out = cv2.VideoWriter(
                output_filename,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (frame_width, frame_height)
            )

            if not out.isOpened():
                self.update_progress.emit("Failed to create output video file.")
                return

            max_frames = min(max_data_len, total_frames)
            interactions = self.parse_edgelist(self.edgelist_path) if self.edgelist_path else []

            frame_idx = 0
            while frame_idx < max_frames:
                if self._is_cancelled:
                    break

                if self.use_blank:
                    frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
                    ret = True
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break

                self.draw_flies(frame, frame_idx, transformed_positions)
                if self.draw_boxes:
                    self.draw_interaction_groups(frame, frame_idx, interactions, transformed_positions)

                if self.show_frame_counter:
                    self.draw_frame_counter(frame, frame_idx)

                out.write(frame)
                self.update_progress_bar.emit(int((frame_idx / max_frames) * 100))
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
        position = (20, 40)  # Top-left corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (0, 0, 0) if self.use_blank else (255, 255, 255)
        thickness = 2
        cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)



    def draw_flies(self, frame, frame_idx, transformed_positions):
        label_color = (0, 0, 0) if self.use_blank else (255, 255, 255)
        for fly_id, coords in transformed_positions.items():
            if frame_idx >= len(coords):
                continue
            x, y, ori = coords[frame_idx]
            x, y = int(x), int(y)
            color = self.fly_colors.get(fly_id, (255, 255, 255))
            if self.draw_arrows:
                dx = int(40 * np.cos(ori))
                dy = int(-40 * np.sin(ori))
                cv2.arrowedLine(frame, (x, y), (x + dx, y + dy), color, 10, tipLength=0.3)
            else:
                cv2.circle(frame, (x, y), 10, color, -1)
            if self.draw_labels:
                cv2.putText(frame, str(fly_id), (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)


    def draw_interaction_groups(self, frame, frame_idx, interactions, transformed_positions):
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
                            cv2.arrowedLine(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 4, tipLength=0.1)

            if xs and ys:
                margin = 60
                min_x_box = max(0, int(min(xs) - margin))
                max_x_box = min(frame.shape[1], int(max(xs) + margin))
                min_y_box = max(0, int(min(ys) - margin))
                max_y_box = min(frame.shape[0], int(max(ys) + margin))
                group_size = len(group)
                if group_size == 2:
                    box_color = (0, 128, 0)  #  Green
                elif group_size == 3:
                    box_color = (0, 255, 255)  # Yellow
                else:
                    box_color = (0, 0, 255)    # Red

                cv2.rectangle(frame, (min_x_box, min_y_box), (max_x_box, max_y_box), box_color, 3)


class VideoPlayerWindow(QWidget):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle("Video Player")
        self.resize(800, 600)
        layout = QVBoxLayout(self)
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
        self.fullscreen_button = QPushButton("Fullscreen")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.skip_back_button = QPushButton("<<")
        self.skip_back_button.clicked.connect(self.skip_back)
        self.skip_forward_button = QPushButton(">>")
        self.skip_forward_button.clicked.connect(self.skip_forward)

        for btn in [self.play_button, self.pause_button, self.stop_button, self.skip_back_button, self.skip_forward_button, self.fullscreen_button]:
            control_layout.addWidget(btn)

        layout.addLayout(control_layout)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self.seek_video)
        layout.addWidget(self.slider)

        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.positionChanged.connect(self.update_slider_position)
        self.media_player.durationChanged.connect(self.update_slider_range)
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(video_path))))

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
        self.resize(800, 600)
        self.layout = QVBoxLayout(self)
        button_table_layout = QHBoxLayout()

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
        self.options_button = QPushButton("Options")
        self.options_button.clicked.connect(self.open_options_dialog)
        left_column.addWidget(self.options_button)

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

        button_table_layout.addLayout(left_column)
        button_table_layout.addLayout(right_column)
        self.layout.addLayout(button_table_layout)

        self.text_preview = QTextEdit()
        self.text_preview.setReadOnly(True)
        self.layout.addWidget(self.text_preview)

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


        self.progress_dialog = QProgressDialog("Processing video...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Video Progress")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.canceled.connect(self.cancel_video_processing)
        self.progress_dialog.reset()

    def open_options_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Options")
        layout = QVBoxLayout()
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
        layout.addWidget(chk_blank)
        layout.addWidget(chk_boxes)
        layout.addWidget(chk_labels)
        layout.addWidget(chk_arrows)
        layout.addWidget(chk_frame_counter)
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

    def load_csv(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Fly CSV Files", "", "CSV Files (*.csv)")
        if file_paths:
            try:
                all_data = []
                self.fly_colors.clear()
                for idx, file_path in enumerate(file_paths):
                    df = pd.read_csv(file_path, usecols=["pos x", "pos y", "ori"], nrows =500)
                    fly_id = os.path.splitext(os.path.basename(file_path))[0]
                    df["fly_id"] = fly_id
                    all_data.append(df)
                    self.fly_colors[fly_id] = generate_fly_color(fly_id)
                self.all_flies_df = pd.concat(all_data, ignore_index=True)
                if self.edgelist_path:
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
            if self.all_flies_df is not None:
                self.video_button.setEnabled(True)

    def plot_fly_paths(self):
        if self.all_flies_df is None:
            return
        try:
            plt.figure(figsize=(8, 6))
            for fly_id in self.all_flies_df["fly_id"].unique():
                fly_df = self.all_flies_df[self.all_flies_df["fly_id"] == fly_id]
                color = np.array(self.fly_colors.get(fly_id, (0, 0, 0))) / 255.0
                plt.plot(fly_df["pos x"], fly_df["pos y"], marker="o", markersize=2, linestyle='-', label=fly_id, color=color)
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
        if self.all_flies_df is None or not self.edgelist_path:
            QMessageBox.warning(self, "Missing Inputs", "Please load fly CSVs and edgelist before generating the video.")
            return

        if not self.video_path:
            use_blank = True
            QMessageBox.information(self, "No Video", "No background video loaded. Using blank white background.")
        else:
            use_blank = self.use_blank_background

        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        self.video_thread = VideoProcessingThread(
            self.all_flies_df, self.video_path, self.edgelist_path, self.fly_colors,
            use_blank=use_blank,
            draw_boxes=self.draw_boxes,
            draw_labels=self.show_labels,
            draw_arrows=self.draw_arrows,
            show_frame_counter=self.show_frame_counter
        )
        self.video_thread.update_progress.connect(self.show_progress)
        self.video_thread.update_progress_bar.connect(self.progress_dialog.setValue)
        self.video_thread.video_saved.connect(self.show_video_saved)
        self.video_thread.finished.connect(self.on_video_thread_finished)  # NEW
        self.video_thread.start()

    def cancel_video_processing(self):
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.cancel()

    def on_video_thread_finished(self):
        # Called when thread finishes, either normally or after cancel
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
