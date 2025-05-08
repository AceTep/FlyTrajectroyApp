import sys
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QTextEdit, QMessageBox
)


class CSVFilterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Filter Tool")
        self.resize(600, 400)

        self.layout = QVBoxLayout()

        self.load_button = QPushButton("Load CSVs")
        self.load_button.clicked.connect(self.load_csv)
        self.layout.addWidget(self.load_button)

        self.text_preview = QTextEdit()
        self.text_preview.setReadOnly(True)
        self.layout.addWidget(self.text_preview)

        self.save_button = QPushButton("Save Filtered CSV")
        self.save_button.clicked.connect(self.save_csv)
        self.save_button.setEnabled(False)
        self.layout.addWidget(self.save_button)

        self.plot_button = QPushButton("Plot Fly Paths")
        self.plot_button.clicked.connect(self.plot_fly_paths)
        self.plot_button.setEnabled(False)
        self.layout.addWidget(self.plot_button)

        self.video_button = QPushButton("Generate Video")
        self.video_button.clicked.connect(self.generate_video)
        self.video_button.setEnabled(False)
        self.layout.addWidget(self.video_button)

        self.setLayout(self.layout)
        self.filtered_df = None
        self.all_flies_df = None  # To store data from all flies

    def load_csv(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Open CSV Files", "", "CSV Files (*.csv)"
        )
        if file_paths:
            try:
                all_data = []
                for file_path in file_paths:
                    df = pd.read_csv(file_path, usecols=["pos x", "pos y", "ori"], nrows=1000)
                    fly_id = file_path.split("/")[-1].split(".")[0]
                    df["fly_id"] = fly_id
                    all_data.append(df)

                self.all_flies_df = pd.concat(all_data, ignore_index=True)
                self.text_preview.setPlainText(str(self.all_flies_df.head()))

                self.save_button.setEnabled(True)
                self.plot_button.setEnabled(True)
                self.video_button.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSVs:\n{str(e)}")

    def save_csv(self):
        if self.all_flies_df is None:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Filtered CSV", "", "CSV Files (*.csv)"
        )
        if file_path:
            try:
                self.all_flies_df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", "Filtered CSV saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save CSV:\n{str(e)}")

    def plot_fly_paths(self):
        if self.all_flies_df is None:
            return

        try:
            plt.figure(figsize=(8, 6))
            unique_flies = self.all_flies_df["fly_id"].unique()
            for fly_id in unique_flies:
                fly_df = self.all_flies_df[self.all_flies_df["fly_id"] == fly_id]
                plt.plot(fly_df["pos x"], fly_df["pos y"], marker="o", markersize=2, linestyle='-', label=fly_id)

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
        if self.all_flies_df is None:
            return

        try:
            video_path, _ = QFileDialog.getOpenFileName(self, "Select Background Video", "", "Video Files (*.mp4 *.avi *.mov)")
            if not video_path:
                return

            start_time = time.time()

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open video file.")
                return

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_path = "overlayed_fly_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            fly_data = {
                fly_id: df.reset_index(drop=True)
                for fly_id, df in self.all_flies_df.groupby("fly_id")
            }

            min_rows = min(len(df) for df in fly_data.values())
            max_frames = min(min_rows, total_frames)

            # Bounding box for scaling
            all_x = self.all_flies_df["pos x"].values
            all_y = self.all_flies_df["pos y"].values
            min_x, max_x = all_x.min(), all_x.max()
            min_y, max_y = all_y.min(), all_y.max()

            bbox_width = max_x - min_x
            bbox_height = max_y - min_y

            scale = 0.9 * min(frame_width / bbox_width, frame_height / bbox_height)
            offset_x = (frame_width - scale * bbox_width) / 2
            offset_y = (frame_height - scale * bbox_height) / 2

            # Precompute transformed positions for all flies
            transformed_positions = {
                fly_id: np.stack([
                    ((df["pos x"].values - min_x) * scale + offset_x),
                    ((df["pos y"].values - min_y) * scale + offset_y),
                    df["ori"].values
                ], axis=1)
                for fly_id, df in fly_data.items()
            }

            # Assign random colors to each fly
            fly_colors = {
                fly_id: tuple(int(c) for c in np.random.randint(0, 255, 3))
                for fly_id in fly_data.keys()
            }

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
                    color = fly_colors[fly_id]

                    cv2.circle(frame, (x, y), 10, color, -1)
                    cv2.line(frame, (x, y), (x + dx, y + dy), (255, 255, 0), 2)

                    # Optional: enable to draw fly ID text 
                    # cv2.putText(frame, fly_id, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                out.write(frame)
                frame_idx += 1

            cap.release()
            out.release()
            elapsed_time = time.time() - start_time
            QMessageBox.information(self, "Success", f"Video saved to: {output_path}\nTime taken: {elapsed_time:.2f} seconds")

        except Exception as e:
            QMessageBox.critical(self, "Video Error", f"Failed to generate overlay video:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSVFilterApp()
    window.show()
    sys.exit(app.exec_())
