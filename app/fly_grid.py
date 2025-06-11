import cv2
import math

from PyQt5.QtCore import pyqtSignal, Qt, QObject
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,QGridLayout
)
from PyQt5.QtGui import QPixmap, QImage
from utils.theme import (
    DARK_GRAY
)

class FlyGridWorker(QObject):
    updated = pyqtSignal(dict)

    def __init__(self, fly_positions,crop_size=160):
        super().__init__()
        self.fly_positions = fly_positions
        self.crop_size = crop_size  

    def update_from_frame(self, frame, frame_idx):
        crops = {}
        for fly_id, positions in self.fly_positions.items():
            if frame_idx < len(positions):
                x, y, _ = positions[frame_idx]
                x, y = int(x), int(y)
                crop_size = 100
                x1 = max(0, x - crop_size)
                y1 = max(0, y - crop_size)
                x2 = min(frame.shape[1], x + crop_size)
                y2 = min(frame.shape[0], y + crop_size)
                fly_crop = frame[y1:y2, x1:x2]
                if fly_crop.size != 0:
                    fly_crop = cv2.resize(fly_crop, (self.crop_size, self.crop_size))
                    crops[fly_id] = fly_crop

        self.updated.emit(crops)
class FlyGridWindow(QWidget):
    def __init__(self, fly_positions, frame_idx, small_zoom=False):
        super().__init__()
        self.setWindowTitle("Fly Grid View")
        self.fly_positions = fly_positions
        self.crop_size = 50 if small_zoom else 160
        self.worker = FlyGridWorker(fly_positions, crop_size=self.crop_size)
        self.worker.updated.connect(self.update_grid)
        self.labels = {}
        self.active = True
        self.current_frame_idx = frame_idx

        self.setStyleSheet(f"""
            background-color: {DARK_GRAY};
            QLabel {{
                color: white;
                font-size: 12px;
            }}
        """)
        self.layout = QGridLayout(self)
        self.layout.setSpacing(10)

        sorted_fly_ids = sorted(
            fly_positions.keys(),
            key=lambda x: int(x[3:]) if x.lower().startswith('fly') and x[3:].isdigit() else x
        )

        num_flies = len(sorted_fly_ids)
        if num_flies == 12:
            rows, cols = 3, 4
        elif num_flies == 24:
            rows, cols = 4, 6
        else:
            cols = 4
            rows = math.ceil(num_flies / cols)


        for idx, fly_id in enumerate(sorted_fly_ids):
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(5)

            id_label = QLabel(fly_id)
            id_label.setAlignment(Qt.AlignCenter)
            id_label.setStyleSheet("font-weight: bold; color: white;")
            container_layout.addWidget(id_label)

            video_label = QLabel()
            video_label.setAlignment(Qt.AlignCenter)
            video_label.setStyleSheet("background-color: black;")
            container_layout.addWidget(video_label)

            row = idx // cols
            col = idx % cols
            self.layout.addWidget(container, row, col)
            self.labels[fly_id] = video_label

    def update_from_frame(self, frame, frame_idx):
        self.current_frame_idx = frame_idx
        if self.active:
            self.worker.update_from_frame(frame, frame_idx)

    def update_grid(self, crops):
        for fly_id, crop in crops.items():
            if fly_id in self.labels:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.labels[fly_id].setPixmap(QPixmap.fromImage(qimg))

    def pause(self):
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False
        for label in self.labels.values():
            label.clear()

    def set_frame_index(self, frame_idx):
        self.current_frame_idx = frame_idx



