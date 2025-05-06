# gui_mainwindow.py
from PySide6.QtWidgets import (
    QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
)
from tracker import track_flies

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fly Tracker")
        self.setFixedSize(1000, 800)

        self.video_path = None
        self.label = QLabel("Load a video to begin.")

        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)

        self.track_button = QPushButton("Track Flies")
        self.track_button.clicked.connect(self.track_video)
        self.track_button.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.load_button)
        layout.addWidget(self.track_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_path:
            self.video_path = file_path
            self.label.setText(f"Loaded: {file_path}")
            self.track_button.setEnabled(True)

    def track_video(self):
        if self.video_path:
            self.label.setText("Tracking in progress...")
            track_flies(self.video_path)
            self.label.setText("Tracking complete.")
