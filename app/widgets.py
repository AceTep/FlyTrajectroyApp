from PyQt5.QtWidgets import QPushButton, QLabel
from utils.theme import ACCENT_COLOR, LIGHT_GRAY, DARK_GRAY, MEDIUM_GRAY, TEXT_COLOR

# Create a styled QPushButton with consistent theme
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

# Create a styled QLabel to be used as a section title
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
