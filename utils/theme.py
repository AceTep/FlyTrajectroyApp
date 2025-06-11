from PyQt5.QtGui import QColor, QPalette

DARK_GRAY = "#2D2D2D"
MEDIUM_GRAY = "#3E3E3E"
LIGHT_GRAY = "#4F4F4F"
ACCENT_COLOR = "#5D9B9B"
WHITE = "#FFFFFF"
TEXT_COLOR = "#E0E0E0"


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