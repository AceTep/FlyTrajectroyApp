import sys
import os
from PyQt5.QtWidgets import QApplication
from app.ui_main import CSVFilterApp
from utils.theme import setup_dark_theme

if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/path/to/your/qt/plugins/platforms"  
    os.environ["QT_MULTIMEDIA_PREFERRED_PLUGINS"] = "windowsmedia"
    app = QApplication(sys.argv)
    setup_dark_theme(app)
    app.setStyle("Fusion")
    window = CSVFilterApp()
    window.show()
    sys.exit(app.exec_())