from PySide2.QtWidgets import QApplication
from src.config import *
from src.ui.Stats import Stats

if __name__ == '__main__':
    app = QApplication([])
    stats = Stats(UI_PATH)
    stats.ui.show()
    app.exec_()
