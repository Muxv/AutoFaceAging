from PySide2.QtWidgets import QApplication
from ui.Stats import Stats
from config import *

if __name__ == '__main__':
    app = QApplication([])

    stats = Stats(UI_PATH)

    stats.ui.show()
    app.exec_()