from PySide2.QtWidgets import QApplication
from ui.Stats import Stats
from age_predictor.predictor import AgePredictor
from age_editor.editor import AgeEditor
from config import *

if __name__ == '__main__':
    app = QApplication([])

    age_predictor = AgePredictor(VGG_PATH)
    age_editor = AgeEditor(EDITOR_PATH)
    stats = Stats(UI_PATH,
                  age_predictor=age_predictor,
                  age_editor=age_editor)


    stats.ui.show()
    app.exec_()