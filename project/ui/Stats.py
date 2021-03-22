from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile, QObject, Signal, Slot, Qt
from PIL import Image

import cv2
import numpy as np

def age_interval(benchmark):
    bit = benchmark % 10
    return {
        0: 20+bit,
        1: 30+bit,
        2: 40+bit,
        3: 50+bit,
        4: 60+bit,
    }

class TextSignal(QObject):
    set_signal = Signal(str)
    append_signal = Signal(str)

signaler = TextSignal()

class Stats():
    def __init__(self, ui_file, age_predictor, age_editor):
        qfile_stats = QFile(ui_file)
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()

        self.ui = QUiLoader().load(qfile_stats)
        self.src_img = None # RGB
        self.dst_img = None

        self.src_age = 0
        self.dst_age = 0

        self.src_file = "" # use for output

        self.setup_signals()
        self.setup_event()
        self.setup_components()

        self.age_predictor = age_predictor
        self.age_editor = age_editor
        self.feedback("年龄预测器与生成器初始化完成")

    def setup_signals(self):
        signaler.set_signal.connect(self.slot_set_text)
        signaler.append_signal.connect(self.slot_append_text)
        self.feedback("信号系统初始化成功")

    @Slot(str)
    def slot_set_text(self, text):
        self.ui.feedbackBrowser.setPlainText(text)

    @Slot(str)
    def slot_append_text(self, text):
        self.ui.feedbackBrowser.append(text)

    def feedback(self, text, append=True):
        if append:
            signaler.append_signal.emit(text)
        else:
            signaler.set_signal.emit(text)


    def setup_components(self):
        self.ui.targetBox.addItems(
            [f" {v}岁左右" for v in age_interval(5).values()]
        )
        self.ui.targetBox.currentIndexChanged.connect(self.handle_selection_change)
        self.feedback("组件初始化成功")

    def setup_event(self):
        self.ui.readButton.clicked.connect(self.open_img)       # block
        self.ui.predictButton.clicked.connect(self.predict_age) # block
        self.ui.transferButton.clicked.connect(self.edit_age) # block
        self.ui.saveButton.clicked.connect(self.save_img) # block
        self.feedback("事件系统初始化成功")

    def handle_selection_change(self):
        current_index = self.ui.targetBox.currentIndex()
        if current_index == -1:
            current_index = 0
        # self.dst_age =  [k for k, v in age_interval(self.src_age).items() if v == current_text]
        self.dst_age = age_interval(self.src_age)[current_index]
        # print(self.dst_age)

    def predict_age(self):
        self.src_age = self.age_predictor.predict_age(self.src_img).item()
        self.ui.predictEdit.setText(f"预测年龄为{self.src_age}")

        self.ui.targetBox.clear()
        self.ui.targetBox.addItems(
            [f" {v}岁左右" for v in age_interval(self.src_age).values()]
        )

        self.feedback("预测年龄完成")

    def edit_age(self):
        self.dst_img = self.age_editor.edit_age(self.src_img, self.dst_age)
        self.show_img_on_label(self.ui.outputImg, self.dst_img)
        self.ui.evaluateButton.setEnabled(True)
        self.feedback("转换年龄完成")


    def open_img(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择你要转换的图片",  # 标题
            r"./",  # 起始目录
            "图片类型 (*.png *.jpg *.bmp)"  # 选择类型过滤项，过滤内容在括号中
        )
        if file_path == "":
            return

        self.src_file = file_path.split("/")[-1]
        print(self.src_file)
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.src_img = img
        self.show_img_on_label(self.ui.inputImg, img)
        self.ui.predictButton.setEnabled(True)
        self.ui.transferButton.setEnabled(True)
        self.feedback(f"成功读入{file_path}")

    def save_img(self):
        save_path = QFileDialog.getExistingDirectory(
            self.ui,
            "选择要保存的位置",
            "r./",
        )
        img = cv2.cvtColor(self.dst_img, cv2.COLOR_RGB2BGR)
        save_path = f"{save_path}/new_{self.src_file}"
        cv2.imwrite(save_path, img)
        self.feedback(f"结果保存为{save_path}")


    def show_img_on_label(self, q_label, img):
        """show a img on a QLabel

        Args:
            q_label: QLabel Component on the ui
            img: a RGB numpy.ndarray vector size of (3, w, h)
        """
        assert type(img) == np.ndarray
        assert len(img.shape) == 3
        h, w, _ = img.shape
        q_img = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        q_label.setPixmap(QPixmap.fromImage(q_img))
        q_label.setAlignment(Qt.AlignCenter)

    def warning(self, content):
        msgBox = QMessageBox()
        msgBox.setWindowTitle('警告')
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText(content)
        msgBox.addButton('确认', QMessageBox.AcceptRole)
        msgBox.addButton('取消', QMessageBox.RejectRole)
        msgBox.exec()