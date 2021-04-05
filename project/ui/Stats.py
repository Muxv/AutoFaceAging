import copy

from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QWidget, QMessageBox, QFileDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile, QObject, Signal, Slot, Qt
from project.config import *
from project.request.predict import predict_request
from project.request.edit import edit_request
from PIL import Image
from PIL.ImageQt import ImageQt


def age_interval(benchmark):
    bit = benchmark % 10
    return {
        0: 20 + bit,
        1: 30 + bit,
        2: 40 + bit,
        3: 50 + bit
    }


class TextSignal(QObject):
    set_signal = Signal(str)
    append_signal = Signal(str)


signaler = TextSignal()


class Stats():
    def __init__(self, ui_file):
        qfile_stats = QFile(ui_file)
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()

        self.ui = QUiLoader().load(qfile_stats)
        # self.ui.setFixedSize(self.ui.size())
        self.src_img = None  # PIL
        self.dst_img = None  # PIL

        # https://www.coder.work/article/2034121
        self.src_qim = None # keep this to prevent image collapse
        self.dst_qim = None

        self.src_age = 0
        self.dst_age = 0

        self.src_path = "" # the complete path of image
        self.src_file = ""  # use for output

        self.setup_signals()
        self.setup_event()
        self.setup_components()

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
        self.ui.readButton.clicked.connect(self.open_img)  # block
        self.ui.predictButton.clicked.connect(self.predict_age)  # block
        self.ui.evaluateButton.clicked.connect(self.predict_age_again)  # block
        self.ui.transferButton.clicked.connect(self.edit_age)  # block
        self.ui.saveButton.clicked.connect(self.save_img)  # block
        self.ui.refreshButton.clicked.connect(self.refresh_img)
        self.feedback("事件系统初始化成功")

    def handle_selection_change(self):
        current_index = self.ui.targetBox.currentIndex()
        if current_index == -1:
            current_index = 0
        # self.dst_age =  [k for k, v in age_interval(self.src_age).items() if v == current_text]
        self.dst_age = age_interval(self.src_age)[current_index]

    def predict_age(self):
        age = predict_request(self.src_img)
        if age == -1:
            self.feedback("服务器连接失败")
        else:
            self.src_age = age
        self.ui.predictEdit.setText(f"预测年龄为{self.src_age}")

        self.ui.targetBox.clear()
        self.ui.targetBox.addItems(
            [f" {v}岁左右" for v in age_interval(self.src_age).values()]
        )

        self.feedback("预测年龄完成")

    def predict_age_again(self):
        self.dst_age = predict_request(self.dst_img)
        self.ui.evaluateEdit.setText(f"预测年龄为{self.dst_age}")
        self.feedback("预测年龄完成")

    def edit_age(self):
        self.dst_img = copy.deepcopy(self.src_img)
        self.dst_img = edit_request(self.dst_img, self.dst_age)
        # self.refresh_img()
        self.show_img_on_label(where="dst")
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
            # self.refresh_img()
            return
        self.src_path = file_path
        self.src_file = file_path.split("/")[-1]
        # print(self.src_file)
        # img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(file_path)
        self.src_img = Image.open(file_path)
        # self.refresh_img()
        self.show_img_on_label(where="src")
        self.ui.predictButton.setEnabled(True)
        self.ui.transferButton.setEnabled(True)
        self.feedback(f"成功读入{file_path}")

    def save_img(self):

        save_path = QFileDialog.getExistingDirectory(
            self.ui,
            "选择要保存的位置",
            "r./",
        )
        self.dst_img.save(f"{save_path}/new_{self.src_file}", quality=75)
        # img = cv2.cvtColor(self.dst_img.copy(), cv2.COLOR_RGB2BGR)
        # save_path = f"{save_path}/new_{self.src_file}"
        # cv2.imwrite(save_path, img)
        # self.refresh_img()
        self.feedback(f"结果保存为{save_path}")

    def show_img_on_label(self, where="src"):
        """show a img on a QLabel

        Args:
            where: choose to show src_img or dst_img
            img: a RGB PIL Image
        """
        img = self.src_img if where == "src" else self.dst_img
        if img is None:
            return
        assert isinstance(img, Image.Image)
        img = copy.deepcopy(img)
        resized = img.resize((TARGET_SIZE, TARGET_SIZE))
        if where == 'src':
            self.src_qim = ImageQt(resized)
            q_img = self.src_qim
            q_label = self.ui.inputImg
        else:
            self.dst_qim = ImageQt(resized)
            q_img = self.dst_qim
            q_label = self.ui.outputImg

        # q_img = QImage(resized.data, TARGET_SIZE, TARGET_SIZE, TARGET_SIZE * 3, QImage.Format_RGB888)
        # q_label.setPixmap(None)
        q_label.setPixmap(None)
        q_label.setPixmap(QPixmap.fromImage(q_img))
        q_label.setScaledContents(False)
        q_label.setAlignment(Qt.AlignCenter)
        q_label.show()

    def warning(self, content):
        msgBox = QMessageBox()
        msgBox.setWindowTitle('警告')
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText(content)
        msgBox.addButton('确认', QMessageBox.AcceptRole)
        msgBox.addButton('取消', QMessageBox.RejectRole)
        msgBox.exec()

    def refresh_img(self):
        self.show_img_on_label(where="src")
        self.show_img_on_label(where="dst")
