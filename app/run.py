import torch
print("TORCH OK")
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QStackedWidget, QPushButton, QHBoxLayout,QListWidget, QFrame)
from PyQt5.QtCore import Qt
from styles import get_dark_style
from step1 import Step1
from step2 import Step2
from step3 import Step3

from app.models.text_vectorizer import TextVectorizer
from app.models.image_vectorizer import ImageVectorizer
import sys

class AppState:
    def __init__(self):
        self.sources = {
            "text": False,
            "image": False,
            "audio_video": False,
        }

        self.data = {
            "title": "",
            "description": "",
            "image_path": "",
            "audio_video_path": "",
        }
        self.method = None
        self.predict = []

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.text_vectorizer = TextVectorizer()
        if not self.text_vectorizer.ready:
            self.main.show_error(
                "Model not loaded. Check internet connection. Model needs connection only at first run."
            )
        self.image_vectorizer = ImageVectorizer()
        self.cat_list = [
            'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Family',
            'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance',
            'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']

        self.setWindowTitle("Movie Classifier")
        self.setStyleSheet(get_dark_style())

        # rozmiar
        self.resize(680, 680)
        self.setMinimumSize(680, 680)
        self.setMaximumSize(680, 680)

        # =========================
        # LEWA STRONA (kategorie)
        # =========================
        self.category_list = QListWidget()
        self.category_list.addItems(self.cat_list)
        self.category_list.setMaximumWidth(150)
        self.category_list.setMinimumWidth(150)
        self.category_list.setMinimumHeight(610)
        self.category_list.setDisabled(True)

        sidebar_layout = QVBoxLayout()
        sidebar_layout.addWidget(QLabel("All Categories"))
        sidebar_layout.addWidget(self.category_list)
        sidebar_layout.addStretch()

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar_layout)
        sidebar_widget.setMaximumWidth(200)

        # =========================
        # PRAWA STRONA
        # =========================
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # HEADER
        self.header = QLabel("Movie Classifier")
        self.header.setObjectName("Header")
        self.header.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.header)

        # =========================
        # PRZYCISKI
        # =========================
        actions_layout = QHBoxLayout()

        self.info_btn = QPushButton("Info")
        self.clean_btn = QPushButton("Clear")
        self.exit_btn = QPushButton("Exit")

        self.info_btn.clicked.connect(self.show_info)
        self.clean_btn.clicked.connect(self.reset_app)
        self.exit_btn.clicked.connect(self.close)

        actions_layout.addWidget(self.info_btn)
        actions_layout.addWidget(self.clean_btn)
        actions_layout.addStretch()
        actions_layout.addWidget(self.exit_btn)

        right_layout.addLayout(actions_layout)

        # =========================
        # STACK
        # =========================
        self.stack = QStackedWidget()
        self.state = AppState()

        self.step1 = Step1(self, self.go_step2)
        self.step2 = Step2(self, self.state, self.go_step3, self.go_step1)
        self.step3 = Step3(self, self.go_step2)

        self.stack.addWidget(self.step1)
        self.stack.addWidget(self.step2)
        self.stack.addWidget(self.step3)

        # =========================
        # FORM CARD
        # =========================
        self.form_container = QFrame()
        self.form_container.setObjectName("FormCard")

        form_layout = QVBoxLayout()
        form_layout.setAlignment(Qt.AlignTop)
        form_layout.addWidget(self.stack)

        self.form_container.setLayout(form_layout)
        right_layout.addWidget(self.form_container)

        # =========================
        # OUTPUT
        # =========================
        self.output_label = QLabel("")
        self.output_label.setObjectName("Output")
        self.output_label.setWordWrap(True)

        right_layout.addWidget(self.output_label)

        # =========================
        # MAIN
        # =========================
        main_layout = QHBoxLayout()
        main_layout.addWidget(sidebar_widget, 1)
        main_layout.addLayout(right_layout, 4)

        self.setLayout(main_layout)

    # =========================
    # NAVIGACJA
    # =========================
    def go_step1(self):
        self.stack.setCurrentIndex(0)

    def go_step2(self):
        self.step2.update_ui()
        self.stack.setCurrentIndex(1)

    def go_step3(self):
        self.step3.load_data()
        self.stack.setCurrentIndex(2)

    # =========================
    # OUTPUT
    # =========================
    def show_error(self, msg):
        self.output_label.setStyleSheet("color: red;")
        self.output_label.setText(f"Error: {msg}")

    def show_result(self, msg):
        self.output_label.setStyleSheet("color: lightgreen;")
        self.output_label.setText(msg)

    def clear_output(self):
        self.output_label.setText("")

    # =========================
    # AKCJE
    # =========================
    def show_info(self):
        self.show_result("Movie classifier – demo")

    def reset_app(self):

        self.state.sources = {
            "text": False,
            "image": False,
            "audio_video": False
        }
        self.state.data = {
            "title": "",
            "description": "",
            "image_path": "",
            "audio_video_path": "",
        }
        self.state.method = None

        self.step1.title_cb.setChecked(False)
        self.step1.image_cb.setChecked(False)
        self.step1.aud_vid_cb.setChecked(False)

        self.step2.title_input.clear()
        self.step2.desc_input.clear()
        self.step2.image_label.setText("Image:")
        self.step2.audio_video_label.setText("Audio-Video:")

        self.clear_output()
        self.go_step1()

import os, sys
def resource_path(relative_path):
    # PyInstaller
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.abspath(".")

    return os.path.join(base, relative_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    BASE_DIR = os.path.dirname(sys.executable)
    icon = QIcon(resource_path("assets/icon.ico"))
    app.setWindowIcon(icon)
    window = MainWindow()
    window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec_())