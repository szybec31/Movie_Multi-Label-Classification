from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QVBoxLayout, QWidget, QCheckBox, QPushButton, QLabel, QHBoxLayout)

class Step1(QWidget):
    def __init__(self, main, next_callback):
        super().__init__()

        self.main = main
        self.next_callback = next_callback

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        container = QWidget()
        container.setMaximumWidth(300)

        layout = QVBoxLayout()
        layout.setSpacing(15)

        title = QLabel("Select your sources")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.title_cb = QCheckBox("Title and description")
        self.image_cb = QCheckBox("Image")

        for cb in [self.title_cb, self.image_cb]:
            cb.setStyleSheet("padding: 6px;")

        btn = QPushButton("Next")
        btn.setFixedHeight(35)
        btn.clicked.connect(self.on_next_clicked)

        layout.addWidget(title)
        layout.addSpacing(10)
        layout.addWidget(self.title_cb)
        layout.addWidget(self.image_cb)
        layout.addSpacing(15)
        layout.addWidget(btn)

        container.setLayout(layout)

        wrapper = QHBoxLayout()
        wrapper.addStretch()
        wrapper.addWidget(container)
        wrapper.addStretch()

        main_layout.addLayout(wrapper)
        self.setLayout(main_layout)

    def validate(self):
        selected = any([
            self.title_cb.isChecked(),
            self.image_cb.isChecked(),
        ])

        if not selected:
            self.main.show_error("Select at least one source")
            return False

        self.main.state.sources["text"] = self.title_cb.isChecked()
        self.main.state.sources["image"] = self.image_cb.isChecked()

        self.main.clear_output()
        return True

    def on_next_clicked(self):
        if self.validate():
            self.next_callback()
