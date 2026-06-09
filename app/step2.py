from PyQt5.QtWidgets import (
    QVBoxLayout, QWidget, QPushButton, QLabel,
    QLineEdit, QTextEdit, QFileDialog, QHBoxLayout, QSizePolicy
)
from PyQt5.QtCore import Qt
import os


class Step2(QWidget):
    def __init__(self, main, state, next_callback, back_callback):
        super().__init__()

        self.main = main
        self.state = state
        self.next_callback = next_callback
        self.back_callback = back_callback

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        container = QWidget()
        container.setMaximumWidth(400)
        container.setMinimumWidth(400)
        container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        layout = QVBoxLayout()
        layout.setSpacing(12)

        self.title_label = QLabel("Title:")
        self.title_input = QLineEdit()

        self.desc_label = QLabel("Description:")
        self.desc_input = QTextEdit()
        self.desc_input.setMinimumHeight(130)

        self.image_label = QLabel("Image:")
        self.image_btn = QPushButton("Select image file")

        self.fields = {
            "text": [self.title_label, self.title_input, self.desc_label, self.desc_input],
            "image": [self.image_label, self.image_btn],
        }

        buttons_layout = QHBoxLayout()

        self.back_btn = QPushButton("Back")
        self.next_btn = QPushButton("Next")

        self.back_btn.clicked.connect(self.back_callback)
        self.next_btn.clicked.connect(self.on_next_clicked)

        self.back_btn.setFixedHeight(35)
        self.next_btn.setFixedHeight(35)

        buttons_layout.addWidget(self.back_btn)
        buttons_layout.addWidget(self.next_btn)

        self.image_btn.clicked.connect(self.select_image)

        text_block = QVBoxLayout()
        text_block.setSpacing(4)
        text_block.addWidget(self.title_label)
        text_block.addWidget(self.title_input)
        text_block.addSpacing(8)
        text_block.addWidget(self.desc_label)
        text_block.addWidget(self.desc_input)
        text_block.addStretch()

        image_block = QVBoxLayout()
        image_block.setSpacing(4)
        image_block.addWidget(self.image_label)
        image_block.addWidget(self.image_btn)
        image_block.addStretch()

        layout.addLayout(text_block)
        layout.addSpacing(15)
        layout.addLayout(image_block)
        layout.addSpacing(10)
        layout.addLayout(buttons_layout)

        container.setLayout(layout)

        wrapper = QHBoxLayout()
        wrapper.addStretch()
        wrapper.addWidget(container)
        wrapper.addStretch()

        main_layout.addLayout(wrapper)
        self.setLayout(main_layout)

    def update_ui(self):
        for key, widgets in self.fields.items():
            visible = self.state.sources.get(key, False)
            for w in widgets:
                w.setVisible(visible)

        count = sum(self.state.sources.values())
        if count <= 1:
            self.next_btn.setText("Predict")
        else:
            self.next_btn.setText("Next")

    def validate(self):
        title = self.title_input.text().strip()
        desc = self.desc_input.toPlainText().strip()

        if self.state.sources["text"]:
            if not title:
                self.main.show_error("No title")
                return False
            if not desc:
                self.main.show_error("No description")
                return False
            if self.word_count(desc) < 25:
                self.main.show_error("The description must consist of at least 25 words")
                return False

        if self.state.sources["image"]:
            if not self.constrains(self.state.data["image_path"], (".jpg", ".png")):
                return False

        self.main.clear_output()
        return True

    def word_count(self, text):
        return len([w for w in text.split() if w.strip()])

    def constrains(self, fpath, allowed):
        path = os.path.basename(fpath)
        if not fpath:
            self.main.show_error("No file selected")
            return False
        if not os.path.exists(fpath):
            self.main.show_error("File does not exist")
            return False
        if os.path.getsize(fpath) == 0:
            self.main.show_error("File is empty")
            return False
        if not path.lower().endswith(allowed):
            self.main.show_error("Unsupported file format")
            return False
        return True

    def on_next_clicked(self):
        if self.validate():
            self.save_data()
            count = sum(self.state.sources.values())

            # W trybach pojedynczych liczymy od razu i idziemy do Step 3
            if count <= 1:
                try:
                    self.main.clear_output()
                    self.state.predict = self.main.run_real_prediction()
                except Exception as e:
                    self.main.show_error(f"Prediction failed: {e}")
                    return

            # Dla fuzji przechodzimy czysto dalej, bez odpalania modeli w tym miejscu
            self.next_callback()

    def save_data(self):
        if self.state.sources["text"]:
            self.state.data["title"] = self.title_input.text()
            self.state.data["description"] = self.desc_input.toPlainText()

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg)")
        if path:
            self.state.data["image_path"] = path
            filename = os.path.basename(path)
            self.image_label.setText("Image: " + str(filename))
