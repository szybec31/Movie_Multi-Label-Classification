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

        # =========================
        # GŁÓWNY LAYOUT (centrowanie)
        # =========================
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        # =========================
        # KONTENER (ograniczenie szerokości)
        # =========================
        container = QWidget()
        container.setMaximumWidth(400)
        container.setMinimumWidth(400)
        container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # =========================
        # POLA
        # =========================
        self.title_label = QLabel("Title:")
        self.title_input = QLineEdit()

        self.desc_label = QLabel("Description:")
        self.desc_input = QTextEdit()
        #self.desc_input.setFixedHeight(140)
        self.desc_input.setMinimumHeight(130)

        self.image_label = QLabel("Image:")
        self.image_btn = QPushButton("Select image file")

        self.audio_video_label = QLabel("Audio-Video:")
        self.audio_video_btn = QPushButton("Select audio-video file")

        # =========================
        # MAPA pól
        # =========================
        self.fields = {
            "text": [self.title_label, self.title_input, self.desc_label, self.desc_input],
            "image": [self.image_label, self.image_btn],
            "audio_video": [self.audio_video_label, self.audio_video_btn],
        }

        # =========================
        # PRZYCISKI (OBOK SIEBIE)
        # =========================
        buttons_layout = QHBoxLayout()

        back_btn = QPushButton("Back")
        next_btn = QPushButton("Next")

        back_btn.clicked.connect(self.back_callback)
        next_btn.clicked.connect(self.on_next_clicked)

        # równa szerokość
        back_btn.setFixedHeight(35)
        next_btn.setFixedHeight(35)

        buttons_layout.addWidget(back_btn)
        buttons_layout.addWidget(next_btn)

        # =========================
        # AKCJE
        # =========================
        self.image_btn.clicked.connect(self.select_image)
        self.audio_video_btn.clicked.connect(self.select_audio_video)

        # =========================
        # SKŁADANIE
        # =========================
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


        av_block = QVBoxLayout()
        av_block.setSpacing(4)

        av_block.addWidget(self.audio_video_label)
        av_block.addWidget(self.audio_video_btn)
        av_block.addStretch()

        layout.addLayout(text_block)
        layout.addSpacing(15)

        layout.addLayout(image_block)
        layout.addSpacing(15)

        layout.addLayout(av_block)

        layout.addSpacing(10)
        layout.addLayout(buttons_layout)

        container.setLayout(layout)

        # wycentrowanie kontenera
        wrapper = QHBoxLayout()
        wrapper.addStretch()
        wrapper.addWidget(container)
        wrapper.addStretch()

        main_layout.addLayout(wrapper)
        self.setLayout(main_layout)

    # =========================
    # UI UPDATE
    # =========================
    def update_ui(self):
        for key, widgets in self.fields.items():
            visible = self.state.sources.get(key, False)
            for w in widgets:
                w.setVisible(visible)

    # =========================
    # WALIDACJA
    # =========================
    def validate(self):

        title = self.title_input.text().strip()
        desc = self.desc_input.toPlainText().strip()

        # TEXT ONLY IF ENABLED
        if self.state.sources["text"]:

            if not title:
                self.main.show_error("No title")
                return False

            if not desc:
                self.main.show_error("No description")
                return False

            if self.word_count(desc) < 25:
                self.main.show_error(
                    "The description must consist of at least 25 words"
                )
                return False

        # IMAGE VALIDATION
        if self.state.sources["image"]:

            if not self.constrains(
                    self.state.data["image_path"],
                    (".jpg", ".png")
            ):
                return False

        # AUDIO/VIDEO VALIDATION
        if self.state.sources["audio_video"]:

            if not self.constrains(
                    self.state.data["audio_video_path"],
                    (".mp4", ".avi", ".mp3", ".wav")
            ):
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
            self.next_callback()

    # =========================
    # SAVE
    # =========================
    def save_data(self):
        if self.state.sources["text"]:
            self.state.data["title"] = self.title_input.text()
            self.state.data["description"] = self.desc_input.toPlainText()

    # =========================
    # FILES
    # =========================
    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg)"
        )
        if path:
            self.state.data["image_path"] = path
        filename = os.path.basename(path)
        self.image_label.setText("Image: "+str(filename))

    def select_audio_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio-Video", "", "Media (*.mp4 *.avi *.mp3 *.wav)"
        )
        if path:
            self.state.data["audio_video_path"] = path
        filename = os.path.basename(path)
        self.audio_video_label.setText("Audio-Video: " + str(filename))