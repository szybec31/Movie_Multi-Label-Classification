from PyQt5.QtWidgets import (QVBoxLayout, QWidget, QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog)

# --- KROK 2 ---
class Step2(QWidget):
    def __init__(self,main, state, next_callback, back_callback):
        super().__init__()

        layout = QVBoxLayout()

        self.title_label = QLabel("Tytuł")
        self.title_input = QLineEdit()

        self.desc_label = QLabel("Opis")
        self.desc_input = QTextEdit()

        self.image_label = QLabel("Obraz")
        self.image_btn = QPushButton("Wybierz obraz")

        self.audio_video_label = QLabel("Audio-Video")
        self.audio_video_btn = QPushButton("Wybierz plik audio-video")

        self.fields = {
            "text": [self.title_label, self.title_input, self.desc_label, self.desc_input],
            "image": [self.image_label, self.image_btn],
            "audio_video": [self.audio_video_label, self.audio_video_btn],
        }

        next_btn = QPushButton("Dalej")
        back_btn = QPushButton("Wstecz")

        self.main = main
        self.state = state
        self.next_callback = next_callback
        self.back_callback = back_callback

        next_btn.clicked.connect(self.on_next_clicked)
        back_btn.clicked.connect(self.back_callback)

        self.image_btn.clicked.connect(self.select_image)
        self.audio_video_btn.clicked.connect(self.select_audio_video)

        layout.addWidget(self.title_label)
        layout.addWidget(self.title_input)

        layout.addWidget(self.desc_label)
        layout.addWidget(self.desc_input)

        layout.addWidget(self.image_label)
        layout.addWidget(self.image_btn)

        layout.addWidget(self.audio_video_label)
        layout.addWidget(self.audio_video_btn)

        layout.addWidget(back_btn)
        layout.addWidget(next_btn)

        self.setLayout(layout)

    def update_ui(self):
        for key, widgets in self.fields.items():
            visible = self.state.sources.get(key, False)
            for w in widgets:
                w.setVisible(visible)

    def validate(self):
        # TEXT
        if self.state.sources["text"]:
            if not self.title_input.text().strip():
                self.main.show_error("Brak tytułu")
                return False

            if not self.desc_input.toPlainText().strip():
                self.main.show_error("Brak opisu")
                return False

        # IMAGE
        if self.state.sources["image"]:
            if not self.state.data["image_path"]:
                self.main.show_error("Nie wybrano obrazu")
                return False

        # AUDIO-VIDEO
        if self.state.sources["audio_video"]:
            if not self.state.data["audio_video_path"]:
                self.main.show_error("Nie wybrano audio-video")
                return False

        self.main.clear_output()
        return True

    def on_next_clicked(self):
        if self.validate():
            self.save_data()
            self.next_callback()

    def save_data(self):
        if self.state.sources["text"]:
            self.state.data["title"] = self.title_input.text()
            self.state.data["description"] = self.desc_input.toPlainText()

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Wybierz obraz", "", "Images (*.png *.jpg)")
        if path:
            self.state.data["image_path"] = path

    def select_audio_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Wybierz audio-video", "", "Media (*.mp4 *.avi *.mp3 *.wav)")
        if path:
            self.state.data["audio_video_path"] = path
