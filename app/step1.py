from PyQt5.QtWidgets import (QVBoxLayout, QWidget, QCheckBox, QPushButton, QLabel)

# --- KROK 1 ---
class Step1(QWidget):
    def __init__(self,main, next_callback):
        super().__init__()


        layout = QVBoxLayout()

        self.title_cb = QCheckBox("Tytuł i opis")
        self.image_cb = QCheckBox("Zdjęcie")
        self.aud_vid_cb = QCheckBox("Audio-Video")

        btn = QPushButton("Dalej")
        btn.clicked.connect(self.on_next_clicked)

        self.main = main
        self.next_callback = next_callback

        layout.addWidget(QLabel("Wybierz źródła:"))
        layout.addWidget(self.title_cb)
        layout.addWidget(self.image_cb)
        layout.addWidget(self.aud_vid_cb)
        layout.addWidget(btn)

        self.setLayout(layout)

    def validate(self):
        selected = any([
            self.title_cb.isChecked(),
            self.image_cb.isChecked(),
            self.aud_vid_cb.isChecked(),
        ])

        if not selected:
            self.main.show_error("Wybierz przynajmniej jedno źródło")
            return False

        self.main.state.sources["text"] = self.title_cb.isChecked()
        self.main.state.sources["image"] = self.image_cb.isChecked()
        self.main.state.sources["audio_video"] = self.aud_vid_cb.isChecked()


        self.main.clear_output()
        return True

    def on_next_clicked(self):
        if self.validate():
            self.next_callback()