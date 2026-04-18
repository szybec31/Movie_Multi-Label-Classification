from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QVBoxLayout, QWidget, QCheckBox, QPushButton, QLabel, QHBoxLayout)
# --- KROK 1 ---
class Step1(QWidget):
    def __init__(self, main, next_callback):
        super().__init__()

        self.main = main
        self.next_callback = next_callback

        # =========================
        # GŁÓWNY LAYOUT
        # =========================
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)  # centrowanie całości

        # =========================
        # KONTENER (żeby nie było rozciągnięte)
        # =========================
        container = QWidget()
        container.setMaximumWidth(300)  # ograniczenie szerokości

        layout = QVBoxLayout()
        layout.setSpacing(15)

        # =========================
        # TYTUŁ
        # =========================
        title = QLabel("Select your sources")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")

        # =========================
        # CHECKBOXY
        # =========================
        self.title_cb = QCheckBox("Title and description")
        self.image_cb = QCheckBox("Image")
        self.aud_vid_cb = QCheckBox("Audio-Video")

        # większe checkboxy
        for cb in [self.title_cb, self.image_cb, self.aud_vid_cb]:
            cb.setStyleSheet("padding: 6px;")

        # =========================
        # PRZYCISK
        # =========================
        btn = QPushButton("Next")
        btn.setFixedHeight(35)
        btn.clicked.connect(self.on_next_clicked)

        # =========================
        # SKŁADANIE
        # =========================
        layout.addWidget(title)

        layout.addSpacing(10)

        layout.addWidget(self.title_cb)
        layout.addWidget(self.image_cb)
        layout.addWidget(self.aud_vid_cb)

        layout.addSpacing(15)
        layout.addWidget(btn)

        container.setLayout(layout)

        # centrowanie kontenera
        wrapper = QHBoxLayout()
        wrapper.addStretch()
        wrapper.addWidget(container)
        wrapper.addStretch()

        main_layout.addLayout(wrapper)

        self.setLayout(main_layout)

    # =========================
    # LOGIKA
    # =========================
    def validate(self):
        selected = any([
            self.title_cb.isChecked(),
            self.image_cb.isChecked(),
            self.aud_vid_cb.isChecked(),
        ])

        if not selected:
            self.main.show_error("Select at least one source")
            return False

        # zapis do state
        self.main.state.sources["text"] = self.title_cb.isChecked()
        self.main.state.sources["image"] = self.image_cb.isChecked()
        self.main.state.sources["audio_video"] = self.aud_vid_cb.isChecked()

        self.main.clear_output()
        return True

    def on_next_clicked(self):
        if self.validate():
            self.next_callback()