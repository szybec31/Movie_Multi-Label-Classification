from PyQt5.QtWidgets import (
    QVBoxLayout, QWidget, QPushButton,
    QLabel, QRadioButton, QHBoxLayout
)
from PyQt5.QtCore import Qt
# --- KROK 3 ---
class Step3(QWidget):
    def __init__(self, main, back_callback):
        super().__init__()

        self.main = main

        self.title = ""
        self.description = ""
        self.image_path = ""
        self.audio_video_path = ""

        # =========================
        # GŁÓWNY LAYOUT (centrowanie)
        # =========================
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        # =========================
        # KONTENER (spójny z Step2)
        # =========================
        container = QWidget()
        container.setMaximumWidth(400)
        container.setMinimumWidth(400)

        layout = QVBoxLayout()
        layout.setSpacing(12)

        # =========================
        # TYTUŁ
        # =========================
        title = QLabel("Select multimodal method")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        # =========================
        # RADIO BUTTONY
        # =========================
        self.late = QRadioButton("Late Fusion")
        self.early = QRadioButton("Early Fusion")

        layout.addWidget(self.late,alignment=Qt.AlignHCenter)
        layout.addWidget(self.early,alignment=Qt.AlignHCenter)

        # =========================
        # PRZYCISKI (OBOK SIEBIE)
        # =========================
        buttons_layout = QHBoxLayout()

        back_btn = QPushButton("Back")
        predict_btn = QPushButton("Predict")

        back_btn.setFixedHeight(35)
        predict_btn.setFixedHeight(35)

        back_btn.clicked.connect(back_callback)
        predict_btn.clicked.connect(self.predict)

        buttons_layout.addWidget(back_btn)
        buttons_layout.addWidget(predict_btn)

        # =========================
        # WYNIK (opcjonalnie lokalny)
        # =========================
        self.result = QLabel("")
        self.result.setAlignment(Qt.AlignCenter)

        # =========================
        # SKŁADANIE
        # =========================
        layout.addSpacing(10)
        layout.addLayout(buttons_layout)
        layout.addSpacing(10)
        layout.addWidget(self.result)

        #layout.addStretch()

        container.setLayout(layout)

        # wycentrowanie kontenera
        wrapper = QHBoxLayout()
        wrapper.addStretch()
        wrapper.addWidget(container)
        wrapper.addStretch()

        main_layout.addLayout(wrapper)
        self.setLayout(main_layout)

    # =========================
    # DANE
    # =========================
    def load_data(self):
        data = self.main.state.data

        self.title = data["title"]
        self.description = data["description"]
        self.image_path = data["image_path"]
        self.audio_video_path = data["audio_video_path"]

        print("Title:", self.title)
        print("Description:", self.description)
        print("Image path:", self.image_path)
        print("Audio-Video path:", self.audio_video_path)

    # =========================
    # PREDYKCJA
    # =========================
    def predict(self):
        if self.late.isChecked():
            self.main.state.method = "late"
        elif self.early.isChecked():
            self.main.state.method = "early"
        else:
            self.main.show_error("Wybierz metodę")
            return

        y_pred = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
        self.display_predict_result(y_pred)

    def display_predict_result(self,y_pred):
        res = []
        for idx, i in enumerate(y_pred):
            if i == 1:
                res.append(self.main.cat_list[idx])
        if len(res) == 0:
            self.main.show_result("Prediction: no results")
            return
        self.main.show_result(f"Prediction: {', '.join(res)}")