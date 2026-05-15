import numpy as np
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
        self.text_vectorizer = main.text_vectorizer
        self.image_vectorizer = main.image_vectorizer

        self.movie_title = ""
        self.movie_description = ""
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
        self.title_label = QLabel("Select multimodal method")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.title_label)
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

    def selected_sources_count(self):
        return sum(self.main.state.sources.values())

    # =========================
    # DANE
    # =========================
    def load_data(self):
        data = self.main.state.data

        self.movie_title = data["title"]
        self.movie_description = data["description"]
        self.image_path = data["image_path"]
        self.audio_video_path = data["audio_video_path"]

        print("Title:", self.movie_title)
        print("Description:", self.movie_description)
        print("Image path:", self.image_path)
        print("Audio-Video path:", self.audio_video_path)

        count = self.selected_sources_count()

        if count <= 1:
            self.title_label.setText("Ready for prediction")
            self.late.hide()
            self.early.hide()
        else:
            self.title_label.setText("Select multimodal method")
            self.late.show()
            self.early.show()

    # =========================
    # PREDYKCJA
    # =========================
    def predict(self):
        textFlag = False
        imgFlag = False

        count = self.selected_sources_count()

        # TEXT
        if len(self.movie_description) != 0:
            textFlag = True
            text = self.movie_title + " " + self.movie_description

            text_vector = self.text_vectorizer.encode(text)
            text_vector = text_vector.reshape(1, -1)
            print(text_vector.shape)

        # IMAGE
        if len(self.image_path) != 0:
            imgFlag = True
            image_vector = self.image_vectorizer.encode(self.image_path)
            image_vector = image_vector.reshape(1, -1)
            print(image_vector.shape)

        # SINGLE MODAL
        if count <= 1:
            self.main.state.method = "single"

            if textFlag == True:
                print("Text single-modal")


            if imgFlag == True:
                print("Image single-modal")


        # MULTIMODAL
        else:
            if self.late.isChecked():
                self.main.state.method = "late"
                print("Late multi-modal")

            elif self.early.isChecked():
                self.main.state.method = "early"
                print("Early multi-modal")

                if textFlag and imgFlag:
                    early_features = np.concatenate(
                        (text_vector, image_vector),
                        axis=1
                    )

                    print(early_features.shape)
            else:
                self.main.show_error("Select multimodal method")
                return

        y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
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