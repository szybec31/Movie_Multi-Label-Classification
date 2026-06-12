import numpy as np
from PyQt5.QtWidgets import (QVBoxLayout, QWidget, QPushButton, QLabel, QRadioButton, QHBoxLayout)
from PyQt5.QtCore import Qt


class Step3(QWidget):
    def __init__(self, main, back_callback):
        super().__init__()

        self.main = main
        self.back_callback = back_callback  # To jest oryginalny callback powrotu do Step 2
        self.text_vectorizer = main.text_vectorizer
        self.image_vectorizer = main.image_vectorizer

        self.movie_title = ""
        self.movie_description = ""
        self.image_path = ""

        # Flaga pomagająca określić, czy jesteśmy na ekranie wyniku, czy wyboru metody
        self.showing_results = False

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        container = QWidget()
        container.setMaximumWidth(400)
        container.setMinimumWidth(400)

        layout = QVBoxLayout()
        layout.setSpacing(12)

        # Główny nagłówek stanu
        self.title_label = QLabel("Select multimodal method")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.title_label)

        # Centralne pole tekstowe na wyniki predykcji
        self.genres_result_label = QLabel("")
        self.genres_result_label.setAlignment(Qt.AlignCenter)
        self.genres_result_label.setStyleSheet("font-size: 18px; color: #2ecc71; font-weight: bold; padding: 15px;")
        self.genres_result_label.setWordWrap(True)
        self.genres_result_label.hide()
        layout.addWidget(self.genres_result_label)

        # Radio buttony do fuzji multimedialnej
        self.late = QRadioButton("Late Fusion")
        self.early = QRadioButton("Early Fusion")

        layout.addWidget(self.late, alignment=Qt.AlignHCenter)
        layout.addWidget(self.early, alignment=Qt.AlignHCenter)

        buttons_layout = QHBoxLayout()

        self.back_btn = QPushButton("Back")
        self.predict_btn = QPushButton("Predict")

        self.back_btn.setFixedHeight(35)
        self.predict_btn.setFixedHeight(35)

        # Inteligentna obsługa przycisku powrotu i akcji predykcji
        self.back_btn.clicked.connect(self.handle_back)
        self.predict_btn.clicked.connect(self.predict)

        buttons_layout.addWidget(self.back_btn)
        buttons_layout.addWidget(self.predict_btn)

        layout.addSpacing(10)
        layout.addLayout(buttons_layout)

        container.setLayout(layout)

        wrapper = QHBoxLayout()
        wrapper.addStretch()
        wrapper.addWidget(container)
        wrapper.addStretch()

        main_layout.addLayout(wrapper)
        self.setLayout(main_layout)

    def selected_sources_count(self):
        return sum(self.main.state.sources.values())

    def load_data(self):
        data = self.main.state.data

        self.movie_title = data["title"]
        self.movie_description = data["description"]
        self.image_path = data["image_path"]

        count = self.selected_sources_count()

        # TRYB JEDNO-MODALNY (Sam tekst lub Sama grafika)
        if count <= 1:
            self.showing_results = True
            self.title_label.setText("Predicted Genres:")
            self.late.hide()
            self.early.hide()
            self.predict_btn.hide()

            genres = self.main.state.predict
            if genres:
                self.genres_result_label.setText(", ".join(genres))
            else:
                self.genres_result_label.setText("None (No matching genres)")

            self.genres_result_label.show()
            self.main.clear_output()

        # TRYB MULTIMODALNY (Tekst + Grafika)
        else:
            self.showing_results = False
            self.title_label.setText("Select multimodal method")
            self.late.show()
            self.early.show()
            self.predict_btn.show()
            self.genres_result_label.hide()
            # Odznaczamy radio buttony przy świeżym wejściu, by zmusić do wyboru
            self.late.setAutoExclusive(False)
            self.early.setAutoExclusive(False)
            self.late.setChecked(False)
            self.early.setChecked(False)
            self.late.setAutoExclusive(True)
            self.early.setAutoExclusive(True)

    def handle_back(self):
        count = self.selected_sources_count()

        # Jeśli jesteśmy w trybie fuzji i aktualnie pokazujemy już wynik końcowy na środku...
        if count > 1 and self.showing_results:
            # ...to cofamy się tylko do wyboru metody fuzji (ponownie ładujemy ten krok)
            self.load_data()
        else:
            # W przeciwnym wypadku (tryb pojedynczy LUB tryb fuzji na etapie wyboru radio) – wracamy do Step 2
            self.back_callback()

    def predict(self):
        if self.late.isChecked():
            self.main.state.method = "late"

            try:
                genres = self.main.run_late_fusion_prediction()

                self.showing_results = True
                self.title_label.setText("Predicted Genres:")

                self.late.hide()
                self.early.hide()
                self.predict_btn.hide()

                if genres:
                    self.genres_result_label.setText(", ".join(genres))
                    print(f"Wynik fuzji: {genres}")
                else:
                    self.genres_result_label.setText("None (No matching genres)")

                self.genres_result_label.show()
                self.main.clear_output()

            except Exception as e:
                self.main.show_error(f"Prediction failed: {e}")

        elif self.early.isChecked():
            self.main.state.method = "early"
            print("Early multi-modal execution...")

            try:
                # Wykonanie prawdziwej predykcji z pliku .pkl
                genres = self.main.run_real_prediction()
                print(f"Wynik fuzji: {genres}")

                # Zmieniamy widok interfejsu na centralną predykcję
                self.showing_results = True
                self.title_label.setText("Predicted Genres:")
                self.late.hide()
                self.early.hide()
                self.predict_btn.hide()

                if genres:
                    self.genres_result_label.setText(", ".join(genres))
                else:
                    self.genres_result_label.setText("None (No matching genres)")

                self.genres_result_label.show()
                self.main.clear_output()  # Czyszczenie dolnego paska zgodnie z opisem

            except Exception as e:
                self.main.show_error(f"Prediction failed: {e}")
        else:
            self.main.show_error("Select multimodal method")
