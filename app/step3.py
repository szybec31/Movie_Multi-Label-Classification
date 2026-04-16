from PyQt5.QtWidgets import (QVBoxLayout, QWidget, QPushButton, QLabel, QRadioButton)

# --- KROK 3 ---
class Step3(QWidget):
    def __init__(self,main, back_callback):
        super().__init__()

        self.main = main

        self.title = ""
        self.description = ""
        self.image_path = ""
        self.audio_video_path = ""
        self.method = None

        layout = QVBoxLayout()

        self.late = QRadioButton("Late Fusion")
        self.early = QRadioButton("Early Fusion")

        predict_btn = QPushButton("Predict")
        back_btn = QPushButton("Wstecz")

        predict_btn.clicked.connect(self.predict)
        back_btn.clicked.connect(back_callback)

        self.result = QLabel("")

        layout.addWidget(self.late)
        layout.addWidget(self.early)
        layout.addWidget(back_btn)
        layout.addWidget(predict_btn)
        layout.addWidget(self.result)

        self.setLayout(layout)

    def load_data(self):
        data = self.main.state.data

        self.title = data["title"]
        self.description = data["description"]
        self.image_path = data["image_path"]
        self.audio_video_path = data["audio_video_path"]

        print("Title: ",self.title)
        print("Description: ",self.description)
        print("Image path: ",self.image_path )
        print("Audio-Video path: ",self.audio_video_path)

    def predict(self):
        if self.late.isChecked():
            self.method = "late"
        elif self.early.isChecked():
            self.method = "early"
        else:
            self.main.show_error("Wybierz metodę")
            return









        self.main.show_result("Predykcja: [akcja, dramat]")
