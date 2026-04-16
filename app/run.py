from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QStackedWidget)
from step1 import Step1
from step2 import Step2
from step3 import Step3
import sys

class AppState:
    def __init__(self):
        self.sources = {
            "text": False,
            "image": False,
            "audio_video": False,
        }

        self.data = {
            "title": "",
            "description": "",
            "image_path": "",
            "audio_video_path": "",
        }

# --- MAIN ---
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Movie Classifier")

        self.stack = QStackedWidget()
        self.state = AppState()

        self.step1 = Step1(self, self.go_step2)
        self.step2 = Step2(self, self.state, self.go_step3, self.go_step1)
        self.step3 = Step3(self, self.go_step2)

        self.stack.addWidget(self.step1)
        self.stack.addWidget(self.step2)
        self.stack.addWidget(self.step3)

        self.output_label = QLabel("")

        layout = QVBoxLayout()
        layout.addWidget(self.stack)
        layout.addWidget(self.output_label)

        self.setLayout(layout)

    def go_step1(self):
        self.stack.setCurrentIndex(0)

    def go_step2(self):
        self.step2.update_ui()
        self.stack.setCurrentIndex(1)

    def go_step3(self):
        self.step3.load_data()
        self.stack.setCurrentIndex(2)

    def show_error(self, msg):
        self.output_label.setText(f"BŁĄD: {msg}")

    def show_result(self, msg):
        self.output_label.setText(msg)

    def clear_output(self):
        self.output_label.setText("")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(400, 400)
    window.show()
    sys.exit(app.exec_())