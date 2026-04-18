def get_dark_style():
    return """
    QWidget {
        background-color: #1e1e1e;
        color: white;
        font-size: 14px;
    }

    QLabel#Header {
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }

    QPushButton {
        background-color: #2d2d2d;
        border: none;
        padding: 8px;
        border-radius: 6px;
    }

    QPushButton:hover {
        background-color: #3d3d3d;
    }

    QListWidget {
        background-color: #2a2a2a;
        border: none;
        padding: 5px;
    }

    QListWidget::item {
        padding: 6px;
    }

    QListWidget::item:selected {
        background-color: #3d3d3d;
    }

    QFrame#FormCard {
        background-color: #2a2a2a;
        border-radius: 10px;
        padding: 8px;
        border: none;
    }

    QLabel#Output {
        background-color: #2a2a2a;
        padding: 10px;
        border-radius: 6px;
    }
    QCheckBox {
        padding: 10px;
        font-size: 16px;
        spacing: 8px;
    }
    
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
    }
    
    QCheckBox:hover {
        background-color: #333;
        border-radius: 4px;
    }
    QLineEdit, QTextEdit {
        background-color: #2a2a2a;
        border: none;
        padding: 6px;
        border-radius: 4px;
    }
    QRadioButton {
        padding: 12px;
        border-radius: 6px;
        font-size: 15px;
    }
    QRadioButton::indicator {
        width: 18px;
        height: 18px;
    }
    """