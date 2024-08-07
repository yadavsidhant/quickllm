import sys
import os
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QComboBox, QLabel
from PyQt5.QtCore import Qt
from .models import load_model_and_tokenizer
from .chat import chat_with_model

class ChatInterface(QWidget):
    def __init__(self, output_dir, device):
        super().__init__()
        self.output_dir = output_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.list_saved_models())
        model_layout.addWidget(self.model_combo)
        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        model_layout.addWidget(load_button)
        layout.addLayout(model_layout)

        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        layout.addWidget(self.chat_history)

        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(send_button)
        layout.addLayout(input_layout)

        # Quit button
        quit_button = QPushButton("Quit")
        quit_button.clicked.connect(self.close)
        layout.addWidget(quit_button)

        self.setLayout(layout)
        self.setGeometry(300, 300, 500, 500)
        self.setWindowTitle('QuickLLM Chat Interface')
        self.show()

    def list_saved_models(self):
        saved_models = []
        for root, dirs, files in os.walk(self.output_dir):
            if 'config.json' in files:
                saved_models.append(os.path.relpath(root, self.output_dir))
        return saved_models

    def load_model(self):
        model_path = os.path.join(self.output_dir, self.model_combo.currentText())
        self.model, self.tokenizer = load_model_and_tokenizer(model_path)
        self.model = self.model.to(self.device)
        self.chat_history.append("Model loaded successfully!")

    def send_message(self):
        if not self.model or not self.tokenizer:
            self.chat_history.append("Please load a model first.")
            return

        user_input = self.input_field.text()
        self.chat_history.append(f"You: {user_input}")
        self.input_field.clear()

        response = chat_with_model(self.model, user_input, self.device)

        self.chat_history.append(f"Model: {response}")
        self.chat_history.append("")  # Add a blank line for readability

if __name__ == '__main__':
    app = QApplication(sys.argv)
    output_dir = input("Enter the path to your output directory: ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ex = ChatInterface(output_dir, device)
    sys.exit(app.exec_())
