from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow
import sys

app = QApplication(sys.argv)
window = QMainWindow()
window.setWindowTitle("Test PyQt6")
window.setGeometry(100, 100, 400, 300)
label = QLabel("PyQt6 Working!", window)
label.move(150, 130)
window.show()
sys.exit(app.exec())