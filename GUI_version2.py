import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QFileDialog, QGridLayout, QTextEdit, QDialog
from PyQt5.QtGui import QPixmap
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon


class ImageViewerDialog(QDialog):
    def __init__(self, image_url):
        super().__init__()
        self.image_url = image_url

class ImageClassificationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Detection")
         # Đường dẫn đến file ảnh icon
        icon_path = "D:\\Face_Detection\\face-recognition-icon-20.jpg"

        # Thiết lập ảnh icon cho tiêu đề của giao diện
        self.setWindowIcon(QIcon(icon_path))
        self.setGeometry(500, 80, 900, 900)

        self.image_path = None

        self.image_label = QLabel("Image Link:")
        self.image_textbox = QLineEdit()
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_image)

        self.image_preview = QLabel()
        self.image_preview.setFixedSize(256,256)
        self.image_preview.setScaledContents(True)

        self.image_info = QLabel()

        # Thêm link show ảnh gốc
        self.original_link = QLabel("<a href='#'>Show original image</a>")
        self.original_link.setOpenExternalLinks(True)
        self.original_link.linkActivated.connect(self.on_link_activated)
        self.original_link.setVisible(False)


        self.message_box = QTextEdit()
        self.message_box.setReadOnly(True)

        self.test_button = QPushButton("Test")
        self.test_again_button = QPushButton("Test Again")
        self.test_again_button.clicked.connect(self.reset_info)
        self.test_button.clicked.connect(self.test_image)

        self.pie_chart = QLabel()
        self.pie_chart.setFixedSize(400, 400)
        self.pie_chart.setScaledContents(True)

        self.info_layout = QVBoxLayout()
        self.info_layout.addWidget(self.image_info)
        self.info_layout.addWidget(self.original_link)
    

        layout = QGridLayout()

        link_layout = QHBoxLayout()
        link_layout.addWidget(self.image_label)
        link_layout.addWidget(self.image_textbox)
        link_layout.addWidget(self.browse_button)

        layout.addLayout(link_layout, 0, 0, 1, 3)  # Thiết lập layout cho phần link ảnh và browse

        layout.addWidget(self.image_preview, 1, 0, 1, 2)
        layout.addLayout(self.info_layout, 1, 1, 1, 2)
        layout.addWidget(self.message_box, 2, 0, 1, 3)
        layout.addWidget(self.pie_chart, 3, 0, 1, 3)
    
        self.test_button.setFixedSize(100, 50) 
        self.test_again_button.setFixedSize(100, 50)

        # Tạo một QHBoxLayout
        button_layout = QHBoxLayout()
        # Thêm nút "Test Again" và "Test" vào button_layout
        button_layout.addWidget(self.test_again_button)
        button_layout.addWidget(self.test_button)

        # Tạo một QWidget để chứa button_layout
        button_container = QWidget()
        button_container.setLayout(button_layout)

        # Thêm button_container vào layout chính
        layout.addWidget(button_container, 4, 0, 1, 3, alignment=Qt.AlignRight)
        
        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
    

    def on_link_activated(self, url):
        if url == self.image_path:
            image_viewer = ImageViewerDialog(url)
            image_viewer.exec_()

    def browse_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
        if file_dialog.exec_():
            self.reset_info()  # Xóa thông tin cũ khi browse ảnh mới
            self.image_path = file_dialog.selectedFiles()[0]
            self.image_textbox.setText(self.image_path)
            self.display_image()
        
    def display_image(self):
        pixmap = QPixmap(self.image_path)
        self.image_preview.setPixmap(pixmap)

        image = Image.open(self.image_path)
        image_width, image_height = image.size
        image_size = os.path.getsize(self.image_path) // 1024
        image_format = os.path.splitext(self.image_path)[1].upper()
        image_name = os.path.splitext(os.path.basename(self.image_path))[0]
        info_text = f"Image Name: {image_name}\n"
        info_text += f"Image Size: {image_width}x{image_height} pixels\n"
        info_text += f"File Size: {image_size} KB\n"
        info_text += f"Image Format: {image_format}"
        self.image_info.setText(info_text)

        # Cập nhật đường dẫn cho liên kết "Show original image"
        self.original_link.setText(f'<a href="{self.image_path}">Show original image</a>')
        self.original_link.setVisible(True)
    
    def show_original_image(self):
        if self.image_path:
            original_pixmap = QPixmap(self.image_path)
            original_image_window = QLabel()
            original_image_window.setPixmap(original_pixmap)
            original_image_window.setWindowTitle("Original Image")
            original_image_window.show()
        else:
            self.message_box.setText("Please select an image first!")

    def reset_info(self):
        self.image_path = None
        self.image_textbox.clear()
        self.image_preview.clear()
        self.image_info.clear()
        self.message_box.clear()
        self.pie_chart.clear()
        self.original_link.clear()


    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize = tf.image.resize(img_rgb, (256, 256))
        normalized_image = resize / 255.0
        return normalized_image

    def classify_image(self, image_path):
        model = tf.keras.models.load_model("D:\\Face_Detection\\Model\\myModel.hdf5")
        preprocessed_image = self.preprocess_image(image_path)
        prediction = model.predict(np.expand_dims(preprocessed_image, 0))
        return prediction

    def test_image(self):
        if self.image_path:
            prediction = self.classify_image(self.image_path)

            labels = ["Fake", "Real"]
            sizes = [1 - prediction[0][0], prediction[0][0]]
            colors = ["red", "green"]

            fig, ax = plt.subplots(figsize=(6, 6))
            patches, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            plt.setp(autotexts, size=12, weight="bold")

            # Thêm phần chỉ dẫn
            ax.legend(patches, labels, loc="best")

            # Lưu biểu đồ tròn vào một tệp ảnh tạm thời
            temp_image_path = "temp_pie_chart.png"
            plt.savefig(temp_image_path)
            plt.close()

            # Hiển thị biểu đồ tròn trên giao diện
            pixmap = QPixmap(temp_image_path)
            self.pie_chart.setPixmap(pixmap)

            # Xóa tệp ảnh tạm thời
            os.remove(temp_image_path)

            # Hiển thị thông báo về ảnh real hay fake
            if prediction[0][0] > 0.5:
                message = "This is a real image!\n"
                accuracy = str(prediction[0][0] * 100)
                message += "Classification Accuracy: " + accuracy 
            else:
                message = "This is a fake image!\n"
                accuracy = str((1-prediction[0][0]) * 100)
                message += "Classification Accuracy: " + accuracy

            self.message_box.setText(message)
        else:
            self.message_box.setText("Please select an image first!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassificationWindow()
    window.show()
    sys.exit(app.exec_())