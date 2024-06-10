import os  # Modul zum Arbeiten mit dem Betriebssystem, z.B. zum Überprüfen von Dateipfaden
import sys
import numpy as np  # Bibliothek für numerische Berechnungen, insbesondere für Arrays
import cv2  # Bibliothek für die Bild- und Videobearbeitung
from typing import List, Union  # Hilft bei der Angabe von Datentypen in Funktionssignaturen
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QMessageBox
from PyQt5.QtCore import Qt

# Funktion zur Erstellung einer Maske für den Greenscreen-Bereich im Bild
def create_greenscreen_masks(image: np.ndarray, num_greenscreens: int = 2) -> List[np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels_im = cv2.connectedComponents(mask)
    
    component_sizes = [(label, np.sum(labels_im == label)) for label in range(1, num_labels)]
    largest_components = sorted(component_sizes, key=lambda x: x[1], reverse=True)[:num_greenscreens]

    masks = []
    for label, _ in largest_components:
        component_mask = np.uint8(labels_im == label) * 255
        masks.append(component_mask)
    
    return masks

# Funktion zum Ersetzen des Greenscreens durch ein Hintergrundvideo
def replace_greenscreens_with_videos(original_img: np.ndarray, video_paths: List[str], masks: List[np.ndarray], output_video_path: str) -> None:
    if len(video_paths) != len(masks):
        raise ValueError("Die Anzahl der Videos muss mit der Anzahl der Greenscreen-Bereiche übereinstimmen. Prüfe ob es ein Greenscreen Bild ist!.")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (original_img.shape[1], original_img.shape[0]))

    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]
    if not all(cap.isOpened() for cap in caps):
        raise ValueError("Es ist fehlgeschlagen, eine oder mehrere Videodateien zu öffnen.")
    
    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frames.append(None)
            else:
                frames.append(frame)

        if any(frame is None for frame in frames):
            break
        
        result = original_img.copy()
        for mask, frame in zip(masks, frames):
            x, y, w, h = cv2.boundingRect(mask)
            background_img_resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            mask_cropped = mask[y:y+h, x:x+w]
            mask_inv = cv2.bitwise_not(mask_cropped)

            for c in range(3):
                result[y:y+h, x:x+w, c] = (
                    background_img_resized[:, :, c] * (mask_cropped / 255.0) + 
                    result[y:y+h, x:x+w, c] * (mask_inv / 255.0)
                )

        out.write(result)
    
    for cap in caps:
        cap.release()
    out.release()

class GreenScreenApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('VidReplaceFX')
        
        self.layout = QVBoxLayout()
        
        self.imageLabel = QLabel('Bild auswählen:')
        self.layout.addWidget(self.imageLabel)
        
        self.imageButton = QPushButton('Bild auswählen')
        self.imageButton.clicked.connect(self.select_image)
        self.layout.addWidget(self.imageButton)
        
        self.videoLabels = []
        self.videoButtons = []
        self.addVideoButton1 = QPushButton('Video 1 auswählen')
        self.addVideoButton1.clicked.connect(self.add_video1)
        self.layout.addWidget(self.addVideoButton1)

        self.addVideoButton2 = QPushButton('Video 2 auswählen')
        self.addVideoButton2.clicked.connect(self.add_video2)
        self.layout.addWidget(self.addVideoButton2)
        
        self.outputLabel = QLabel('Wählen Sie Ausgabeordner:')
        self.layout.addWidget(self.outputLabel)
        
        self.outputButton = QPushButton('Wählen Sie Ausgabeordner:')
        self.outputButton.clicked.connect(self.select_output_folder)
        self.layout.addWidget(self.outputButton)
        
        self.runButton = QPushButton('Erstelen')
        self.runButton.clicked.connect(self.run)
        self.layout.addWidget(self.runButton)
        
        self.setLayout(self.layout)
    
    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Wählen Sie eine Bilddatei", "", "Bilderdateien (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.image_path = file_name
            self.imageLabel.setText(f"Image: {file_name}")
    
    def add_video1(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Wählen Sie Videodatei 1", "", "Videodateien (*.mp4 *.avi *.mov);;Videodateien (*)", options=options)
        if file_name:
            self.video_path1 = file_name
            self.layout.addWidget(QLabel(f"Video 1: {file_name}"))
    
    def add_video2(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "SWählen Sie Videodatei 2", "", "Videodateien (*.mp4 *.avi *.mov);;Videodateien (*)", options=options)
        if file_name:
            self.video_path2 = file_name
            self.layout.addWidget(QLabel(f"Video 2: {file_name}"))
    
    def select_output_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        folder_name = QFileDialog.getExistingDirectory(self, "Ausgabeordner auswählen", options=options)
        if folder_name:
            self.output_folder = folder_name
            self.outputLabel.setText(f"Ausgabeordner: {folder_name}")
    
    def run(self):
        try:
            original_image_path = getattr(self, 'image_path', None)
            video_path1 = getattr(self, 'video_path1', None)
            video_path2 = getattr(self, 'video_path2', None)
            output_folder = getattr(self, 'output_folder', None)
            if not original_image_path or not video_path1 or not video_path2 or not output_folder:
                raise ValueError("Bitte wählen Sie alle benötigten Dateien und Ordner.")
            
            video_paths = [video_path1, video_path2]
            output_video_path = os.path.join(output_folder, 'output_video.mp4')
            original_img = cv2.imread(original_image_path)
            if original_img is None:
                raise ValueError(f"Fehler beim Laden des Originalbildes aus '{original_image_path}'.")
            
            masks = create_greenscreen_masks(original_img, num_greenscreens=len(video_paths))
            replace_greenscreens_with_videos(original_img, video_paths, masks, output_video_path)
            
            QMessageBox.information(self, 'Erfolgreich', f'Ergebnis gespeichert unter {output_video_path}')
        except Exception as e:
            QMessageBox.critical(self, 'Fehler', str(e))

def main():
    app = QApplication(sys.argv)
    ex = GreenScreenApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
