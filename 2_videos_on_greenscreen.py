import os  # Modul zum Arbeiten mit dem Betriebssystem, z.B. zum Überprüfen von Dateipfaden
import sys  # Modul zum Zugriff auf Systemfunktionen wie Argumente und Exit
import numpy as np  # Bibliothek für numerische Berechnungen, insbesondere für Arrays
import cv2  # Bibliothek für die Bild- und Videobearbeitung
from typing import List, Union  # Hilft bei der Angabe von Datentypen in Funktionssignaturen
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QMessageBox
from PyQt5.QtCore import Qt

# Funktion zur Erstellung einer Maske für den Greenscreen-Bereich im Bild
def create_greenscreen_masks(image: np.ndarray, num_greenscreens: int = 2) -> List[np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Konvertiere das Bild von BGR zu HSV Farbraum
    lower_green = np.array([35, 100, 100])  # Definiere die untere Grenze für den Grünfarbton
    upper_green = np.array([85, 255, 255])  # Definiere die obere Grenze für den Grünfarbton
    mask = cv2.inRange(hsv, lower_green, upper_green)  # Erstelle eine Maske, die nur den grünen Bereich enthält
    
    kernel = np.ones((5, 5), np.uint8)  # Erstelle einen Kernel zur Rauschunterdrückung
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Schließe kleine Löcher in der Maske
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Entferne kleine weiße Punkte aus der Maske
    
    num_labels, labels_im = cv2.connectedComponents(mask)  # Finde alle verbundenen Komponenten in der Maske
    
    # Finde die größten verbundenen Komponenten
    component_sizes = [(label, np.sum(labels_im == label)) for label in range(1, num_labels)]
    largest_components = sorted(component_sizes, key=lambda x: x[1], reverse=True)[:num_greenscreens]

    masks = []
    for label, _ in largest_components:
        component_mask = np.uint8(labels_im == label) * 255  # Erstelle eine Maske für die größte Komponente
        masks.append(component_mask)
    
    return masks  # Rückgabe der Masken für die größten Greenscreen-Bereiche

# Funktion zum Ersetzen des Greenscreens durch ein Hintergrundvideo
def replace_greenscreens_with_videos(original_img: np.ndarray, video_paths: List[str], masks: List[np.ndarray], output_video_path: str) -> None:
    if len(video_paths) != len(masks):  # Überprüfe, ob die Anzahl der Videos mit der Anzahl der Greenscreen-Bereiche übereinstimmt
        raise ValueError("Die Anzahl der Videos muss mit der Anzahl der Greenscreen-Bereiche übereinstimmen.")
    
    # Initialisiere den Video-Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (original_img.shape[1], original_img.shape[0]))

    # Öffne die Video-Dateien
    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]
    if not all(cap.isOpened() for cap in caps):
        raise ValueError("Es ist fehlgeschlagen, eine oder mehrere Videodateien zu öffnen.")
    
    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()  # Lese einen Frame aus dem Video
            if not ret:
                frames.append(None)  # Wenn das Lesen fehlschlägt, füge None hinzu
            else:
                frames.append(frame)  # Füge den gelesenen Frame hinzu

        if any(frame is None for frame in frames):  # Breche ab, wenn ein Frame nicht gelesen werden konnte
            break
        
        result = original_img.copy()  # Erstelle eine Kopie des Originalbildes
        for mask, frame in zip(masks, frames):
            x, y, w, h = cv2.boundingRect(mask)  # Bestimme die Begrenzungsbox der Maske
            background_img_resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)  # Passe die Größe des Hintergrundbildes an
            mask_cropped = mask[y:y+h, x:x+w]  # Schneide die Maske zu
            mask_inv = cv2.bitwise_not(mask_cropped)  # Invertiere die Maske

            for c in range(3):  # Übertrage die Kanäle des Bildes
                result[y:y+h, x:x+w, c] = (
                    background_img_resized[:, :, c] * (mask_cropped / 255.0) + 
                    result[y:y+h, x:x+w, c] * (mask_inv / 255.0)
                )

        out.write(result)  # Schreibe das Ergebnis in das Ausgabevideo
    
    for cap in caps:  # Schließe die Video-Dateien
        cap.release()
    out.release()  # Schließe die Ausgabe

# Klasse für die GUI-Anwendung
class GreenScreenApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()  # Initialisiere die Benutzeroberfläche
    
    def initUI(self):
        self.setWindowTitle('VidReplaceFX')  # Setze den Titel des Fensters
        
        self.layout = QVBoxLayout()  # Erstelle ein vertikales Layout
        
        self.imageLabel = QLabel('Bild auswählen:')  # Label für die Bildauswahl
        self.layout.addWidget(self.imageLabel)
        
        self.imageButton = QPushButton('Bild auswählen')  # Schaltfläche für die Bildauswahl
        self.imageButton.clicked.connect(self.select_image)
        self.layout.addWidget(self.imageButton)
        
        self.videoLabels = []
        self.videoButtons = []
        self.addVideoButton1 = QPushButton('Video 1 auswählen')  # Schaltfläche für die Auswahl des ersten Videos
        self.addVideoButton1.clicked.connect(self.add_video1)
        self.layout.addWidget(self.addVideoButton1)

        self.addVideoButton2 = QPushButton('Video 2 auswählen')  # Schaltfläche für die Auswahl des zweiten Videos
        self.addVideoButton2.clicked.connect(self.add_video2)
        self.layout.addWidget(self.addVideoButton2)
        
        self.outputLabel = QLabel('Wählen Sie Ausgabeordner:')  # Label für die Auswahl des Ausgabeordners
        self.layout.addWidget(self.outputLabel)
        
        self.outputButton = QPushButton('Wählen Sie Ausgabeordner')  # Schaltfläche für die Auswahl des Ausgabeordners
        self.outputButton.clicked.connect(self.select_output_folder)
        self.layout.addWidget(self.outputButton)
        
        self.runButton = QPushButton('Erstellen')  # Schaltfläche zum Starten der Verarbeitung
        self.runButton.clicked.connect(self.run)
        self.layout.addWidget(self.runButton)
        
        self.setLayout(self.layout)  # Setze das Layout für das Fenster
    
    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Wählen Sie eine Bilddatei", "", "Bilderdateien (*.png *.jpg *.bmp);;Alle Dateien (*)", options=options)
        if file_name:
            self.image_path = file_name
            self.imageLabel.setText(f"Bild: {file_name}")  # Aktualisiere das Label mit dem ausgewählten Dateipfad
    
    def add_video1(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Wählen Sie Videodatei 1", "", "Videodateien (*.mp4 *.avi *.mov);;Alle Dateien (*)", options=options)
        if file_name:
            self.video_path1 = file_name
            self.layout.addWidget(QLabel(f"Video 1: {file_name}"))  # Füge ein Label mit dem Dateipfad des ersten Videos hinzu
    
    def add_video2(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Wählen Sie Videodatei 2", "", "Videodateien (*.mp4 *.avi *.mov);;Alle Dateien (*)", options=options)
        if file_name:
            self.video_path2 = file_name
            self.layout.addWidget(QLabel(f"Video 2: {file_name}"))  # Füge ein Label mit dem Dateipfad des zweiten Videos hinzu
    
    def select_output_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        folder_name = QFileDialog.getExistingDirectory(self, "Ausgabeordner auswählen", options=options)
        if folder_name:
            self.output_folder = folder_name
            self.outputLabel.setText(f"Ausgabeordner: {folder_name}")  # Aktualisiere das Label mit dem ausgewählten Ordner
    
    def run(self):
        try:
            # Überprüfe, ob alle notwendigen Dateien und Ordner ausgewählt wurden
            original_image_path = getattr(self, 'image_path', None)
            video_path1 = getattr(self, 'video_path1', None)
            video_path2 = getattr(self, 'video_path2', None)
            output_folder = getattr(self, 'output_folder', None)
            if not original_image_path or not video_path1 or not video_path2 or not output_folder:
                raise ValueError("Bitte wählen Sie alle benötigten Dateien und Ordner.")
            
            video_paths = [video_path1, video_path2]  # Liste der Videodateipfade
            output_video_path = os.path.join(output_folder, 'output_video.mp4')  # Pfad für das Ausgabevideo
            original_img = cv2.imread(original_image_path)  # Lade das Originalbild
            if original_img is None:
                raise ValueError(f"Fehler beim Laden des Originalbildes aus '{original_image_path}'.")
            
            # Erstelle Masken für die Greenscreen-Bereiche
            masks = create_greenscreen_masks(original_img, num_greenscreens=len(video_paths))
            # Ersetze die Greenscreen-Bereiche durch die Videos
            replace_greenscreens_with_videos(original_img, video_paths, masks, output_video_path)
            
            QMessageBox.information(self, 'Erfolgreich', f'Ergebnis gespeichert unter {output_video_path}')
        except Exception as e:
            QMessageBox.critical(self, 'Fehler', str(e))  # Zeige eine Fehlermeldung an

def main():
    app = QApplication(sys.argv)  # Erstelle eine Anwendung
    ex = GreenScreenApp()  # Erstelle eine Instanz der GreenScreenApp
    ex.show()  # Zeige die Anwendung
    sys.exit(app.exec_())  # Führe die Anwendung aus und warte auf das Beenden

if __name__ == '__main__':
    main()  # Rufe die Hauptfunktion auf, wenn das Skript direkt ausgeführt wird
