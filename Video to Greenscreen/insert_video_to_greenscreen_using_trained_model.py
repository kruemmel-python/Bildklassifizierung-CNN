import os  # Modul zum Arbeiten mit dem Betriebssystem, z.B. zum Überprüfen von Dateipfaden
import sys  # Modul zum Zugriff auf Systemfunktionen wie Argumente und Exit
import json  # Modul zum Arbeiten mit JSON-Dateien
import numpy as np  # Bibliothek für numerische Berechnungen, insbesondere für Arrays
import cv2  # Bibliothek für die Bild- und Videobearbeitung
from typing import Union  # Hilft bei der Angabe von Datentypen in Funktionssignaturen
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QMessageBox, QComboBox
from PyQt5.QtCore import Qt

# Funktion zur Erstellung einer Maske für den Greenscreen-Bereich im Bild
def create_greenscreen_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Bild von BGR (Blau, Grün, Rot) in HSV (Farbton, Sättigung, Helligkeit) umwandeln
    lower_green = np.array([35, 100, 100])  # Untere Grenzen für die Grünfarbe im HSV-Farbraum definieren
    upper_green = np.array([85, 255, 255])  # Obere Grenzen für die Grünfarbe im HSV-Farbraum definieren
    mask = cv2.inRange(hsv, lower_green, upper_green)  # Maske erstellen, die nur die grünen Bereiche des Bildes enthält
    
    kernel = np.ones((5, 5), np.uint8)  # Rauschunterdrückung anwenden, um die Maske zu bereinigen
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels_im = cv2.connectedComponents(mask)  # Kleine weiße Flecken entfernen, indem verbundene Komponenten analysiert werden
    
    max_label, max_size = max(  # Komponenten analysieren und die größte Komponente beibehalten
        ((label, np.sum(labels_im == label)) for label in range(1, num_labels)),
        key=lambda x: x[1],
        default=(0, 0)
    )

    mask = np.uint8(labels_im == max_label) * 255  # Nur die größte Komponente als endgültige Maske verwenden
    
    return mask

# Funktion zum Ersetzen des Greenscreens durch ein Hintergrundvideo
def replace_greenscreen_with_video(original_img: np.ndarray, video_path: str, mask: np.ndarray, output_video_path: str) -> None:
    x, y, w, h = cv2.boundingRect(mask)  # Bounding Box des Greenscreen-Bereichs ermitteln (Position und Größe des Rechtecks, das den Greenscreen umgibt)
    print(f"Greenscreen area - Width: {w} px, Height: {h} px")
    
    cap = cv2.VideoCapture(video_path)  # Video öffnen
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file '{video_path}'.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video-Writer initialisieren, um das Ausgabevideo zu speichern
    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (original_img.shape[1], original_img.shape[0]))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        background_img_resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)  # Hintergrundvideo auf die Größe des Greenscreen-Bereichs strecken

        result = original_img.copy()  # Ergebnisbild erstellen
        mask_cropped = mask[y:y+h, x:x+w]
        mask_inv = cv2.bitwise_not(mask_cropped)

        for c in range(0, 3):  # Hintergrundbild in den Greenscreen-Bereich einfügen
            result[y:y+h, x:x+w, c] = (
                background_img_resized[:, :, c] * (mask_cropped / 255.0) + 
                result[y:y+h, x:x+w, c] * (mask_inv / 255.0)
            )

        out.write(result)  # Ergebnisbild zum Ausgabevideo hinzufügen
    
    cap.release()  # Ressourcen freigeben
    out.release()

class GreenScreenApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()  # Initialisiere die Benutzeroberfläche
    
    def load_language(self, language_code: str):
        with open(f"{language_code}.json", "r", encoding="utf-8") as file:
            self.translations = json.load(file)

    def initUI(self):
        self.setWindowTitle('Greenscreen Ersetzer')  # Setze den Titel des Fensters
        
        self.layout = QVBoxLayout()  # Erstelle ein vertikales Layout
        
        self.language_selector = QComboBox(self)
        self.language_selector.addItem("Deutsch", "de")
        self.language_selector.addItem("English", "en")
        self.language_selector.addItem("Русский", "ru")
        self.language_selector.currentIndexChanged.connect(self.change_language)
        self.layout.addWidget(self.language_selector)

        self.imageLabel = QLabel('Bild auswählen:')  # Label für die Bildauswahl
        self.layout.addWidget(self.imageLabel)
        
        self.imageButton = QPushButton('Bild auswählen')  # Schaltfläche für die Bildauswahl
        self.imageButton.clicked.connect(self.select_image)
        self.layout.addWidget(self.imageButton)
        
        self.videoLabel = QLabel('Video auswählen:')  # Label für die Videoauswahl
        self.layout.addWidget(self.videoLabel)
        
        self.videoButton = QPushButton('Video auswählen')  # Schaltfläche für die Videoauswahl
        self.videoButton.clicked.connect(self.select_video)
        self.layout.addWidget(self.videoButton)
        
        self.outputLabel = QLabel('Ausgabeordner auswählen:')  # Label für die Auswahl des Ausgabeordners
        self.layout.addWidget(self.outputLabel)
        
        self.outputButton = QPushButton('Ausgabeordner auswählen')  # Schaltfläche für die Auswahl des Ausgabeordners
        self.outputButton.clicked.connect(self.select_output_folder)
        self.layout.addWidget(self.outputButton)
        
        self.runButton = QPushButton('Erstellen')  # Schaltfläche zum Starten der Verarbeitung
        self.runButton.clicked.connect(self.run)
        self.layout.addWidget(self.runButton)
        
        self.setLayout(self.layout)  # Setze das Layout für das Fenster

        self.change_language()  # Initiale Sprachänderung durchführen

    def change_language(self):
        language_code = self.language_selector.currentData()
        self.load_language(language_code)

        # Update GUI elements with the selected language
        self.setWindowTitle(self.translations["title"])
        self.imageLabel.setText(self.translations["choose_image"])
        self.imageButton.setText(self.translations["select_image"])
        self.videoLabel.setText(self.translations["choose_video"])
        self.videoButton.setText(self.translations["select_video"])
        self.outputLabel.setText(self.translations["choose_output_folder"])
        self.outputButton.setText(self.translations["select_output_folder"])
        self.runButton.setText(self.translations["run"])
    
    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly  # Öffne den Dialog nur zum Lesen
        file_name, _ = QFileDialog.getOpenFileName(self, self.translations["choose_image"], "", "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)  # Öffne einen Dateiauswahldialog
        if file_name:
            self.image_path = file_name
            self.imageLabel.setText(f"{self.translations['choose_image']} {file_name}")  # Aktualisiere das Label mit dem ausgewählten Dateipfad
    
    def select_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly  # Öffne den Dialog nur zum Lesen
        file_name, _ = QFileDialog.getOpenFileName(self, self.translations["choose_video"], "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)", options=options)  # Öffne einen Dateiauswahldialog
        if file_name:
            self.video_path = file_name
            self.videoLabel.setText(f"{self.translations['choose_video']} {file_name}")  # Aktualisiere das Label mit dem ausgewählten Dateipfad
    
    def select_output_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly  # Öffne den Dialog nur zum Auswählen von Verzeichnissen
        folder_name = QFileDialog.getExistingDirectory(self, self.translations["choose_output_folder"], options=options)  # Öffne einen Verzeichnisauswahldialog
        if folder_name:
            self.output_folder = folder_name
            self.outputLabel.setText(f"{self.translations['choose_output_folder']} {folder_name}")  # Aktualisiere das Label mit dem ausgewählten Ordner
    
    def run(self):
        try:
            # Überprüfe, ob alle notwendigen Dateien und Ordner ausgewählt wurden
            original_image_path = getattr(self, 'image_path', None)
            video_path = getattr(self, 'video_path', None)
            output_folder = getattr(self, 'output_folder', None)
            if not original_image_path or not video_path or not output_folder:
                raise ValueError(self.translations["select_files_and_folder"])
            
            output_video_path = os.path.join(output_folder, 'output_video.mp4')  # Pfad für das Ausgabevideo
            original_img = cv2.imread(original_image_path)  # Lade das Originalbild
            if original_img is None:
                raise ValueError(self.translations["failed_to_load_image"].format(original_image_path=original_image_path))
            
            mask = create_greenscreen_mask(original_img)  # Erstelle die Maske für den Greenscreen-Bereich
            replace_greenscreen_with_video(original_img, video_path, mask, output_video_path)  # Ersetze den Greenscreen-Bereich durch das Video
            
            QMessageBox.information(self, self.translations["success"], self.translations["result_saved"].format(output_video_path=output_video_path))  # Zeige eine Erfolgsmeldung an
        except Exception as e:
            QMessageBox.critical(self, self.translations["error"], str(e))  # Zeige eine Fehlermeldung an

# Hauptfunktion zum Starten der Anwendung
def main():
    app = QApplication(sys.argv)  # Erstelle eine Anwendung
    ex = GreenScreenApp()  # Erstelle eine Instanz der GreenScreenApp
    ex.show()  # Zeige die Anwendung
    sys.exit(app.exec_())  # Führe die Anwendung aus und warte auf das Beenden

if __name__ == '__main__':
    main()  # Rufe die Hauptfunktion auf, wenn das Skript direkt ausgeführt wird
