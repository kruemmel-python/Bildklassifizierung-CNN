import os  # Modul zum Arbeiten mit dem Betriebssystem, z.B. zum Überprüfen von Dateipfaden
import numpy as np  # Bibliothek für numerische Berechnungen, insbesondere für Arrays
import cv2  # Bibliothek für die Bild- und Videobearbeitung
from typing import Union  # Hilft bei der Angabe von Datentypen in Funktionssignaturen

# Funktion zur Erstellung einer Maske für den Greenscreen-Bereich im Bild
def create_greenscreen_mask(image: np.ndarray) -> np.ndarray:
    # Bild von BGR (Blau, Grün, Rot) in HSV (Farbton, Sättigung, Helligkeit) umwandeln
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Untere und obere Grenzen für die Grünfarbe im HSV-Farbraum definieren
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    
    # Maske erstellen, die nur die grünen Bereiche des Bildes enthält
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Rauschunterdrückung anwenden, um die Maske zu bereinigen
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Kleine weiße Flecken entfernen, indem verbundene Komponenten analysiert werden
    num_labels, labels_im = cv2.connectedComponents(mask)
    
    # Komponenten analysieren und die größte Komponente beibehalten
    max_label = 1
    max_size = 0
    for label in range(1, num_labels):
        size = np.sum(labels_im == label)
        if size > max_size:
            max_size = size
            max_label = label

    # Nur die größte Komponente als endgültige Maske verwenden
    mask = np.uint8(labels_im == max_label) * 255
    
    return mask

# Funktion zum Ersetzen des Greenscreens durch ein Hintergrundvideo
def replace_greenscreen_with_video(original_img: np.ndarray, video_path: str, mask: np.ndarray, output_video_path: str) -> None:
    # Bounding Box des Greenscreen-Bereichs ermitteln (Position und Größe des Rechtecks, das den Greenscreen umgibt)
    x, y, w, h = cv2.boundingRect(mask)
    print(f"Greenscreen area - Width: {w} px, Height: {h} px")
    
    # Video öffnen
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file '{video_path}'.")

    # Video-Writer initialisieren, um das Ausgabevideo zu speichern
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (original_img.shape[1], original_img.shape[0]))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Hintergrundvideo auf die Größe des Greenscreen-Bereichs strecken
        background_img_resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        # Ergebnisbild erstellen
        result = original_img.copy()
        mask_cropped = mask[y:y+h, x:x+w]
        mask_inv = cv2.bitwise_not(mask_cropped)

        # Hintergrundbild in den Greenscreen-Bereich einfügen
        for c in range(0, 3):
            result[y:y+h, x:x+w, c] = (
                background_img_resized[:, :, c] * (mask_cropped / 255.0) + 
                result[y:y+h, x:x+w, c] * (mask_inv / 255.0)
            )

        # Ergebnisbild zum Ausgabevideo hinzufügen
        out.write(result)
    
    # Ressourcen freigeben
    cap.release()
    out.release()

# Hauptfunktion, um das Skript auszuführen
def main():
    # Pfade zu den Eingabedateien und der Ausgabedatei
    original_image_path: Union[str, None] = 'l1.jpg'  # Pfad zum Bild mit Greenscreen
    video_path: Union[str, None] = 'maus.mp4'  # Pfad zum Hintergrundvideo
    output_video_path: Union[str, None] = 'output_maus.mp4'  # Pfad zum Ausgabevideo

    try:
        # Überprüfen, ob die Dateien existieren
        if not os.path.exists(original_image_path):
            raise FileNotFoundError(f"Original image file '{original_image_path}' not found.")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Background video file '{video_path}' not found.")
        
        # Originalbild laden
        original_img = cv2.imread(original_image_path)
        if original_img is None:
            raise ValueError(f"Failed to load original image from '{original_image_path}'.")

        # Maske für den Greenscreen erstellen
        mask = create_greenscreen_mask(original_img)
        
        # Greenscreen durch das Hintergrundvideo ersetzen
        replace_greenscreen_with_video(original_img, video_path, mask, output_video_path)

        print(f'Result saved to {output_video_path}')
    except Exception as e:
        print(f"An error occurred: {e}")

# Wenn das Skript direkt ausgeführt wird, die Hauptfunktion aufrufen
if __name__ == "__main__":
    main()
