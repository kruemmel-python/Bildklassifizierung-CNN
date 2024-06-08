import os  # Modul zum Arbeiten mit dem Betriebssystem, z.B. zum Überprüfen von Dateipfaden
import numpy as np  # Bibliothek für numerische Berechnungen, insbesondere für Arrays
import cv2  # Bibliothek für die Bild- und Videobearbeitung
from typing import List, Union  # Hilft bei der Angabe von Datentypen in Funktionssignaturen

# Funktion zur Erstellung einer Maske für den Greenscreen-Bereich im Bild
def create_greenscreen_masks(image: np.ndarray, num_greenscreens: int = 2) -> List[np.ndarray]:
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
    
    # Komponenten analysieren und die größten Komponenten beibehalten
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
        raise ValueError("Number of videos must match number of greenscreen areas.")
    
    # Video-Writer initialisieren, um das Ausgabevideo zu speichern
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (original_img.shape[1], original_img.shape[0]))

    # Video-Captures für jedes Video öffnen
    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]
    if not all(cap.isOpened() for cap in caps):
        raise ValueError("Failed to open one or more video files.")
    
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

            for c in range(0, 3):
                result[y:y+h, x:x+w, c] = (
                    background_img_resized[:, :, c] * (mask_cropped / 255.0) + 
                    result[y:y+h, x:x+w, c] * (mask_inv / 255.0)
                )

        out.write(result)
    
    # Ressourcen freigeben
    for cap in caps:
        cap.release()
    out.release()

# Hauptfunktion, um das Skript auszuführen
def main():
    # Pfade zu den Eingabedateien und der Ausgabedatei
    original_image_path: Union[str, None] = 'jr.jpg'  # Pfad zum Bild mit Greenscreen
    video_paths: List[Union[str, None]] = ['j.mp4', 'r.mp4']  # Pfade zu den Hintergrundvideos
    output_video_path: Union[str, None] = 'output_2_maus.mp4'  # Pfad zum Ausgabevideo

    try:
        # Überprüfen, ob die Dateien existieren
        if not os.path.exists(original_image_path):
            raise FileNotFoundError(f"Original image file '{original_image_path}' not found.")
        for video_path in video_paths:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Background video file '{video_path}' not found.")
        
        # Originalbild laden
        original_img = cv2.imread(original_image_path)
        if original_img is None:
            raise ValueError(f"Failed to load original image from '{original_image_path}'.")

        # Masken für die Greenscreen-Bereiche erstellen
        masks = create_greenscreen_masks(original_img, num_greenscreens=len(video_paths))
        
        # Greenscreens durch die Hintergrundvideos ersetzen
        replace_greenscreens_with_videos(original_img, video_paths, masks, output_video_path)

        print(f'Result saved to {output_video_path}')
    except Exception as e:
        print(f"An error occurred: {e}")

# Wenn das Skript direkt ausgeführt wird, die Hauptfunktion aufrufen
if __name__ == "__main__":
    main()
