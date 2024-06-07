# Importieren des Moduls zur Interaktion mit dem Betriebssystem
import os

# Importieren der Image-Klasse aus der Pillow-Bibliothek, die für die Bildverarbeitung verwendet wird
from PIL import Image

# Pfade zum Eingabe- und Ausgabeordner
# 'input_folder' ist der Ordner, der die Originalbilder enthält
# 'output_folder' ist der Ordner, in den die skalierten Bilder gespeichert werden
input_folder = r'D:\images\gemischt'
output_folder = r'C:\Users\ralfk\source\Repos\Bildklassifizierung-CNN\fzn'

# Erstellen des Ausgabeordners, falls er nicht existiert
os.makedirs(output_folder, exist_ok=True)

# Zielgröße für die Bilder, die auf 512x512 Pixel festgelegt ist
# Dies entspricht der Bildgröße, die im Modelltraining verwendet wird (im BE.py Code definiert als 'image_size')
target_size = (512, 512)

# Zähler, um die Bilder fortlaufend zu nummerieren
counter = 1

# Durchlaufen aller Dateien im Eingabeordner
for filename in os.listdir(input_folder):
    # Überprüfen, ob die Datei eine Bilddatei ist, indem die Dateiendung geprüft wird
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        img_path = os.path.join(input_folder, filename)  # Erstellen des vollständigen Pfads zur Bilddatei
        img = Image.open(img_path)  # Öffnen des Bildes
        # Skalieren des Bildes auf die Zielgröße unter Verwendung des LANCZOS-Resampling-Filters für hohe Qualität
        img_resized = img.resize(target_size, Image.LANCZOS)
        
        # Erstellen des vollständigen Pfads zur Ausgabe-Bilddatei mit fortlaufender Nummerierung
        output_path = os.path.join(output_folder, f'{counter}.jpg')
        img_resized.save(output_path)  # Speichern des skalierten Bildes im Ausgabeordner
        print(f'Saved resized image to {output_path}')  # Ausgabe einer Bestätigungsmeldung
        counter += 1  # Erhöhen des Zählers für die nächste Datei

# Ausgabe einer Abschlussmeldung, nachdem alle Bilder verarbeitet wurden
print('Alle Bilder wurden konvertiert und gespeichert.')
