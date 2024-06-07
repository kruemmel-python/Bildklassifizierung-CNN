# Importieren des Moduls zur Interaktion mit dem Betriebssystem
import os  

# Importieren der Bibliothek für wissenschaftliches Rechnen in Python
import numpy as np  

# Importieren von Funktionen und Klassen aus TensorFlow und Keras zur Bildverarbeitung und zum Erstellen von Modellen
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from typing import Dict, Tuple

# Datenanreicherung: Erzeugt Variationen der Trainingsbilder zur Verbesserung der Generalisierung des Modells
# rescale: Skalierung der Bildpixelwerte auf den Bereich [0, 1]
# rotation_range: Zufällige Rotationen der Bilder um bis zu 20 Grad
# width_shift_range: Zufällige horizontale Verschiebung der Bilder um bis zu 20% der Breite
# height_shift_range: Zufällige vertikale Verschiebung der Bilder um bis zu 20% der Höhe
# shear_range: Zufällige Schertransformationen der Bilder
# zoom_range: Zufälliger Zoom der Bilder um bis zu 20%
# horizontal_flip: Zufälliges horizontales Spiegeln der Bilder
# fill_mode: Methode zum Auffüllen von Pixeln, die durch Transformationen verloren gehen
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Festlegen der Bildgröße als ein Tuple von Breite und Höhe in Pixeln
image_size = (512, 512)

# Ermitteln des Basisverzeichnisses, in dem sich dieses Skript befindet
base_dir = os.path.dirname(os.path.abspath(__file__))

# Definieren eines Dictionaries, das die Namen der Ordner (Schlüssel) und deren Pfade (Werte) enthält
folders = {
    'Gemischt': os.path.join(base_dir, 'fzn'),
    'greenscreen': os.path.join(base_dir, 'fzgs')
}

# Funktion zum Laden der Daten und Labels aus den angegebenen Ordnern
# Die Funktion nimmt ein Dictionary von Ordnerpfaden und gibt ein Tuple von NumPy-Arrays (Bilder und Labels) zurück
def load_data(folders: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    images = []  # Liste zum Speichern der Bilder
    labels = []  # Liste zum Speichern der zugehörigen Labels
    # Durchlaufen der Ordner und deren Dateien
    for label, folder in enumerate(folders.values()):
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)  # Erstellen des vollständigen Pfads zum Bild
            img = load_img(img_path, target_size=image_size)  # Laden des Bildes und Anpassen der Größe
            img_array = img_to_array(img)  # Konvertieren des Bildes in ein NumPy-Array
            images.append(img_array)  # Hinzufügen des Bildarrays zur Liste der Bilder
            labels.append(label)  # Hinzufügen des Labels zur Liste der Labels
    images = np.array(images)  # Konvertieren der Liste der Bilder in ein NumPy-Array
    labels = np.array(labels)  # Konvertieren der Liste der Labels in ein NumPy-Array
    # Rückgabe der Bilder und Labels als von ImageDataGenerator erzeugter Datenstrom
    return datagen.flow(images, labels, batch_size=32)

# Funktion zum Erstellen eines Keras-Modells
# Die Funktion nimmt die Eingabeform als Tuple von drei Werten (Höhe, Breite, Kanäle) und gibt ein Keras Sequential-Modell zurück
def create_model(input_shape: Tuple[int, int, int]) -> Sequential:
    model = Sequential([  # Erstellen eines Sequenziellen Modells
        # Hinzufügen einer 2D-Faltungsschicht mit 32 Filtern, einer Filtergröße von 3x3 und ReLU-Aktivierungsfunktion
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        # Hinzufügen einer Max-Pooling-Schicht mit einer Poolgröße von 2x2
        MaxPooling2D((2, 2)),
        # Flatten-Schicht zum Umwandeln der 2D-Features in 1D-Feature-Vektoren
        Flatten(),
        # Dense-Schicht (voll verbundene Schicht) mit 128 Neuronen und ReLU-Aktivierungsfunktion
        Dense(128, activation='relu'),
        # Dense-Schicht für die Klassifikation, Anzahl der Neuronen entspricht der Anzahl der Ordner, Softmax-Aktivierung für Wahrscheinlichkeitsverteilung
        Dense(len(folders), activation='softmax')
    ])
    return model

# Funktion zum Kompilieren des Modells
# Die Funktion nimmt ein Keras Sequential-Modell und kompiliert es mit dem Adam-Optimizer und einer Verlustfunktion für mehrklassige Klassifikation
def compile_model(model: Sequential) -> None:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Hauptfunktion des Skripts
def main():
    # Laden der Trainingsdaten aus den angegebenen Ordnern
    train_data_gen = load_data(folders)
    
    # Erstellen des Modells mit der angegebenen Eingabeform (Höhe, Breite, 3 Farbkanäle)
    model = create_model((image_size[0], image_size[1], 3))
    
    # Kompilieren des Modells
    compile_model(model)
    
    # Trainieren des Modells mit den Trainingsdaten für 10 Epochen
    model.fit(train_data_gen, epochs=10)
    
    # Speichern des trainierten Modells in einer Datei
    model.save('images.keras')

# Überprüfen, ob dieses Skript direkt ausgeführt wird (nicht importiert)
if __name__ == "__main__":
    main()  # Aufrufen der Hauptfunktion
