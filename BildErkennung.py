import os  # Das Modul 'os' ermöglicht die Interaktion mit dem Betriebssystem.
import numpy as np  # 'numpy' ist eine Bibliothek für wissenschaftliches Rechnen in Python.

# Importieren von Keras-Funktionen für das Laden und Verarbeiten von Bildern sowie für das Erstellen von Modellen.
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Datenanreicherung: Erzeugt Variationen der Trainingsbilder zur Verbesserung der Generalisierung des Modells.
datagen = ImageDataGenerator(
    rescale=1./255,  # Skaliert die Pixelwerte auf das Intervall [0, 1].
    rotation_range=20,  # Erlaubt zufällige Rotationen des Bildes bis zu 20 Grad.
    width_shift_range=0.2,  # Erlaubt zufällige horizontale Verschiebungen bis zu 20% der Bildbreite.
    height_shift_range=0.2,  # Erlaubt zufällige vertikale Verschiebungen bis zu 20% der Bildhöhe.
    shear_range=0.2,  # Erlaubt zufällige Schertransformationen.
    zoom_range=0.2,  # Erlaubt zufälliges Zoomen des Bildes.
    horizontal_flip=True,  # Erlaubt das zufällige Spiegeln des Bildes horizontal.
    fill_mode='nearest'  # Bestimmt, wie neu entstandene Pixel nach Transformationen gefüllt werden.
)

# Festlegen der Bildgröße und der Pfade zu den Bildordnern.
image_size = (512, 512)  # Die Größe der Bilder, die geladen werden sollen.
folders = {
    'Gemischt': r'D:\images\fzn',  # Pfad zum Ordner mit gemischten Bildern.
    'greenscreen': r'D:\images\fzgs'}  # Pfad zum Ordner mit Greenscreen-Bildern.

# Funktion zum Laden der Daten und Labels.
def load_data(folders):
    images = []  # Liste für die Bilder.
    labels = []  # Liste für die Labels.
    for label, folder in enumerate(folders.values()):  # Iteriert über die Ordner und deren Labels.
        for file in os.listdir(folder):  # Iteriert über alle Dateien im Ordner.
            img_path = os.path.join(folder, file)  # Erstellt den vollständigen Pfad zur Bilddatei.
            img = load_img(img_path, target_size=image_size)  # Lädt das Bild und passt die Größe an.
            img_array = img_to_array(img)  # Konvertiert das Bild in ein NumPy-Array.
            images.append(img_array)  # Fügt das Array zur Liste der Bilder hinzu.
            labels.append(label)  # Fügt das Label zur Liste der Labels hinzu.
    return np.array(images), np.array(labels)  # Gibt die Listen als NumPy-Arrays zurück.

# Erstellen des Modells mit Keras.
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),  # Konvolutionsschicht.
    MaxPooling2D((2, 2)),  # Pooling-Schicht zur Reduzierung der räumlichen Dimensionen.
    Flatten(),  # Schicht zum Abflachen des Tensors in einen Vektor.
    Dense(128, activation='relu'),  # Vollverbundene Schicht.
    Dense(3, activation='softmax')  # Ausgabeschicht für 3 Klassen mit Softmax-Aktivierung.
])

# Kompilieren des Modells mit dem Adam-Optimierer und der Sparse-Categorical-Crossentropy als Verlustfunktion.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Laden der Trainingsdaten.
train_images, train_labels = load_data(folders)

# Trainieren des Modells mit den geladenen Daten.
model.fit(train_images, train_labels, epochs=10)  # 'epochs' gibt an, wie oft das Training durchgeführt wird.

# Speichern des trainierten Modells.
model.save('images.keras')
