import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Datenanreicherung
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisierung
    rotation_range=20,  # Zufällige Rotation des Bildes
    width_shift_range=0.2,  # Zufällige horizontale Verschiebung
    height_shift_range=0.2,  # Zufällige vertikale Verschiebung
    shear_range=0.2,  # Scherintensität
    zoom_range=0.2,  # Zufälliges Zoomen
    horizontal_flip=True,  # Zufälliges Spiegeln
    fill_mode='nearest'  # Füllmodus für neu erstellte Pixel
)

# Bildgröße und Pfade definieren
image_size = (512, 512)  # Beispielgröße, anpassen nach Bedarf
folders = {
    'Gemischt': r'D:\images\fzn',
    'greenscreen': r'D:\images\fzgs'}

# Daten und Labels laden
def load_data(folders):
    images = []
    labels = []
    for label, folder in enumerate(folders.values()):
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

# Modell erstellen
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 Klassen
])

# Modell kompilieren
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Daten laden
train_images, train_labels = load_data(folders)

# Modell trainieren
model.fit(train_images, train_labels, epochs=10)  # Anzahl der Epochen anpassen

# Modell speichern
model.save('images.keras')
