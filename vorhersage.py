from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


# Modell laden
model = load_model('images.keras')
 # Bildgröße definieren
image_size = (512, 512)  # Beispielgröße, anpassen nach Bedarf
# Neues Bild laden und vorbereiten
img_path = 'D:\\images\\fzgs\\43.jpg'  # Pfad zum neuen Bild
img = load_img(img_path, target_size=image_size)  # Bildgröße muss mit dem trainierten Modell übereinstimmen
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Bild zu einem Batch hinzufügen
img_array /= 255.  # Normalisierung, wie im ImageDataGenerator

# Vorhersage machen
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)  # Klasse mit der höchsten Wahrscheinlichkeit


# Klassenbezeichnungen für bessere Lesbarkeit
klassen_namen = ['nicht greenscreen fähig', 'greenscreen fähig']

# Vorhergesagte Klasse ausgeben
if predicted_class[0] == 0:
    print(f"Das Modell sagt Klasse {predicted_class[0]} für das Bild voraus. Es ist {klassen_namen[0]}!")
else:
    print(f"Das Modell sagt Klasse {predicted_class[0]} für das Bild voraus. Es ist {klassen_namen[1]}!")


