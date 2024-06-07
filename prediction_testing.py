import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Modell laden
model = load_model('images.keras')

# Liste der Bildpfade
image_paths = ['D:\\images\\test3.jpg', 'D:\\images\\test2.jpg', 'D:\\images\\test1.jpg', 'D:\\images\\test.jpg']  # usw.

# Bildgröße definieren (muss mit dem trainierten Modell übereinstimmen)
image_size = (512, 512)

# Klassenbezeichnungen für bessere Lesbarkeit
klassen_namen = ['nicht greenscreen fähig', 'greenscreen fähig']

# Vorhersagen für jedes Bild machen
for img_path in image_paths:
    img = load_img(img_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Bild zu einem Batch hinzufügen
    img_array /= 255.  # Normalisierung, wie im ImageDataGenerator

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)  # Klasse mit der höchsten Wahrscheinlichkeit

    # Vorhergesagte Klasse ausgeben
    if predicted_class[0] == 0:
        print(f"Das Modell sagt Klasse {predicted_class[0]} für das Bild {img_path} voraus. Es ist {klassen_namen[0]}!")
    else:
        print(f"Das Modell sagt Klasse {predicted_class[0]} für das Bild {img_path} voraus. Es ist {klassen_namen[1]}!")

