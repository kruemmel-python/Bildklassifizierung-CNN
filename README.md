# das Modell kann hier heruntergeladen werden. Achtung Es ist bereits trainiert und ist 3 GB groß

https://1drv.ms/u/s!AroxmBWhYNuLzk4hn9CbWPVjzitm?e=nZp9aP

# BildErkennung und Vorhersage

Dieses Repository enthält 5 Python-Skripte zur Bildklassifizierung und Vorhersage Test, Nutzung des keras model um Bilder oder vidoes in den Greenscreen einzufügen mit einem Convolutional Neural Network (CNN) unter Verwendung von TensorFlow und Keras.
![l1](https://github.com/kruemmel-python/Bildklassifizierung-CNN/assets/169469747/4f691426-3ba0-40f0-9850-b8e926714318)


https://github.com/kruemmel-python/Bildklassifizierung-CNN/assets/169469747/fa53ca2d-2324-4d73-bb3c-4e28ba12eaab



## model_create_and_training.py

Das Skript `model_create_and_training.py` ist für das Training eines CNN-Modells verantwortlich. Es lädt Bilddaten aus zwei Ordnern, die gemischte Bilder und Greenscreen-Bilder enthalten, und verwendet Datenanreicherungstechniken, um die Robustheit des Modells zu verbessern.

### Hauptmerkmale:
- Datenanreicherung mit Rotation, Verschiebung, Scherung, Zoom und Spiegelung.
- Definition der Bildgröße und Pfade zu den Bildordnern.
- Laden von Daten und Labels aus den Ordnern.
- Erstellen eines sequentiellen Modells mit Convolutional, Pooling, Flatten und Dense Schichten.
- Kompilieren des Modells mit Adam-Optimierer und sparse_categorical_crossentropy als Verlustfunktion.
- Trainieren des Modells mit den geladenen Daten.
- Speichern des trainierten Modells.

## prediction_testing.py

Das Skript `prediction_testing.py` wird verwendet, um Vorhersagen mit dem trainierten Modell zu machen. Es lädt ein neues Bild, bereitet es vor und führt eine Vorhersage durch, um die Klasse des Bildes zu bestimmen.


### Hauptmerkmale:
- Laden des trainierten Modells.
- Vorbereiten eines neuen Bildes für die Vorhersage.
- Durchführen der Vorhersage und Ausgabe der vorhergesagten Klasse.

## Installation

Bevor Sie die Skripte ausführen, stellen Sie sicher, dass Sie die folgenden Bibliotheken installiert haben. Sie können sie mit `pip` installieren, dem Paketmanager für Python.

```bash
pip install numpy
pip install tensorflow
pip install keras


## Lizenz

Dieses Projekt ist lizenziert unter der MIT Lizenz - siehe die LICENSE Datei für Details.
