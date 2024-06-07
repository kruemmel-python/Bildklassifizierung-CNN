
![l1](https://github.com/kruemmel-python/Bildklassifizierung-CNN/assets/169469747/4f691426-3ba0-40f0-9850-b8e926714318)


https://github.com/kruemmel-python/Bildklassifizierung-CNN/assets/169469747/fa53ca2d-2324-4d73-bb3c-4e28ba12eaab


Anleitung zur Verwendung des Bildklassifizierungs-CNN-Projekts
Dieses Projekt verwendet Convolutional Neural Networks (CNNs) zur Bildklassifizierung und zum Einfügen von Bildern oder Videos in einen Greenscreen-Hintergrund. Folgen Sie den untenstehenden Schritten, um die verschiedenen Funktionen des Projekts zu nutzen.

Schritt 1: Installation der notwendigen Software
Repository klonen:
Laden Sie das Projekt von der Repository-Seite herunter und navigieren Sie in das Projektverzeichnis.

Virtuelle Umgebung einrichten (optional, aber empfohlen):
Erstellen und aktivieren Sie eine virtuelle Python-Umgebung, um Abhängigkeiten isoliert zu installieren.

Abhängigkeiten installieren:
Installieren Sie alle benötigten Bibliotheken, die im requirements.txt-File aufgelistet sind.

Schritt 2: Bildkonvertierung
Verwenden Sie das Bildkonvertierungsskript, um Ihre Bilder in das erforderliche Format für das Modell zu bringen. Dies umfasst das Anpassen der Bildgröße und die Normalisierung der Pixelwerte.

Schritt 3: Erstellen und Trainieren des Modells
Erstellen und trainieren Sie das CNN-Modell:

Laden Sie Ihren Datensatz.
Definieren Sie die Modellarchitektur.
Trainieren Sie das Modell über mehrere Epochen.
Das Skript speichert das trainierte Modell automatisch zur späteren Verwendung.
Alternativ können Sie ein bereits trainiertes Modell herunterladen, das für weiteres Training oder die direkte Nutzung verwendet werden kann. Das Modell ist etwa 3 GB groß und kann hier https://1drv.ms/u/s!AroxmBWhYNuLzk6WupG1isy0NcPk?e=up2Iup heruntergeladen werden.

Schritt 4: Testen der Modellvorhersagen
Überprüfen Sie die Leistung des Modells, indem Sie neue Bilder durch das Modell laufen lassen und die Vorhersagen analysieren. Dies hilft Ihnen, die Genauigkeit des Modells auf ungesehenen Daten zu beurteilen und mögliche Verbesserungsbereiche zu identifizieren.

Schritt 5: Einfügen eines Bildes in einen Greenscreen-Hintergrund
Nutzen Sie das entsprechende Skript, um ein Bild in einen Greenscreen-Hintergrund einzufügen. Das trainierte Modell erkennt den Bereich des Bildes, der in den Greenscreen-Hintergrund eingefügt werden soll.

Schritt 6: Einfügen eines Videos in einen Greenscreen-Hintergrund
Verwenden Sie das Videobearbeitungsskript, um ein Video in einen Greenscreen-Hintergrund einzufügen. Dies ist besonders nützlich für die Erstellung von Videos mit Spezialeffekten.

Schritt 7: Beispielvideo
Nutzen Sie das bereitgestellte Beispielvideo, um die Fähigkeiten des

Modells in einer realen Anwendung zu demonstrieren. Das Video zeigt, wie das Modell in der Praxis funktioniert, beispielsweise beim Einfügen in einen Greenscreen-Hintergrund.

Lizenz und Dokumentation
Stellen Sie sicher, dass Sie die Lizenzbedingungen des Projekts verstehen, und verwenden Sie die bereitgestellte Dokumentation, um detaillierte Anweisungen und Erklärungen zu den einzelnen Komponenten und deren Verwendung zu erhalten.

