# CDA Tennis Trainer

CDA Tennis Trainer ist ein Projekt zur Analyse und Verbesserung von Tennisspieltechniken mithilfe von Computer Vision und maschinellem Lernen.

## Projektstruktur

Die Hauptverzeichnisse und -dateien des Projekts sind wie folgt:

- `Dashboard/` - Enthält den Code für das Dashboard. Dieses kann ausgeführt werden, um verschiedene Plots in einer interaktiven Umgebung zu visualisieren.
- `daten/` - Hier sind die Datensätze gespeichert, die für die Analysen verwendet werden.
- `EDA_Schlieren/` - Beinhaltet Data-Wrangling-Skripte sowie Analysen und Visualisierungen der vorhandenen Daten.
- `video_recognition/` - Enthält alle relevanten Python-Skripte sowie die YOLO-Modelldateien zur Objekterkennung in Videos.
- `videos_schlieren/` - Hier ist das Video gespeichert, das als Eingabedaten für die Videoanalyse genutzt wurde.
- `output/` - Enthält das verarbeitete Video mit der Objekterkennung als Ergebnis des YOLO-Prozesses.
- `requirements.txt` - Liste der benötigten Python-Pakete für die Ausführung des Projekts.
- `README.md` - Diese Datei mit Informationen zum Projekt.

## Installation

1. Klonen Sie das Repository:
   ```bash
   git clone https://github.com/ilyaskayihan/cda_tennis_trainer.git
   ```

2. Wechseln Sie in das Projektverzeichnis:
   ```bash
   cd cda_tennis_trainer
   ```

3. Erstellen Sie eine virtuelle Umgebung und aktivieren Sie sie:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Installieren Sie die erforderlichen Pakete:
   ```bash
   pip install -r requirements.txt
   ```

## Nutzung

- **Dashboard:** Führen Sie das Dashboard-Skript aus, um Plots und Analysen zu visualisieren.
- **Datenanalyse:** Nutzen Sie die Dateien im Verzeichnis `EDA_Schlieren/` zur Datenaufbereitung und Visualisierung.
- **Videoanalyse:** Verwenden Sie die Skripte im Verzeichnis `video_recognition/` zur Objekterkennung in den Videos.
- **Ergebnisse:** Das verarbeitete Video mit den erkannten Objekten finden Sie im Verzeichnis `output/`.

## Beitragende

- Adem Cutura und Ilyas Kayihan




