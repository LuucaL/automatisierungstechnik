# Automatisierungstechnik - Datenverarbeitung Projekt

## Projektübersicht

Dieses Projekt implementiert ein einfaches System zur Datenspeicherung und Visualisierung für industrielle Sensordaten. Es umfasst CSV-basierte Datenspeicherung, einen MQTT-Client zum Datensammeln, Datenvisualisierung, Regressionsmodelle und Klassifikationsanalysen.

## Projektstruktur

```
automatisierungstechnik/
├── README.md                      # Diese Datei - Projektdokumentation
├── requirements.txt               # Erforderliche Pakete für die Installation
├── X.csv                          # Testdaten für Regression
├── prediction.py                  # Einfaches Regressionsmodell zur Vorhersage des Füllstands
├── database/                      # Modul für Datenspeicherung und -transformation
│   ├── __init__.py
│   ├── database.py                # CSV-Datenspeicherung und Testdatengenerierung
│   ├── transform.py               # Funktionen für Datentransformation
│   ├── data.csv                   # Gespeicherte Sensordaten
│   ├── classification_data.json   # Daten für die Klassifikationsaufgaben
│   └── regression_data.json       # Daten für die Regressionsaufgaben
├── mqtt_client/                   # Modul für die MQTT-Kommunikation
│   ├── __init__.py
│   └── mqtt_client.py             # MQTT-Client für Datenabruf von Sensoren
├── visualisierung/                # Modul für Datenvisualisierung
│   ├── __init__.py
│   └── visualisierung.py          # Funktionen für Zeitreihen-Visualisierung
├── classification/                # Modul für Klassifikationsanalysen
│   ├── __init__.py
│   ├── classification_model.py    # Modell zur Erkennung von defekten Flaschen
│   └── classification_analysis.ipynb # Jupyter Notebook zur Klassifikationsanalyse
├── regression/                    # Modul für Regressionsanalysen
│   ├── __init__.py
│   ├── linear_regression.py       # Implementierung eines linearen Regressionsmodells
│   ├── regression_analysis.ipynb  # Jupyter Notebook zur Regressionsanalyse
│   └── X.csv                      # Lokale Kopie der Testdaten
└── data/                          # Ausgabedaten und Visualisierungen
    ├── confusion_matrix.png       # Visualisierung der Klassifikationsergebnisse
    ├── vibration_analysis.png     # Visualisierung der Vibrationsdaten
    ├── regression_results.png     # Visualisierung der Regressionsergebnisse
    └── reg_*.csv                  # Generierte Vorhersagedateien
```

## Funktionen

### Datenspeicherung (`database/database.py`)
- CSV-Datei als einfache Datenbank für Sensordaten
- Speichern von Sensordaten mit Zeitstempel, Sensortyp, Wert, Einheit und Geräte-ID
- Abrufen von Daten mit optionaler Filterung nach Sensortyp und Zeitraum
- Generieren von realistischen Testdaten für Entwicklung und Tests

### Datentransformation (`database/transform.py`)
- Berechnung gleitender Durchschnitt für Zeitreihendaten
- Einheitenumrechnung zwischen verschiedenen Messgrößen
- Aggregation von Daten für Analysen

### MQTT-Client (`mqtt_client/mqtt_client.py`)
- Verbindung zu einem MQTT-Broker für IoT-Kommunikation
- Abonnieren von Sensordaten-Topics für verschiedene Sensoren
- Automatisches Speichern eingehender Daten in der CSV-Datenbank
- Verarbeitung von JSON- und Roh-Payloads von MQTT-Nachrichten

### Datenvisualisierung (`visualisierung/visualisierung.py`)
- Zeitreihen-Plots für einzelne Sensoren zur Datenanalyse
- Vergleichsplots für mehrere Sensoren gleichzeitig
- Speichern von Plots als Bilddateien für Berichte

### Klassifikation (`classification/classification_model.py`)
- Erkennung defekter Flaschen basierend auf Vibrationsdaten
- Extraktion relevanter Features aus Vibrationsdaten
- Training eines Random-Forest-Klassifikators
- Evaluation mit F1-Score und Konfusionsmatrix

### Regression (`regression/linear_regression.py` und `prediction.py`)
- Lineares Regressionsmodell zur Vorhersage des Füllstands basierend auf Vibrationsdaten
- Training und Evaluierung mit Testdaten
- Berechnung des mittleren quadratischen Fehlers (MSE)
- Anwendung des Modells auf neue Daten zur Vorhersage

## Installation und Ausführung

1. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

2. Datenbank initialisieren und mit Testdaten füllen:
```bash
python database/database.py
```

3. Visualisierung der Daten:
```bash
python visualisierung/visualisierung.py
```

4. MQTT-Client starten (benötigt einen laufenden MQTT-Broker):
```bash
python mqtt_client/mqtt_client.py
```

5. Regressionsmodell ausführen:
```bash
python prediction.py
```

6. Klassifikationsmodell ausführen:
```bash
python classification/classification_model.py
```

## Datenstruktur

Das System arbeitet mit folgenden Sensordaten:
- Temperatur (°C) - Messung der Umgebungstemperatur an verschiedenen Geräten
- Vibrations-Index - Messwert für die Vibration an Produktionsgeräten
- Füllstand (Gramm) - Gewichtsmessung des Inhalts in den Spendern

Jeder Messwert enthält:
- Zeitstempel (ISO-Format) - Genaue Zeit der Messung
- Sensortyp (z.B. "temperature_red", "vibration_index_blue") - Identifikation des Sensors
- Wert (numerisch) - Der gemessene Wert
- Einheit (z.B. "C", "grams") - Maßeinheit des Messwerts
- Geräte-ID (z.B. "dispenser_red") - Identifikation des Messgeräts

## Ausgabedateien

Die Ausgabedateien des Projekts werden in folgenden Ordnern gespeichert:

- **Sensordaten**: `database/data.csv` - CSV-Datei mit allen gespeicherten Sensormesswerten
- **Visualisierungen**: `data/` - Enthält generierte Diagramme als PNG-Dateien
- **Vorhersagen**: 
  - `reg_*.csv` - Regressionsvorhersagen (Füllstand basierend auf Vibrationsdaten)
  - `data/confusion_matrix.png` - Visualisierung der Klassifikationsergebnisse

## Jupyter Notebooks

Für eine interaktive Analyse der Daten stehen folgende Jupyter Notebooks zur Verfügung:

- **Regressionsanalyse**: `regression/regression_analysis.ipynb`
- **Klassifikationsanalyse**: `classification/classification_analysis.ipynb`
