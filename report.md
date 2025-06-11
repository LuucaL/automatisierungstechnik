# Projektstruktur und Beschreibung

```
code/1_persitierung/
├── visualisierung/
│   ├── visualisierung.py    # Enthält Funktionen zur Darstellung und Analyse von Visualisierungen
│   └── __init__.py          # Ermöglicht den Import innerhalb des Ordners
├── database/
│   ├── database.py          # Stellt Funktionen für den Datenbankzugriff bereit
│   ├── transform.py         # Beinhaltet Logik zur Transformation von Daten
│   ├── data.csv             # Beispiel-Datensatz für Datenbanktests
│   └── __init__.py          # Ermöglicht den Import innerhalb des Ordners
├── mqtt_client/
│   ├── mqtt_client.py       # Enthält Funktionen zur MQTT-Kommunikation
│   └── __init__.py          # Ermöglicht den Import innerhalb des Ordners
├── README.md                # Allgemeine Projektinformationen und Anleitung
└── requirements.txt         # Listet die benötigten Python-Pakete für das Projekt auf
```

## Kapitel: Regressionsmodell für Endgewicht

### Übersicht
- **Genutzte Spalten (X):** vibration_index_blue  
- **Zielvariable (y):** fill_level_grams_blue  
- **Modell-Typ:** Lineare Regression  

### Evaluierung
| Metrik              | Trainings-MSE | Test-MSE  |
|---------------------|---------------|-----------|
| Lineare Regression  |  (siehe prediction.py-Ausgabe) | (siehe prediction.py-Ausgabe) |

### Regressionsformel
Die ermittelte Regressionsformel lautet:  
**y = m * x + b**  
mit:  
- m = *[Wert von model2.coef_]*  
- b = *[Wert von model2.intercept_]*  

Beispielausgabe:  
`y = 0.1234 * x + 45.6789`

### Prognose
Für das Datenset X.csv wurde eine Prognose erstellt. Die Ergebnisse (Flaschen ID und y_hat) wurden in der Datei  
`reg_12345678-87654321-11223344.csv`  
gespeichert.

*Hinweis: Die exakten MSE-Werte und Parameter (m, b) entnehmen Sie bitte der Konsolenausgabe von prediction.py.*
