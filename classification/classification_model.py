"""
Einfaches Klassifikationsmodell für Fehlererkennung bei Flaschen anhand von Vibrationsdaten.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Einfache Funktion zur Generierung von Testdaten
def generate_vibration_data(n_samples=100):
    """Generiert einfache Vibrationsdaten für normale und defekte Flaschen."""
    
    data = []
    
    for i in range(n_samples):
        # ID für jede Flasche
        bottle_id = f"bottle_{i+1}"
        
        # Simuliere ob Flasche defekt ist (ca. 20% der Flaschen)
        is_cracked = np.random.random() < 0.2
        
        # Grundlegende Vibrationsmuster
        base_pattern = np.random.normal(100, 10, 100)  # 100 Zeitpunkte
        
        if is_cracked:
            # Defekte Flaschen haben höhere und unregelmäßigere Vibrationswerte
            vibration_values = base_pattern + np.random.normal(30, 20, 100)
            # Füge einige "Spikes" hinzu
            spike_points = np.random.choice(range(100), 5)
            vibration_values[spike_points] += np.random.normal(50, 20, 5)
        else:
            # Normale Flaschen haben gleichmäßigere Vibrationswerte
            vibration_values = base_pattern + np.random.normal(0, 5, 100)
        
        # Stelle sicher, dass Werte nicht negativ sind
        vibration_values = np.maximum(0, vibration_values)
        
        data.append({
            'bottle': bottle_id,
            'is_cracked': is_cracked,
            'vibration_values': vibration_values
        })
    
    return pd.DataFrame(data)

# Funktion zum Extrahieren einfacher Features aus Vibrationsdaten
def extract_vibration_features(data):
    """Extrahiert einfache statistische Features aus Vibrationszeitreihen."""
    
    features = []
    
    for idx, row in data.iterrows():
        vibration_values = row['vibration_values']
        
        # Extrahiere statistische Features
        feature_dict = {
            'bottle': row['bottle'],
            'is_cracked': row['is_cracked'],
            'mean': np.mean(vibration_values),
            'std': np.std(vibration_values),
            'max': np.max(vibration_values),
            'min': np.min(vibration_values),
            'median': np.median(vibration_values),
            'range': np.max(vibration_values) - np.min(vibration_values),
            'q25': np.percentile(vibration_values, 25),
            'q75': np.percentile(vibration_values, 75)
        }
        
        features.append(feature_dict)
    
    return pd.DataFrame(features)

def main():
    """Hauptfunktion für die Klassifikationsanalyse."""
    
    # Generiere Testdaten
    data = generate_vibration_data(500)
    print(f"Datensatz mit {len(data)} Proben generiert")
    print(f"Klassenverteilung:")
    print(data['is_cracked'].value_counts())
    print(f"Defekte Flaschen: {data['is_cracked'].sum()}/{len(data)} ({100*data['is_cracked'].mean():.1f}%)")
    
    # Extrahiere Features
    features_df = extract_vibration_features(data)
    print(f"Features extrahiert. Beispiele der Features:")
    print(features_df[['mean', 'std', 'max', 'min', 'range']].head())

    # Analysiere Vibrationsmuster
    plt.figure(figsize=(15, 5))
    
    # Plotte Beispiel-Vibrationsmuster
    plt.subplot(1, 3, 1)
    normal_samples = data[data['is_cracked'] == False]['vibration_values'].iloc[:3]
    for i, pattern in enumerate(normal_samples):
        plt.plot(pattern, alpha=0.7, label=f'Normal {i+1}')
    plt.title('Normale Flaschen - Vibrationsmuster')
    plt.xlabel('Zeit')
    plt.ylabel('Vibration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    cracked_samples = data[data['is_cracked'] == True]['vibration_values'].iloc[:3]
    for i, pattern in enumerate(cracked_samples):
        plt.plot(pattern, alpha=0.7, label=f'Defekt {i+1}', color='red')
    plt.title('Defekte Flaschen - Vibrationsmuster')
    plt.xlabel('Zeit')
    plt.ylabel('Vibration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.hist(features_df[features_df['is_cracked'] == False]['mean'], 
            alpha=0.7, label='Normal', bins=20)
    plt.hist(features_df[features_df['is_cracked'] == True]['mean'], 
            alpha=0.7, label='Defekt', bins=20, color='red')
    plt.title('Verteilung der mittleren Vibration')
    plt.xlabel('Mittlere Vibration')
    plt.ylabel('Anzahl')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vibration_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Erstelle Trainings- und Testdaten
    X = features_df.drop(['bottle', 'is_cracked'], axis=1)
    y = features_df['is_cracked']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Trainingsdaten: {X_train.shape[0]} Proben")
    print(f"Testdaten: {X_test.shape[0]} Proben")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Vorhersagen
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluiere Modell
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"F1-Score (Training): {train_f1:.4f}")
    print(f"F1-Score (Test): {test_f1:.4f}")
    
    print("\nKlassifikationsreport:")
    print(classification_report(y_test, y_test_pred, target_names=['Normal', 'Defekt']))
    
    # Erstelle Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Normal', 'Defekt'], yticklabels=['Normal', 'Defekt'])
    
    plt.title(f"Confusion Matrix - F1-Score: {test_f1:.4f}")
    plt.xlabel("Vorhergesagt")
    plt.ylabel("Tatsächlich")
    plt.tight_layout()
    
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
