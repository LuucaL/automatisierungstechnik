import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    # ===============================================
    # Neuer Abschnitt: Regressionsmodell für Endgewicht
    # Laden des Datensets X.csv
    data = pd.read_csv(r"c:\dev\automatisierung\automatisierungstechnik\X.csv")
    print("Datensatz X.csv geladen. Anzahl der Zeilen:", len(data))
    # Überprüfen, ob die erforderlichen Spalten vorhanden sind
    required_columns = ["bottle", "vibration_index_blue", "fill_level_grams_blue"]
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Fehlende erforderliche Spalten im Datensatz: " + ", ".join(required_columns))
    # Auswahl des Features und der Zielvariable
    X_feat = data[["vibration_index_blue"]]
    y_target = data["fill_level_grams_blue"]
    # Aufteilen in Trainings- und Testdaten (80/20)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_feat, y_target, test_size=0.2, random_state=42)
    # Training des linearen Regressionsmodells
    model2 = LinearRegression()
    model2.fit(X_train2, y_train2)
    # Ermitteln der Modellparameter
    m = model2.coef_[0]
    b = model2.intercept_
    print("Regression Formel: y = {:.4f} * x + {:.4f}".format(m, b))
    # Berechnung des Mean Squared Error (MSE)
    y_pred_train = model2.predict(X_train2)
    y_pred_test = model2.predict(X_test2)
    mse_train = mean_squared_error(y_train2, y_pred_train)
    mse_test = mean_squared_error(y_test2, y_pred_test)
    print("Training MSE: {:.4f}".format(mse_train))
    print("Test MSE: {:.4f}".format(mse_test))
    # Prognose für das gesamte X.csv-Datenset
    data["y_hat"] = model2.predict(X_feat)
    # Speichern der Prognose als CSV (Spalten: Flaschen ID, y_hat)
    output = data[["bottle", "y_hat"]].rename(columns={"bottle": "Flaschen ID"})
    output.to_csv(r"c:\dev\automatisierung\automatisierungstechnik\reg_12345678-87654321-11223344.csv", index=False)
    print("Prognose gespeichert in reg_12345678-87654321-11223344.csv")

if __name__ == "__main__":
    main()
