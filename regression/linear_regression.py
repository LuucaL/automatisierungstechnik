"""
Simple linear regression module for predicting bottle weights based on vibration data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data(file_path="X.csv"):
    """Load data from CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        print("Creating sample data instead...")
        return generate_sample_data()

def generate_sample_data(n_samples=100):
    """Generate sample data for demonstration."""
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        bottle_id = f"bottle_{i+1}"
        
        # Generate vibration index (main predictor)
        vibration_index_red = np.random.normal(100, 20)
        vibration_index_blue = np.random.normal(100, 15) 
        vibration_index_green = np.random.normal(100, 25)
        
        # Generate fill levels with relationship to vibration
        fill_level_red = 450 - 0.5 * vibration_index_red + np.random.normal(0, 20)
        fill_level_blue = 450 - 0.5 * vibration_index_blue + np.random.normal(0, 15)
        fill_level_green = 450 - 0.5 * vibration_index_green + np.random.normal(0, 25)
        
        # Generate temperatures
        temp_red = np.random.normal(35, 2)
        temp_blue = np.random.normal(34, 1.5)
        temp_green = np.random.normal(33, 2.5)
        
        data.append({
            'bottle': bottle_id,
            'vibration_index_red': vibration_index_red,
            'vibration_index_blue': vibration_index_blue,
            'vibration_index_green': vibration_index_green,
            'fill_level_grams_red': fill_level_red,
            'fill_level_grams_blue': fill_level_blue,
            'fill_level_grams_green': fill_level_green,
            'temperature_red': temp_red,
            'temperature_blue': temp_blue,
            'temperature_green': temp_green
        })
    
    return pd.DataFrame(data)

def train_linear_regression(X, y):
    """Train a simple linear regression model."""
    model = LinearRegression()
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    """Evaluate the regression model."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Create formula string
    if X.shape[1] == 1:  # Single feature
        formula = f"y = {model.coef_[0]:.4f} * x + {model.intercept_:.4f}"
        print(f"Linear Regression Formula: {formula}")
    
    return mse, r2, y_pred

def plot_results(y_true, y_pred, feature_name):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    
    plt.title('Tatsächliche vs. Vorhergesagte Werte')
    plt.xlabel('Tatsächliche Werte')
    plt.ylabel('Vorhergesagte Werte')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function for regression analysis."""
    print("=== Einfache Lineare Regression für Füllstandsvorhersage ===\n")
    
    # Load data
    print("1. Lade Daten...")
    data = load_data()
    print(f"Datensatz mit {len(data)} Proben geladen")
    
    # Use vibration_index_blue to predict fill_level_grams_blue as specified in the assignment
    X = data[["vibration_index_blue"]]
    y = data["fill_level_grams_blue"]
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Trainingsdaten: {X_train.shape[0]} Proben")
    print(f"Testdaten: {X_test.shape[0]} Proben")
    
    # Train model
    model = train_linear_regression(X_train, y_train)
    
    # Evaluate on training data
    train_mse, train_r2, y_train_pred = evaluate_model(model, X_train, y_train)
    
    # Evaluate on test data
    test_mse, test_r2, y_test_pred = evaluate_model(model, X_test, y_test)
    
    # Plot results
    plot_results(y_test, y_test_pred, "vibration_index_blue")
    
    # Make predictions on new data if available
    try:
        new_data = pd.read_csv("x.csv")
        X_new = new_data[["vibration_index_blue"]]
        y_hat = model.predict(X_new)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            "Flaschen ID": new_data["id"] if "id" in new_data.columns else range(1, len(y_hat) + 1),
            "y_hat": y_hat
        })
        
        output_filename = "reg_jul-52315817-52315856.csv"
        predictions_df.to_csv(output_filename, index=False)
        print(f"Vorhersagen gespeichert in {output_filename}")
    except FileNotFoundError:
        print("Datei x.csv nicht gefunden. Überspringe Vorhersagen auf neuen Daten.")
    

if __name__ == "__main__":
    main()
