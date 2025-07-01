"""
Simple linear regression for vibration to fill level prediction.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("X.csv")

# Use only the required columns as specified in the assignment
X = df[["vibration_index_blue"]]  # Input feature
y = df["fill_level_grams_blue"]   # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on training and test data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate mean squared error
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Print results
print("Formel: y = {:.4f} * x + {:.4f}".format(model.coef_[0], model.intercept_))
print("MSE (Training):", mse_train)
print("MSE (Test):", mse_test)

# If X.csv exists, make predictions
try:
    df_x = pd.read_csv("X.csv")
    X_new = df_x[["vibration_index_blue"]]
    y_hat = model.predict(X_new)
    
    # Save predictions to a CSV file
    df_out = pd.DataFrame({
        "Flaschen ID": range(1, len(y_hat) + 1),  # Generate IDs if not present
        "y_hat": y_hat
    })
    
    output_filename = "reg_52315852-52315817-52315856.csv"
    df_out.to_csv(output_filename, index=False)
    print(f"Vorhersagen gespeichert in {output_filename}")
except FileNotFoundError:
    print("X.csv nicht gefunden. Keine Vorhersagen erstellt.")