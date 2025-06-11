import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("X.csv")

X = df[["vibration_index_blue"]]  # nur diese Spalte wie in der Aufgabenstellung
y = df["fill_level_grams_blue"]    # updated target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print("Formel: y = {:.4f} * x + {:.4f}".format(model.coef_[0], model.intercept_))
print("MSE (Train):", mse_train)
print("MSE (Test):", mse_test)

df_x = pd.read_csv("x.csv")
X_new = df_x[["vibration_index_blue"]]
y_hat = model.predict(X_new)

df_out = pd.DataFrame({
    "Flaschen ID": df_x["id"],  # falls die Datei eine ID-Spalte hat
    "y_hat": y_hat
})

df_out.to_csv("reg_12345678-87654321-11223344.csv", index=False)
print("Vorhersagen gespeichert in reg_12345678-87654321-11223344.csv")