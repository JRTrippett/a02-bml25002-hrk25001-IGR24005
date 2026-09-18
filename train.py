"""A02 Ping Pong - California Housing MLPRegressor."""
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    alpha=1e-3,
    learning_rate_init=1e-3,
    early_stopping=True,
    max_iter=500,
    random_state=42,
)
model.fit(X_train_s, y_train)

y_train_pred = model.predict(X_train_s)
y_test_pred = model.predict(X_test_s)

def plot_pred(y_true, y_pred, title, path):
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=8, alpha=0.3)
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="perfect")
    plt.xlabel("Actual (median house value, $100k)")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

plot_pred(
    y_train, 
    y_train_pred,
    "Actual vs Predicted - Train", 
    "figures/train_actual_vs_pred.png"
)
plot_pred(
    y_test, 
    y_test_pred,
    "Actual vs Predicted - Test", 
    "figures/test_actual_vs_pred.png"
)

print(f"Train R2: {model.score(X_train_s, y_train):.3f}")
print(f"Test  R2: {model.score(X_test_s, y_test):.3f}")

print(f"Train MAE: {mean_absolute_error(y_train, y_train_pred):.3f}")
print(f"Test  MAE: {mean_absolute_error(y_test, y_test_pred):.3f}")
