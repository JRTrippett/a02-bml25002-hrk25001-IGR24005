"""A02 Ping Pong - California Housing MLPRegressor."""

import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Create folder for saved figures
os.makedirs("figures", exist_ok=True)


# 2. Load California Housing dataset
X, y = fetch_california_housing(return_X_y=True)


# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Standardize the predictors
scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# 5. Train MLPRegressor
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    alpha=1e-3,
    learning_rate_init=1e-3,
    early_stopping=True,
    max_iter=500,
    random_state=42,
)

model.fit(X_train_s, y_train)

# 6. Generating predictions
y_train_pred = model.predict(X_train_s)
y_test_pred = model.predict(X_test_s)


# 7. Calculating model evaluation metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)


print(f"Train R2:  {train_r2:.3f}")
print(f"Test R2:   {test_r2:.3f}")

print(f"Train MAE: {train_mae:.3f}")
print(f"Test MAE:  {test_mae:.3f}")


# 8. Actual vs. Predicted plotting function
def plot_predictions(y_true, y_pred, split_name, path):

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    padding = (hi - lo) * 0.03
    lo -= padding
    hi += padding

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_true, y_pred, s=12, alpha=0.35, edgecolors="none")

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Actual median house value ($100,000s)")

    ax.set_ylabel("Predicted median house value ($100,000s)")

    ax.set_title(f"{split_name}\n" f"R² = {r2:.3f} | MAE = {mae:.3f}")

    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")

    plt.close(fig)


# 9. Required train and test prediction plots
plot_predictions(y_train, y_train_pred, "Training Set: Actual vs. Predicted", "figures/train_actual_vs_pred.png",)

plot_predictions(y_test, y_test_pred, "Test Set: Actual vs. Predicted", "figures/test_actual_vs_pred.png",)


# 10. Training loss plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(model.loss_curve_, linewidth=2)

ax.set_xlabel("Training iteration")
ax.set_ylabel("Loss")
ax.set_title("MLP Training Loss")

ax.grid(True, alpha=0.2)

fig.tight_layout()

fig.savefig("figures/training_loss.png", dpi=160, bbox_inches="tight")

plt.close(fig)


# 11. Residual analysis
test_residuals = y_test - y_test_pred

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(y_test_pred, test_residuals, s=12, alpha=0.35, edgecolors="none")

ax.axhline(y=0, linestyle="--", linewidth=1.5)

ax.set_xlabel("Predicted median house value ($100,000s)")

ax.set_ylabel("Residual (Actual - Predicted)")

ax.set_title("Test Set Residual Analysis")

ax.grid(True, alpha=0.2)

fig.tight_layout()

fig.savefig("figures/test_residuals.png", dpi=160, bbox_inches="tight")

plt.close(fig)