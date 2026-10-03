"""Train a California house-price regressor and save it as model.pkl.

Usage:  python train.py
The dataset ships with scikit-learn (fetch_california_housing downloads it
once and caches it). main.py loads model.pkl at startup. scikit-learn is
pinned in requirements.txt so the version that trains the pickle is the
same one that serves it.
"""
import joblib
import numpy as np
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X, y = data.data, data.target  # target is median house value in $100,000s
print(f"Rows: {X.shape[0]}  Features: {list(data.feature_names)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# HistGradientBoosting: strong on tabular data, trains in seconds, and the
# pickle stays small (a RandomForest on 20k rows would be tens of MB).
model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.08, max_leaf_nodes=31, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
mae_usd = mean_absolute_error(y_test, pred) * 100_000
print(f"scikit-learn {sklearn.__version__}")
print(f"Test R^2: {r2:.4f}")
print(f"Test MAE: ${mae_usd:,.0f}")

# Bundle feature names + metrics with the model so the API never hardcodes them.
artifact = {
    "model": model,
    "feature_names": list(data.feature_names),
    "sklearn_version": sklearn.__version__,
    "metrics": {"r2": round(float(r2), 4), "mae_usd": round(float(mae_usd))},
}
joblib.dump(artifact, "model.pkl")
print("Saved model.pkl")
