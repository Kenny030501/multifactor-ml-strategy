import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("factor_600519.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Target: future 20-day return
df["future_return_20d"] = df["close"].shift(-20) / df["close"] - 1

# Features and target
feature_cols = ["momentum_20d", "reversal_5d", "volatility_20d"]
df_clean = df.dropna(subset=feature_cols + ["future_return_20d"]).copy()

X = df_clean[feature_cols]
y = df_clean["future_return_20d"]

# Time-aware split: first 80% for training, last 20% for testing
# (NOT random split, to avoid using the future to predict the past)
split_idx = int(len(df_clean) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train a Random Forest
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Train R2: {train_r2:.4f}")
print(f"Test R2:  {test_r2:.4f}")

# Feature importance
print("\nFeature importance:")
for name, imp in zip(feature_cols, model.feature_importances_):
    print(f"  {name}: {imp:.4f}")