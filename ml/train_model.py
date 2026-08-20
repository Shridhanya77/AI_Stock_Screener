import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import joblib


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/final_ml_dataset.csv"

MODEL_DIR = "ml"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model.pkl"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "signal_encoded",

    # Price / trend
    "entry_ltp",
    "SMMA20",
    "SMMA120",
    "SMMA_gap",
    "SMMA_gap_pct",

    # Momentum
    "price_change_5m",
    "price_change_15m",
    "price_change_30m",

    # Volume
    "volume",
    "volume_sma_5",
    "volume_sma_20",
    "volume_ratio",

    # Volatility
    "rolling_std_5",
    "rolling_std_20",
    "ATR_14",

    # SMMA momentum
    "SMMA_gap_change",
    "SMMA_gap_change_pct"
]

TARGET = "target"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("AI STOCK SCREENER - ADVANCED ML TRAINING")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

print(
    "\nDataset shape:",
    df.shape
)

print(
    "\nDataset columns:"
)

print(
    df.columns.tolist()
)


# ============================================================
# TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    unit="s",
    errors="coerce"
)

df = df.dropna(
    subset=["timestamp"]
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


# ============================================================
# ENCODE SIGNAL
# ============================================================

df["signal_encoded"] = df["signal"].map({
    "BUY": 1,
    "SELL": 0
})


# ============================================================
# CHECK FEATURES
# ============================================================

missing = [
    column
    for column in FEATURES + [TARGET]
    if column not in df.columns
]

if missing:

    print(
        "\nERROR: Missing columns:"
    )

    print(
        missing
    )

    raise SystemExit(1)


# ============================================================
# PREPARE X / Y
# ============================================================

X = df[FEATURES].copy()

y = df[TARGET].copy()


# ============================================================
# CLEAN FEATURES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


for column in X.columns:

    if X[column].isna().all():

        X[column] = 0

    else:

        X[column] = X[column].fillna(
            X[column].median()
        )


# ============================================================
# INFORMATION
# ============================================================

print(
    "\nFeatures used:"
)

for feature in FEATURES:

    print(
        "  -",
        feature
    )


print(
    "\nTarget distribution:"
)

print(
    y.value_counts()
)


print(
    "\nDataset date range:"
)

print(
    "Start:",
    df["timestamp"].min()
)

print(
    "End:",
    df["timestamp"].max()
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(
    len(df) * 0.80
)

X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


print()
print("=" * 70)
print("CHRONOLOGICAL TRAIN / TEST SPLIT")
print("=" * 70)


print(
    "\nTraining rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)


print(
    "\nTraining period:"
)

print(
    df["timestamp"].iloc[0],
    "→",
    df["timestamp"].iloc[
        split_index - 1
    ]
)


print(
    "\nTesting period:"
)

print(
    df["timestamp"].iloc[
        split_index
    ],
    "→",
    df["timestamp"].iloc[-1]
)


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)


model = RandomForestClassifier(

    n_estimators=300,

    max_depth=6,

    min_samples_leaf=3,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print()
print("=" * 70)
print("MODEL RESULTS")
print("=" * 70)


print(
    f"\nAccuracy: {accuracy:.4f}"
)


print(
    f"ROC-AUC: {roc_auc:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)


importance = pd.DataFrame({

    "feature": FEATURES,

    "importance":
        model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


model_data = {

    "model": model,

    "features": FEATURES
}


joblib.dump(
    model_data,
    MODEL_PATH
)


print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)


print(
    "\nSaved to:",
    MODEL_PATH
)

print(
    "\nTraining complete."
)