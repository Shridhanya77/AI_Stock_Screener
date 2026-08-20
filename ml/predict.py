import os
import joblib
import pandas as pd


MODEL_PATH = "ml/model.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Run train_model.py first."
    )

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
FEATURES = model_data["features"]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_trade(data):
    """
    Predict whether a trade setup is likely to be profitable.

    data can be a dictionary containing the required features.
    """

    df = pd.DataFrame([data])

    # Make sure all required features exist
    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = 0

    # Keep only training features and correct order
    X = df[FEATURES].copy()

    # Replace infinite values
    X = X.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    # Handle missing values
    for column in X.columns:
        if X[column].isna().all():
            X[column] = 0
        else:
            X[column] = X[column].fillna(0)

    # Prediction
    prediction = model.predict(X)[0]

    # Probability
    probabilities = model.predict_proba(X)[0]

    probability = probabilities[prediction]

    # Convert prediction
    if prediction == 1:
        result = "PROFITABLE"
    else:
        result = "NOT_PROFITABLE"

    return {
        "prediction": int(prediction),
        "result": result,
        "confidence": round(float(probability) * 100, 2)
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_trade = {
        "signal_encoded": 1,
        "entry_ltp": 1200,
        "ltq": 5,
        "ltq_avg_2m": 5,
        "ltq_avg_5m": 5,
        "ltq_ratio": 1.0,
        "ltq_change_pct": 0,
        "ltq_zscore": 0,
        "ltp_change_pct": 0,
        "ltp_std_5m": 0
    }

    result = predict_trade(sample_trade)

    print("=" * 60)
    print("ML TRADE PREDICTION")
    print("=" * 60)

    print(f"\nPrediction: {result['result']}")
    print(f"Confidence: {result['confidence']}%")

    print("\nComplete result:")
    print(result)