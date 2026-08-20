from ml.predict import predict_trade


def get_ai_signal(
    signal,
    entry_ltp,
    ltq=0,
    ltq_avg_2m=0,
    ltq_avg_5m=0,
    ltq_ratio=0,
    ltq_change_pct=0,
    ltq_zscore=0,
    ltp_change_pct=0,
    ltp_std_5m=0
):
    """
    Combine SMMA signal with ML prediction.
    """

    # Convert SMMA signal
    if signal == "BUY":
        signal_encoded = 1
    elif signal == "SELL":
        signal_encoded = 0
    else:
        return {
            "smma_signal": signal,
            "ml_result": "NONE",
            "confidence": 0,
            "final_signal": "HOLD"
        }

    features = {
        "signal_encoded": signal_encoded,
        "entry_ltp": entry_ltp,
        "ltq": ltq,
        "ltq_avg_2m": ltq_avg_2m,
        "ltq_avg_5m": ltq_avg_5m,
        "ltq_ratio": ltq_ratio,
        "ltq_change_pct": ltq_change_pct,
        "ltq_zscore": ltq_zscore,
        "ltp_change_pct": ltp_change_pct,
        "ltp_std_5m": ltp_std_5m
    }

    prediction = predict_trade(features)

    confidence = prediction["confidence"]

    # ========================================================
    # FINAL SIGNAL
    # ========================================================

    if prediction["prediction"] == 1 and confidence >= 60:
        final_signal = signal
    else:
        final_signal = "HOLD"

    return {
        "smma_signal": signal,
        "ml_result": prediction["result"],
        "confidence": confidence,
        "final_signal": final_signal
    }