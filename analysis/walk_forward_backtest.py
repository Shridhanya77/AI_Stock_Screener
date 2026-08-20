import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


DATA_PATH = "data/historical_ml_dataset.csv"

FEATURES = [
    "signal_encoded",
    "entry_ltp",
    "SMMA20",
    "SMMA120",
    "SMMA_gap",
    "price_change_pct",
    "volume"
]

TARGET = "target"

INITIAL_TRAIN_SIZE = 300
TEST_WINDOW = 50


print("=" * 70)
print("AI STOCK SCREENER - WALK-FORWARD BACKTEST")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    unit="s",
    errors="coerce"
)

df = df.dropna(
    subset=["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# ENCODE SIGNAL
# ============================================================

df["signal_encoded"] = df["signal"].map({
    "BUY": 1,
    "SELL": 0
})


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].copy()

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
# WALK-FORWARD TEST
# ============================================================

all_predictions = []

start = INITIAL_TRAIN_SIZE

round_number = 1


while start < len(df):

    test_end = min(
        start + TEST_WINDOW,
        len(df)
    )

    print()
    print("-" * 70)
    print(
        f"ROUND {round_number}"
    )

    print(
        "Training:",
        start
        - INITIAL_TRAIN_SIZE,
        "->",
        start - 1
    )

    print(
        "Testing:",
        start,
        "->",
        test_end - 1
    )


    X_train = X.iloc[
        :start
    ]

    y_train = y.iloc[
        :start
    ]

    X_test = X.iloc[
        start:test_end
    ]

    y_test = y.iloc[
        start:test_end
    ]


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PREDICT UNSEEN DATA
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    result = df.iloc[
        start:test_end
    ].copy()

    result["prediction"] = predictions

    result["probability"] = probabilities

    all_predictions.append(
        result
    )


    print(
        "Test accuracy:",
        f"{accuracy_score(y_test, predictions):.2f}"
    )


    start = test_end

    round_number += 1


# ============================================================
# COMBINE RESULTS
# ============================================================

results = pd.concat(
    all_predictions,
    ignore_index=True
)


# ============================================================
# OUT-OF-SAMPLE ML PERFORMANCE
# ============================================================

print()
print("=" * 70)
print("OUT-OF-SAMPLE ML RESULTS")
print("=" * 70)


accuracy = accuracy_score(
    results["target"],
    results["prediction"]
)


print(
    f"\nAccuracy: {accuracy:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        results["target"],
        results["prediction"],
        zero_division=0
    )
)


# ============================================================
# SMMA BASELINE
# ============================================================

smma_pnl = results["pnl"].sum()

smma_trades = len(results)

smma_wins = (
    results["pnl"] > 0
).sum()

smma_win_rate = (
    smma_wins /
    smma_trades *
    100
)


# ============================================================
# ML FILTER
# ============================================================

ml_results = results[
    results["prediction"] == 1
].copy()


ml_trades = len(
    ml_results
)

ml_wins = (
    ml_results["pnl"] > 0
).sum()


if ml_trades > 0:

    ml_win_rate = (
        ml_wins /
        ml_trades *
        100
    )

    ml_pnl = (
        ml_results["pnl"].sum()
    )

    ml_average = (
        ml_results["pnl"].mean()
    )

else:

    ml_win_rate = 0
    ml_pnl = 0
    ml_average = 0


# ============================================================
# PROFIT FACTOR
# ============================================================

gross_profit = ml_results.loc[
    ml_results["pnl"] > 0,
    "pnl"
].sum()


gross_loss = abs(
    ml_results.loc[
        ml_results["pnl"] < 0,
        "pnl"
    ].sum()
)


if gross_loss > 0:

    profit_factor = (
        gross_profit /
        gross_loss
    )

else:

    profit_factor = np.inf


# ============================================================
# MAX DRAWDOWN
# ============================================================

if ml_trades > 0:

    equity = (
        ml_results["pnl"]
        .cumsum()
    )

    peak = (
        equity.cummax()
    )

    drawdown = (
        equity - peak
    )

    max_drawdown = (
        drawdown.min()
    )

else:

    max_drawdown = 0


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("STRATEGY COMPARISON")
print("=" * 70)


print(
    "\nSMMA BASELINE"
)

print(
    "Trades:",
    smma_trades
)

print(
    f"Win rate: {smma_win_rate:.2f}%"
)

print(
    f"P&L: ₹{smma_pnl:.2f}"
)


print(
    "\nSMMA + ML"
)

print(
    "Trades:",
    ml_trades
)

print(
    f"Win rate: {ml_win_rate:.2f}%"
)

print(
    f"Total P&L: ₹{ml_pnl:.2f}"
)

print(
    f"Average P&L: ₹{ml_average:.2f}"
)

print(
    f"Profit factor: {profit_factor:.2f}"
)

print(
    f"Max drawdown: ₹{max_drawdown:.2f}"
)


print()
print("=" * 70)
print("WALK-FORWARD BACKTEST COMPLETE")
print("=" * 70)