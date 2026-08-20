import pandas as pd
import numpy as np
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/historical_ml_dataset.csv"
MODEL_PATH = "ml/model.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AI STOCK SCREENER - STRATEGY BACKTEST")
print("=" * 70)

df = pd.read_csv(DATA_PATH)


# ============================================================
# FIX TIMESTAMP
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
# SORT DATA
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


print("\nHistorical trades:", len(df))

print(
    "Period:",
    df["timestamp"].min(),
    "->",
    df["timestamp"].max()
)


# ============================================================
# BASELINE STRATEGY
# ============================================================

print("\n" + "=" * 70)
print("STRATEGY 1 - SMMA ONLY")
print("=" * 70)


smma_trades = df.copy()

smma_trades["strategy_pnl"] = (
    smma_trades["pnl"]
)


total_trades = len(
    smma_trades
)

winning_trades = (
    smma_trades["strategy_pnl"] > 0
).sum()

losing_trades = (
    smma_trades["strategy_pnl"] <= 0
).sum()


win_rate = (
    winning_trades /
    total_trades *
    100
)


total_pnl = (
    smma_trades["strategy_pnl"]
    .sum()
)


average_pnl = (
    smma_trades["strategy_pnl"]
    .mean()
)


# ============================================================
# PROFIT FACTOR
# ============================================================

gross_profit = smma_trades.loc[
    smma_trades["strategy_pnl"] > 0,
    "strategy_pnl"
].sum()


gross_loss = abs(
    smma_trades.loc[
        smma_trades["strategy_pnl"] < 0,
        "strategy_pnl"
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

equity = (
    smma_trades["strategy_pnl"]
    .cumsum()
)

peak = (
    equity
    .cummax()
)

drawdown = (
    equity -
    peak
)

max_drawdown = (
    drawdown.min()
)


# ============================================================
# PRINT BASELINE
# ============================================================

print(
    f"\nTrades           : {total_trades}"
)

print(
    f"Winning trades   : {winning_trades}"
)

print(
    f"Losing trades    : {losing_trades}"
)

print(
    f"Win rate         : {win_rate:.2f}%"
)

print(
    f"Total P&L        : ₹{total_pnl:.2f}"
)

print(
    f"Average P&L      : ₹{average_pnl:.2f}"
)

print(
    f"Profit factor    : {profit_factor:.2f}"
)

print(
    f"Max drawdown     : ₹{max_drawdown:.2f}"
)


# ============================================================
# LOAD ML MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING ML MODEL")
print("=" * 70)


model_data = joblib.load(
    MODEL_PATH
)

model = model_data["model"]

features = model_data["features"]


print(
    "\nModel features:"
)

for feature in features:

    print(
        "  -",
        feature
    )


# ============================================================
# PREPARE ML FEATURES
# ============================================================

ml_df = df.copy()


ml_df["signal_encoded"] = (
    ml_df["signal"]
    .map({
        "BUY": 1,
        "SELL": 0
    })
)


X = ml_df[features].copy()


# Replace infinity
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# Fill missing values
for column in X.columns:

    if X[column].isna().all():

        X[column] = 0

    else:

        X[column] = X[column].fillna(
            X[column].median()
        )


# ============================================================
# ML PREDICTION
# ============================================================

ml_df["prediction"] = (
    model.predict(X)
)


ml_df["confidence"] = (
    model.predict_proba(X)
    .max(axis=1)
)


# ============================================================
# ML FILTER
# ============================================================

# Take only trades predicted profitable.

filtered_trades = ml_df[
    ml_df["prediction"] == 1
].copy()


print("\n" + "=" * 70)
print("STRATEGY 2 - SMMA + ML")
print("=" * 70)


# ============================================================
# CALCULATE ML RESULTS
# ============================================================

ml_total_trades = len(
    filtered_trades
)


ml_winning_trades = (
    filtered_trades["pnl"] > 0
).sum()


ml_losing_trades = (
    filtered_trades["pnl"] <= 0
).sum()


if ml_total_trades > 0:

    ml_win_rate = (
        ml_winning_trades /
        ml_total_trades *
        100
    )

    ml_total_pnl = (
        filtered_trades["pnl"]
        .sum()
    )

    ml_average_pnl = (
        filtered_trades["pnl"]
        .mean()
    )

else:

    ml_win_rate = 0

    ml_total_pnl = 0

    ml_average_pnl = 0


# ============================================================
# ML PROFIT FACTOR
# ============================================================

ml_gross_profit = (
    filtered_trades.loc[
        filtered_trades["pnl"] > 0,
        "pnl"
    ].sum()
)


ml_gross_loss = abs(
    filtered_trades.loc[
        filtered_trades["pnl"] < 0,
        "pnl"
    ].sum()
)


if ml_gross_loss > 0:

    ml_profit_factor = (
        ml_gross_profit /
        ml_gross_loss
    )

else:

    ml_profit_factor = np.inf


# ============================================================
# ML MAX DRAWDOWN
# ============================================================

if ml_total_trades > 0:

    ml_equity = (
        filtered_trades["pnl"]
        .cumsum()
    )

    ml_peak = (
        ml_equity
        .cummax()
    )

    ml_drawdown = (
        ml_equity -
        ml_peak
    )

    ml_max_drawdown = (
        ml_drawdown.min()
    )

else:

    ml_max_drawdown = 0


# ============================================================
# PRINT ML RESULTS
# ============================================================

print(
    f"\nTrades           : {ml_total_trades}"
)

print(
    f"Winning trades   : {ml_winning_trades}"
)

print(
    f"Losing trades    : {ml_losing_trades}"
)

print(
    f"Win rate         : {ml_win_rate:.2f}%"
)

print(
    f"Total P&L        : ₹{ml_total_pnl:.2f}"
)

print(
    f"Average P&L      : ₹{ml_average_pnl:.2f}"
)

print(
    f"Profit factor    : {ml_profit_factor:.2f}"
)

print(
    f"Max drawdown     : ₹{ml_max_drawdown:.2f}"
)


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("STRATEGY COMPARISON")
print("=" * 70)


print(
    f"\nSMMA only trades     : {total_trades}"
)

print(
    f"SMMA + ML trades     : {ml_total_trades}"
)


print(
    f"\nSMMA only win rate   : {win_rate:.2f}%"
)

print(
    f"SMMA + ML win rate   : {ml_win_rate:.2f}%"
)


print(
    f"\nSMMA only P&L        : ₹{total_pnl:.2f}"
)

print(
    f"SMMA + ML P&L        : ₹{ml_total_pnl:.2f}"
)


print(
    f"\nP&L improvement      : "
    f"₹{ml_total_pnl - total_pnl:.2f}"
)


print(
    f"\nWin-rate improvement: "
    f"{ml_win_rate - win_rate:.2f}%"
)


print("\n" + "=" * 70)
print("BACKTEST COMPLETE")
print("=" * 70)