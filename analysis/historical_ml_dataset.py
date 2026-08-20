import os
import glob
import pandas as pd
import numpy as np

from analysis.smma import detect_crossovers


# ============================================================
# CONFIGURATION
# ============================================================

HISTORICAL_DIR = "data/historical"
OUTPUT_FILE = "data/historical_ml_dataset.csv"

# Number of 5-minute candles to hold the trade
# 12 candles = approximately 1 hour
HOLD_CANDLES = 12


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_stock(filepath):

    symbol = os.path.basename(filepath).replace(".csv", "")

    print()
    print("=" * 60)
    print("PROCESSING:", symbol)
    print("=" * 60)

    df = pd.read_csv(filepath)

    print("Raw candles:", len(df))

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print("Missing columns:", missing)
        return pd.DataFrame()

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    df = df.copy()

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    df["open"] = pd.to_numeric(
        df["open"],
        errors="coerce"
    )

    df["high"] = pd.to_numeric(
        df["high"],
        errors="coerce"
    )

    df["low"] = pd.to_numeric(
        df["low"],
        errors="coerce"
    )

    df["volume"] = pd.to_numeric(
        df["volume"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "close"
        ]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Calculate SMMA
    # --------------------------------------------------------

    df = detect_crossovers(df)

    # --------------------------------------------------------
    # Find crossover events
    # --------------------------------------------------------

    events = df[
        df["BUY_SIGNAL"] |
        df["SELL_SIGNAL"]
    ].copy()

    print(
        "Crossover events:",
        len(events)
    )

    if events.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Build trade dataset
    # --------------------------------------------------------

    trades = []

    for index in events.index:

        signal = (
            "BUY"
            if df.loc[index, "BUY_SIGNAL"]
            else "SELL"
        )

        entry_price = df.loc[
            index,
            "close"
        ]

        # Future candle index
        future_index = index + HOLD_CANDLES

        if future_index >= len(df):
            continue

        exit_price = df.loc[
            future_index,
            "close"
        ]

        # ----------------------------------------------------
        # Calculate P&L
        # ----------------------------------------------------

        if signal == "BUY":

            pnl = exit_price - entry_price

        else:

            pnl = entry_price - exit_price

        # Target
        target = 1 if pnl > 0 else 0

        # ----------------------------------------------------
        # SMMA features
        # ----------------------------------------------------

        smma20 = df.loc[
            index,
            "SMMA20"
        ]

        smma120 = df.loc[
            index,
            "SMMA120"
        ]

        smma_gap = (
            smma20 - smma120
        )

        # ----------------------------------------------------
        # Additional price features
        # ----------------------------------------------------

        previous_close = (
            df.loc[index - 1, "close"]
            if index > 0
            else entry_price
        )

        price_change_pct = (
            (entry_price - previous_close)
            / previous_close
            * 100
        )

        # ----------------------------------------------------
        # Save trade
        # ----------------------------------------------------

        trades.append({

            "symbol": symbol,

            "timestamp": df.loc[
                index,
                "timestamp"
            ],

            "signal": signal,

            "entry_ltp": entry_price,

            "exit_ltp": exit_price,

            "pnl": pnl,

            "target": target,

            "SMMA20": smma20,

            "SMMA120": smma120,

            "SMMA_gap": smma_gap,

            "price_change_pct": price_change_pct,

            "volume": df.loc[
                index,
                "volume"
            ]
        })

    return pd.DataFrame(trades)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("HISTORICAL ML DATASET BUILDER")
    print("=" * 70)

    files = glob.glob(
        os.path.join(
            HISTORICAL_DIR,
            "*.csv"
        )
    )

    print()
    print("Historical files:", len(files))

    if not files:

        print(
            "ERROR: No historical CSV files found."
        )

        return

    all_trades = []

    # --------------------------------------------------------
    # Process each stock
    # --------------------------------------------------------

    for filepath in files:

        try:

            trades = process_stock(
                filepath
            )

            if not trades.empty:

                all_trades.append(
                    trades
                )

        except Exception as e:

            print(
                "ERROR processing:",
                filepath
            )

            print(
                str(e)
            )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    if not all_trades:

        print(
            "\nNo trades generated."
        )

        return

    dataset = pd.concat(
        all_trades,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    dataset = dataset.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET COMPLETE")
    print("=" * 70)

    print(
        "\nTotal training rows:",
        len(dataset)
    )

    print(
        "\nTarget distribution:"
    )

    print(
        dataset["target"].value_counts()
    )

    print(
        "\nSignal distribution:"
    )

    print(
        dataset["signal"].value_counts()
    )

    print(
        "\nTotal P&L:",
        round(
            dataset["pnl"].sum(),
            2
        )
    )

    print(
        "\nAverage P&L:",
        round(
            dataset["pnl"].mean(),
            2
        )
    )

    print(
        "\nWin rate:",
        round(
            dataset["target"].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "\nSaved to:",
        OUTPUT_FILE
    )

    print("=" * 70)


if __name__ == "__main__":

    main()