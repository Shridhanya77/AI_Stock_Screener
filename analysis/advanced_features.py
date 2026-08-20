import os
import glob
import numpy as np
import pandas as pd

from analysis.smma import calculate_smma


INPUT_DIR = "data/historical"
OUTPUT_PATH = "data/advanced_ml_dataset.csv"


# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # SMMA
    # --------------------------------------------------------

    df["SMMA20"] = calculate_smma(
        df["close"],
        20
    )

    df["SMMA120"] = calculate_smma(
        df["close"],
        120
    )

    df["SMMA_gap"] = (
        df["SMMA20"] -
        df["SMMA120"]
    )

    df["SMMA_gap_pct"] = (
        df["SMMA_gap"] /
        df["SMMA120"]
    ) * 100


    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    df["price_change_5m"] = (
        df["close"]
        .pct_change(1)
        * 100
    )

    df["price_change_15m"] = (
        df["close"]
        .pct_change(3)
        * 100
    )

    df["price_change_30m"] = (
        df["close"]
        .pct_change(6)
        * 100
    )


    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    df["volume_sma_5"] = (
        df["volume"]
        .rolling(5)
        .mean()
    )

    df["volume_sma_20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_sma_20"]
    )


    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    df["rolling_std_5"] = (
        df["close"]
        .pct_change()
        .rolling(5)
        .std()
        * 100
    )

    df["rolling_std_20"] = (
        df["close"]
        .pct_change()
        .rolling(20)
        .std()
        * 100
    )


    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = (
        df["close"].shift(1)
    )

    tr1 = (
        df["high"] -
        df["low"]
    )

    tr2 = (
        df["high"] -
        previous_close
    ).abs()

    tr3 = (
        df["low"] -
        previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR_14"] = (
        true_range
        .rolling(14)
        .mean()
    )


    # --------------------------------------------------------
    # SMMA GAP MOMENTUM
    # --------------------------------------------------------

    df["SMMA_gap_change"] = (
        df["SMMA_gap"]
        .diff()
    )

    df["SMMA_gap_change_pct"] = (
        df["SMMA_gap"]
        .pct_change()
        * 100
    )


    return df


# ============================================================
# PROCESS ONE SYMBOL
# ============================================================

def process_symbol(path):

    symbol = os.path.basename(path)

    print()
    print("=" * 60)
    print("PROCESSING:", symbol)
    print("=" * 60)

    df = pd.read_csv(path)

    print(
        "Raw candles:",
        len(df)
    )

    # --------------------------------------------------------
    # Calculate features
    # --------------------------------------------------------

    df = calculate_features(df)

    # --------------------------------------------------------
    # Detect crossover
    # --------------------------------------------------------

    df["PREV_GAP"] = (
        df["SMMA_gap"]
        .shift(1)
    )

    df["BUY_SIGNAL"] = (
        (df["PREV_GAP"] <= 0) &
        (df["SMMA_gap"] > 0)
    )

    df["SELL_SIGNAL"] = (
        (df["PREV_GAP"] >= 0) &
        (df["SMMA_gap"] < 0)
    )

    crossover = df[
        df["BUY_SIGNAL"] |
        df["SELL_SIGNAL"]
    ].copy()

    crossover["signal"] = np.where(
        crossover["BUY_SIGNAL"],
        "BUY",
        "SELL"
    )

    print(
        "Crossovers:",
        len(crossover)
    )

    return crossover


# ============================================================
# BUILD DATASET
# ============================================================

def main():

    print("=" * 70)
    print("ADVANCED HISTORICAL FEATURE DATASET")
    print("=" * 70)

    files = glob.glob(
        os.path.join(
            INPUT_DIR,
            "*.csv"
        )
    )

    print(
        "\nHistorical files:",
        len(files)
    )

    all_data = []

    for path in files:

        symbol = os.path.basename(
            path
        ).replace(
            ".csv",
            ""
        )

        result = process_symbol(
            path
        )

        result["symbol"] = symbol

        all_data.append(
            result
        )


    if not all_data:

        print(
            "No historical data found."
        )

        return


    dataset = pd.concat(
        all_data,
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
    # Remove invalid rows
    # --------------------------------------------------------

    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan
    )

    dataset = dataset.dropna()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    dataset.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print()
    print("=" * 70)
    print("DATASET COMPLETE")
    print("=" * 70)

    print(
        "Rows:",
        len(dataset)
    )

    print(
        "Columns:",
        len(dataset.columns)
    )

    print(
        "\nSaved to:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()