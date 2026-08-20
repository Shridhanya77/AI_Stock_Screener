import os
import pandas as pd
import numpy as np


INPUT_FILE = "data/live_ticks.csv"
OUTPUT_FILE = "data/ltq_features.csv"


def load_ticks():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. "
            "Start the FYERS WebSocket first."
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("live_ticks.csv is empty.")

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # Convert numerical columns
    df["ltp"] = pd.to_numeric(
        df["ltp"],
        errors="coerce"
    )

    df["ltq"] = pd.to_numeric(
        df["ltq"],
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "timestamp",
            "ltp",
            "ltq"
        ]
    )

    # Sort
    df = df.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)

    return df


def calculate_ltq_features(df):

    df = df.copy()

    result = []

    # Calculate features separately for each stock
    for symbol, group in df.groupby("symbol", sort=False):

        group = group.copy()

        group = group.sort_values(
            "timestamp"
        )

        group = group.set_index(
            "timestamp"
        )

        # ==================================================
        # LTQ FEATURES
        # ==================================================

        # LTQ average - previous 2 minutes
        group["ltq_avg_2m"] = (
            group["ltq"]
            .rolling(
                "2min",
                min_periods=1
            )
            .mean()
        )

        # LTQ average - previous 5 minutes
        group["ltq_avg_5m"] = (
            group["ltq"]
            .rolling(
                "5min",
                min_periods=1
            )
            .mean()
        )

        # LTQ ratio
        group["ltq_ratio"] = (
            group["ltq_avg_2m"]
            /
            group["ltq_avg_5m"].replace(
                0,
                np.nan
            )
        )

        # LTQ percentage change
        group["ltq_change_pct"] = (
            group["ltq"]
            .pct_change()
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            * 100
        )

        # LTQ standard deviation
        group["ltq_std_5m"] = (
            group["ltq"]
            .rolling(
                "5min",
                min_periods=2
            )
            .std()
        )

        # LTQ Z-score
        group["ltq_zscore"] = (
            (
                group["ltq"]
                -
                group["ltq_avg_5m"]
            )
            /
            group["ltq_std_5m"].replace(
                0,
                np.nan
            )
        )

        # ==================================================
        # PRICE FEATURES
        # ==================================================

        # LTP percentage change
        group["ltp_change_pct"] = (
            group["ltp"]
            .pct_change()
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            * 100
        )

        # LTP change over approximately 1 minute
        group["ltp_change_1m"] = (
            group["ltp"]
            .pct_change(
                freq="1min"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            * 100
        )

        # LTP volatility
        group["ltp_std_5m"] = (
            group["ltp"]
            .rolling(
                "5min",
                min_periods=2
            )
            .std()
        )

        # ==================================================
        # SIGNAL FEATURES
        # ==================================================

        # LTQ spike
        group["ltq_spike"] = (
            group["ltq_ratio"] >= 1.5
        ).astype(int)

        # Price direction
        group["price_direction"] = np.where(
            group["ltp_change_pct"] > 0,
            "UP",
            np.where(
                group["ltp_change_pct"] < 0,
                "DOWN",
                "FLAT"
            )
        )

        group = group.reset_index()

        result.append(group)

    # Combine all stocks
    if not result:
        raise ValueError(
            "No valid symbol data found."
        )

    df = pd.concat(
        result,
        ignore_index=True
    )

    # Final sorting
    df = df.sort_values(
        ["timestamp", "symbol"]
    ).reset_index(drop=True)

    return df


def save_features(df):

    os.makedirs(
        "data",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return OUTPUT_FILE


def generate_ltq_features():

    print("=" * 60)
    print("LTQ FEATURE ENGINEERING")
    print("=" * 60)

    # Load raw tick data
    df = load_ticks()

    print(
        f"Ticks loaded: {len(df)}"
    )

    # Generate features
    df = calculate_ltq_features(
        df
    )

    # Save features
    output = save_features(
        df
    )

    print(
        f"Features saved to: {output}"
    )

    print(
        f"Feature rows: {len(df)}"
    )

    print(
        f"Symbols: {df['symbol'].nunique()}"
    )

    return df


if __name__ == "__main__":

    generate_ltq_features()