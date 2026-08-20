import os
import pandas as pd
import numpy as np


TRADE_FILE = "data/crossover_dataset.csv"
LTQ_FILE = "data/ltq_features.csv"
OUTPUT_FILE = "data/ml_training_dataset.csv"


def load_data():
    if not os.path.exists(TRADE_FILE):
        raise FileNotFoundError(
            f"{TRADE_FILE} not found."
        )

    if not os.path.exists(LTQ_FILE):
        raise FileNotFoundError(
            f"{LTQ_FILE} not found."
        )

    trades = pd.read_csv(TRADE_FILE)
    ltq = pd.read_csv(LTQ_FILE)

    return trades, ltq


def prepare_trades(trades):

    trades = trades.copy()

    # --------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------

    if "entry_timestamp" in trades.columns:
        trades["entry_timestamp"] = pd.to_numeric(
            trades["entry_timestamp"],
            errors="coerce"
        )

    if "exit_timestamp" in trades.columns:
        trades["exit_timestamp"] = pd.to_numeric(
            trades["exit_timestamp"],
            errors="coerce"
        )

    # --------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------

    for column in [
        "entry_ltp",
        "exit_ltp",
        "pnl"
    ]:
        if column in trades.columns:
            trades[column] = pd.to_numeric(
                trades[column],
                errors="coerce"
            )

    # --------------------------------------------------
    # Target
    # --------------------------------------------------

    trades["target"] = (
        trades["pnl"] > 0
    ).astype(int)

    # --------------------------------------------------
    # Create timezone-naive entry datetime
    # from Unix timestamp
    # --------------------------------------------------

    trades["entry_datetime"] = pd.to_datetime(
        trades["entry_timestamp"],
        unit="s",
        errors="coerce",
        utc=True
    ).dt.tz_localize(None)

    return trades


def prepare_ltq(ltq):

    ltq = ltq.copy()

    # --------------------------------------------------
    # Convert LTQ timestamp
    # --------------------------------------------------

    ltq["timestamp"] = pd.to_datetime(
        ltq["timestamp"],
        errors="coerce",
        utc=True
    ).dt.tz_localize(None)

    # --------------------------------------------------
    # Remove invalid timestamps
    # --------------------------------------------------

    ltq = ltq.dropna(
        subset=["timestamp"]
    )

    # --------------------------------------------------
    # Normalize symbol
    # --------------------------------------------------

    if "symbol" in ltq.columns:
        ltq["symbol"] = (
            ltq["symbol"]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------
    # Sort
    # --------------------------------------------------

    ltq = ltq.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)

    return ltq


def create_dataset(trades, ltq):

    rows = []

    # --------------------------------------------------
    # Make sure trades are sorted
    # --------------------------------------------------

    trades = trades.sort_values(
        "entry_datetime"
    )

    for _, trade in trades.iterrows():

        entry_time = trade["entry_datetime"]
        symbol = trade.get("symbol", "")

        if pd.isna(entry_time):
            continue

        # --------------------------------------------------
        # Only use LTQ data for SAME STOCK
        # --------------------------------------------------

        symbol_ltq = ltq[
            ltq["symbol"] == symbol
        ].copy()

        if symbol_ltq.empty:
            print(
                f"No LTQ data found for {symbol}"
            )
            continue

        # --------------------------------------------------
        # Find nearest LTQ record
        # --------------------------------------------------

        differences = abs(
            symbol_ltq["timestamp"] -
            entry_time
        )

        if differences.empty:
            continue

        nearest_index = differences.idxmin()

        market = symbol_ltq.loc[
            nearest_index
        ]

        # --------------------------------------------------
        # Calculate time difference
        # --------------------------------------------------

        time_difference = abs(
            (
                market["timestamp"] -
                entry_time
            ).total_seconds()
        )

        # --------------------------------------------------
        # Don't match very distant LTQ data
        # --------------------------------------------------

        if time_difference > 300:

            print(
                f"Skipping {symbol}: "
                f"nearest LTQ data is "
                f"{time_difference:.1f} seconds away"
            )

            continue

        # --------------------------------------------------
        # Create ML row
        # --------------------------------------------------

        row = {

            # Trade information
            "signal": trade.get(
                "signal",
                ""
            ),

            "symbol": symbol,

            "entry_timestamp": trade.get(
                "entry_timestamp",
                np.nan
            ),

            "entry_ltp": trade.get(
                "entry_ltp",
                np.nan
            ),

            # SMMA features
            "entry_smma20": trade.get(
                "entry_smma20",
                np.nan
            ),

            "entry_smma120": trade.get(
                "entry_smma120",
                np.nan
            ),

            "smma_spread": trade.get(
                "smma_spread",
                np.nan
            ),

            "smma_spread_pct": trade.get(
                "smma_spread_pct",
                np.nan
            ),

            # Volume features
            "volume": trade.get(
                "volume",
                np.nan
            ),

            "volume_ma_5": trade.get(
                "volume_ma_5",
                np.nan
            ),

            "volume_ma_20": trade.get(
                "volume_ma_20",
                np.nan
            ),

            "volume_ratio": trade.get(
                "volume_ratio",
                np.nan
            ),

            # Return / volatility
            "return_1": trade.get(
                "return_1",
                np.nan
            ),

            "return_3": trade.get(
                "return_3",
                np.nan
            ),

            "return_5": trade.get(
                "return_5",
                np.nan
            ),

            "volatility_5": trade.get(
                "volatility_5",
                np.nan
            ),

            # LTQ features
            "ltq": market.get(
                "ltq",
                np.nan
            ),

            "ltq_avg_2m": market.get(
                "ltq_avg_2m",
                np.nan
            ),

            "ltq_avg_5m": market.get(
                "ltq_avg_5m",
                np.nan
            ),

            "ltq_ratio": market.get(
                "ltq_ratio",
                np.nan
            ),

            "ltq_change_pct": market.get(
                "ltq_change_pct",
                np.nan
            ),

            "ltq_std_5m": market.get(
                "ltq_std_5m",
                np.nan
            ),

            "ltq_zscore": market.get(
                "ltq_zscore",
                np.nan
            ),

            "ltq_spike": market.get(
                "ltq_spike",
                np.nan
            ),

            # Price features
            "ltp_change_pct": market.get(
                "ltp_change_pct",
                np.nan
            ),

            "ltp_change_1m": market.get(
                "ltp_change_1m",
                np.nan
            ),

            "ltp_std_5m": market.get(
                "ltp_std_5m",
                np.nan
            ),

            "price_direction": market.get(
                "price_direction",
                ""
            ),

            # Time difference
            "ltq_time_difference_sec": time_difference,

            # Target
            "pnl": trade.get(
                "pnl",
                np.nan
            ),

            "target": trade.get(
                "target",
                np.nan
            )
        }

        rows.append(row)

    return pd.DataFrame(rows)


def save_dataset(dataset):

    os.makedirs(
        "data",
        exist_ok=True
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"ML dataset saved to: {OUTPUT_FILE}"
    )


def build_ml_dataset():

    print("=" * 60)
    print("BUILDING ML TRAINING DATASET")
    print("=" * 60)

    trades, ltq = load_data()

    print(
        f"Trades loaded: {len(trades)}"
    )

    print(
        f"LTQ records loaded: {len(ltq)}"
    )

    trades = prepare_trades(
        trades
    )

    ltq = prepare_ltq(
        ltq
    )

    dataset = create_dataset(
        trades,
        ltq
    )

    if dataset.empty:

        print(
            "WARNING: ML dataset is empty."
        )

        return dataset

    # --------------------------------------------------
    # Remove rows where essential features are missing
    # --------------------------------------------------

    dataset = dataset.dropna(
        subset=[
            "ltq",
            "ltq_avg_2m",
            "ltq_avg_5m",
            "ltq_ratio",
            "target"
        ]
    )

    save_dataset(
        dataset
    )

    print(
        f"Training rows: {len(dataset)}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        dataset["target"]
        .value_counts()
    )

    print(
        "\nDataset preview:"
    )

    print(
        dataset.head()
    )

    return dataset


if __name__ == "__main__":

    build_ml_dataset()