import os
import pandas as pd

from fyers.market_data import get_history
from analysis.smma import detect_crossovers
from analysis.trade_analyzer import evaluate_trades
from fyers.market_data import get_history

# ============================================================
# STOCKS TO ANALYZE
# ============================================================

SYMBOLS = [
    "NSE:SBIN-EQ",
    "NSE:HDFCBANK-EQ",
    "NSE:ICICIBANK-EQ",
    "NSE:AXISBANK-EQ",
    "NSE:INFY-EQ",
    "NSE:RELIANCE-EQ",
    "NSE:TCS-EQ",
    "NSE:ITC-EQ",
]


# ============================================================
# CONSTANTS
# ============================================================

IST = "Asia/Kolkata"
UTC = "UTC"

LTQ_FILE = "data/ltq_features.csv"
OUTPUT_FILE = "data/crossover_dataset.csv"

LTQ_TOLERANCE_MINUTES = 2


# ============================================================
# HELPER: CONVERT UNIX SECONDS TO UTC
# ============================================================

def unix_to_utc(series):
    """
    Convert Unix timestamps in seconds to
    pandas datetime64[ns, UTC].

    This is the canonical timestamp format used
    throughout this file.
    """

    return pd.to_datetime(
        series,
        unit="s",
        errors="coerce",
        utc=True
    ).astype("datetime64[ns, UTC]")


# ============================================================
# HELPER: CONVERT LTQ CSV TIMESTAMP TO UTC
# ============================================================

def ltq_timestamp_to_utc(series):
    """
    LTQ CSV timestamps are stored as local IST timestamps,
    for example:

        2026-08-18 10:47:57

    Convert:

        IST -> UTC

    Final dtype:

        datetime64[ns, UTC]
    """

    dt = pd.to_datetime(
        series,
        errors="coerce"
    )

    # If timestamps are already timezone-aware,
    # convert directly to UTC.
    if getattr(dt.dt, "tz", None) is not None:

        return dt.dt.tz_convert(UTC).astype(
            "datetime64[ns, UTC]"
        )

    # Otherwise timestamps are assumed to be IST.
    return (
        dt.dt.tz_localize(
            IST,
            ambiguous="NaT",
            nonexistent="NaT"
        )
        .dt.tz_convert(UTC)
        .astype("datetime64[ns, UTC]")
    )


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

def get_stock_dataframe(access_token, symbol):

    response = get_history(
    access_token,
    symbol,
    resolution="5",
    date_format="1",
    range_from="2026-08-14",
    range_to="2026-08-20",
    cont_flag="1"
)

    if response.get("s") != "ok":

        print(
            f"Failed to fetch data for {symbol}: "
            f"{response}"
        )

        return None

    candles = response.get("candles", [])

    if not candles:

        print(
            f"No candles found for {symbol}"
        )

        return None

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    # --------------------------------------------------------
    # Ensure numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "timestamp",
            "close"
        ]
    )

    # --------------------------------------------------------
    # Create UTC datetime only for display/debugging
    #
    # IMPORTANT:
    # Keep original `timestamp` as Unix seconds.
    # This is required because evaluate_trades()
    # and feature merging use entry_timestamp.
    # --------------------------------------------------------

    df["datetime"] = unix_to_utc(
        df["timestamp"]
    )

    df = df.dropna(
        subset=["datetime"]
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(
        "Historical range:",
        df["datetime"].min(),
        "->",
        df["datetime"].max()
    )

    return df


# ============================================================
# TECHNICAL FEATURE ENGINEERING
# ============================================================

def calculate_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Price momentum
    # --------------------------------------------------------

    df["return_1"] = (
        df["close"].pct_change(1)
    )

    df["return_3"] = (
        df["close"].pct_change(3)
    )

    df["return_5"] = (
        df["close"].pct_change(5)
    )

    # --------------------------------------------------------
    # Volume features
    # --------------------------------------------------------

    df["volume_ma_5"] = (
        df["volume"]
        .rolling(5)
        .mean()
    )

    df["volume_ma_20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_ma_20"]
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    df["volatility_5"] = (
        df["return_1"]
        .rolling(5)
        .std()
    )

    # --------------------------------------------------------
    # SMMA
    # --------------------------------------------------------

    df = detect_crossovers(df)

    # --------------------------------------------------------
    # SMMA spread
    # --------------------------------------------------------

    df["smma_spread"] = (
        df["SMMA20"] -
        df["SMMA120"]
    )

    df["smma_spread_pct"] = (
        df["smma_spread"] /
        df["SMMA120"]
    ) * 100

    return df


# ============================================================
# LOAD LTQ FEATURES
# ============================================================

def load_ltq_features():

    if not os.path.exists(LTQ_FILE):

        print(
            "\nLTQ feature file not found:"
            f" {LTQ_FILE}"
        )

        return pd.DataFrame()

    try:

        ltq_df = pd.read_csv(
            LTQ_FILE
        )

    except Exception as e:

        print(
            f"\nFailed to load LTQ features: {e}"
        )

        return pd.DataFrame()

    if ltq_df.empty:

        print(
            "\nLTQ feature file is empty."
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    if "timestamp" not in ltq_df.columns:

        print(
            "\nERROR: LTQ feature file does not contain "
            "'timestamp'."
        )

        return pd.DataFrame()

    if "symbol" not in ltq_df.columns:

        print(
            "\nERROR: LTQ feature file does not contain "
            "'symbol'."
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Normalize symbol
    # --------------------------------------------------------

    ltq_df["symbol"] = (
        ltq_df["symbol"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Convert LTQ timestamp
    #
    # CSV:
    #
    # 2026-08-18 10:47:57
    #
    # interpreted as IST.
    #
    # Then:
    #
    # IST -> UTC
    #
    # Final dtype:
    #
    # datetime64[ns, UTC]
    # --------------------------------------------------------

    ltq_df["ltq_datetime"] = (
        ltq_timestamp_to_utc(
            ltq_df["timestamp"]
        )
    )

    # --------------------------------------------------------
    # Remove invalid timestamps
    # --------------------------------------------------------

    ltq_df = ltq_df.dropna(
        subset=["ltq_datetime"]
    )

    # --------------------------------------------------------
    # Keep only configured stocks
    # --------------------------------------------------------

    ltq_df = ltq_df[
        ltq_df["symbol"].isin(SYMBOLS)
    ].copy()

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    ltq_df = ltq_df.sort_values(
        ["symbol", "ltq_datetime"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Remove duplicate exact ticks
    #
    # Your previous file contained many duplicate timestamps.
    # Keep the last record for an exact symbol + timestamp.
    # --------------------------------------------------------

    ltq_df = ltq_df.drop_duplicates(
        subset=[
            "symbol",
            "ltq_datetime"
        ],
        keep="last"
    ).reset_index(drop=True)

    print(
        "\nLTQ features loaded:",
        len(ltq_df)
    )

    print(
        "LTQ symbols:",
        ltq_df["symbol"].unique()
    )

    print(
        "LTQ time range:",
        ltq_df["ltq_datetime"].min(),
        "->",
        ltq_df["ltq_datetime"].max()
    )

    return ltq_df


# ============================================================
# MERGE LTQ FEATURES WITH CROSSOVER TRADES
# ============================================================

def merge_ltq_features(trades, ltq_df):

    if trades.empty:

        return trades

    if ltq_df.empty:

        print(
            "No LTQ features available."
        )

        return trades

    trades = trades.copy()
    ltq_df = ltq_df.copy()

    # ========================================================
    # VALIDATION
    # ========================================================

    required_trade_columns = [
        "entry_timestamp",
        "symbol"
    ]

    for column in required_trade_columns:

        if column not in trades.columns:

            print(
                f"ERROR: trades does not contain "
                f"'{column}'."
            )

            return trades

    if "ltq_datetime" not in ltq_df.columns:

        print(
            "ERROR: LTQ data does not contain "
            "'ltq_datetime'."
        )

        return trades

    if "symbol" not in ltq_df.columns:

        print(
            "ERROR: LTQ data does not contain "
            "'symbol'."
        )

        return trades

    # ========================================================
    # TRADE TIMESTAMP
    # ========================================================

    trades["entry_datetime"] = unix_to_utc(
        trades["entry_timestamp"]
    )

    trades = trades.dropna(
        subset=["entry_datetime"]
    )

    # ========================================================
    # FORCE LTQ TIMESTAMP TO SAME EXACT DTYPE
    #
    # THIS FIXES:
    #
    # datetime64[s, UTC]
    # vs
    # datetime64[us, UTC]
    #
    # Both become:
    #
    # datetime64[ns, UTC]
    # ========================================================

    ltq_df["ltq_datetime"] = pd.to_datetime(
        ltq_df["ltq_datetime"],
        errors="coerce",
        utc=True
    ).astype(
        "datetime64[ns, UTC]"
    )

    ltq_df = ltq_df.dropna(
        subset=["ltq_datetime"]
    )

    # ========================================================
    # NORMALIZE SYMBOL
    # ========================================================

    trades["symbol"] = (
        trades["symbol"]
        .astype(str)
        .str.strip()
    )

    ltq_df["symbol"] = (
        ltq_df["symbol"]
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # KEEP USEFUL LTQ COLUMNS
    # ========================================================

    ltq_columns = [
        "ltq_datetime",
        "symbol",
        "ltp",
        "ltq",
        "ltq_avg_2m",
        "ltq_avg_5m",
        "ltq_ratio",
        "ltq_change_pct",
        "ltq_std_5m",
        "ltq_zscore",
        "ltp_change_pct",
        "ltp_change_1m",
        "ltp_std_5m",
        "ltq_spike",
        "price_direction"
    ]

    available_columns = [
        column
        for column in ltq_columns
        if column in ltq_df.columns
    ]

    ltq_df = ltq_df[
        available_columns
    ].copy()

    # ========================================================
    # SORT
    # ========================================================

    trades = trades.sort_values(
        ["symbol", "entry_datetime"]
    ).reset_index(drop=True)

    ltq_df = ltq_df.sort_values(
        ["symbol", "ltq_datetime"]
    ).reset_index(drop=True)

    # ========================================================
    # DEBUG
    # ========================================================

    print(
        "Trade datetime:",
        trades["entry_datetime"].min(),
        "->",
        trades["entry_datetime"].max()
    )

    print(
        "LTQ datetime:",
        ltq_df["ltq_datetime"].min(),
        "->",
        ltq_df["ltq_datetime"].max()
    )

    print(
        "Trade timestamp dtype:",
        trades["entry_datetime"].dtype
    )

    print(
        "LTQ timestamp dtype:",
        ltq_df["ltq_datetime"].dtype
    )

    # ========================================================
    # CHECK OVERLAP
    # ========================================================

    trade_start = trades["entry_datetime"].min()
    trade_end = trades["entry_datetime"].max()

    ltq_start = ltq_df["ltq_datetime"].min()
    ltq_end = ltq_df["ltq_datetime"].max()

    if (
        trade_end < ltq_start
        or trade_start > ltq_end
    ):

        print(
            "\nWARNING: NO TIME OVERLAP BETWEEN "
            "CROSSOVER TRADES AND LTQ DATA."
        )

        print(
            "Trades:",
            trade_start,
            "->",
            trade_end
        )

        print(
            "LTQ:",
            ltq_start,
            "->",
            ltq_end
        )

        print(
            "LTQ features will therefore be NaN "
            "for these trades."
        )

        # Return trades with empty LTQ columns
        # rather than crashing.
        for column in available_columns:

            if column not in [
                "ltq_datetime",
                "symbol"
            ]:

                trades[column] = pd.NA

        trades["ltq_time_difference_sec"] = pd.NA

        return trades.drop(
            columns=["entry_datetime"],
            errors="ignore"
        )

    # ========================================================
    # IMPORTANT:
    # USE "BACKWARD", NOT "NEAREST"
    #
    # This prevents future LTQ ticks from being used to
    # predict a historical trade.
    #
    # Example:
    #
    # Trade = 10:00:00
    #
    # LTQ:
    # 09:59:58
    # 10:00:02
    #
    # backward selects 09:59:58.
    #
    # nearest could select 10:00:02,
    # causing future-data leakage.
    # ========================================================

    merged = pd.merge_asof(
        trades,
        ltq_df,
        left_on="entry_datetime",
        right_on="ltq_datetime",
        by="symbol",
        direction="backward",
        tolerance=pd.Timedelta(
            minutes=LTQ_TOLERANCE_MINUTES
        )
    )

    # ========================================================
    # CALCULATE LTQ TIME DIFFERENCE
    # ========================================================

    if "ltq_datetime" in merged.columns:

        merged["ltq_time_difference_sec"] = (
            merged["entry_datetime"]
            - merged["ltq_datetime"]
        ).dt.total_seconds()

    # ========================================================
    # REMOVE TEMPORARY DATETIME
    # ========================================================

    merged = merged.drop(
        columns=[
            "entry_datetime"
        ],
        errors="ignore"
    )

    return merged


# ============================================================
# BUILD COMPLETE ML DATASET
# ============================================================

def build_dataset(access_token):

    print("\n" + "=" * 60)
    print("BUILDING ML DATASET")
    print("=" * 60)

    all_trades = []

    # ========================================================
    # LOAD LTQ DATA ONCE
    # ========================================================

    ltq_df = load_ltq_features()

    # ========================================================
    # PROCESS EACH STOCK
    # ========================================================

    for symbol in SYMBOLS:

        print("\n" + "=" * 60)
        print(
            f"Processing: {symbol}"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # HISTORICAL DATA
        # ----------------------------------------------------

        df = get_stock_dataframe(
            access_token,
            symbol
        )

        if df is None:

            continue

        print(
            f"Candles received: {len(df)}"
        )

        # ----------------------------------------------------
        # TECHNICAL FEATURES
        # ----------------------------------------------------

        df = calculate_features(
            df
        )

        # ----------------------------------------------------
        # EVALUATE CROSSOVER TRADES
        # ----------------------------------------------------

        trades = evaluate_trades(
            df
        )

        if trades.empty:

            print(
                f"No completed trades found "
                f"for {symbol}"
            )

            continue

        # ----------------------------------------------------
        # ADD SYMBOL
        # ----------------------------------------------------

        trades["symbol"] = symbol

        # ====================================================
        # IMPORTANT:
        #
        # Keep entry_timestamp as Unix seconds.
        #
        # DO NOT convert this column to datetime here.
        #
        # This fixes the earlier merge problem between:
        #
        # trades["entry_timestamp"]
        #
        # and
        #
        # feature_df["entry_timestamp"]
        # ====================================================

        trades["entry_timestamp"] = pd.to_numeric(
            trades["entry_timestamp"],
            errors="coerce"
        )

        # ----------------------------------------------------
        # FEATURE COLUMNS
        # ----------------------------------------------------

        feature_columns = [
            "timestamp",
            "close",
            "SMMA20",
            "SMMA120",
            "smma_spread",
            "smma_spread_pct",
            "volume",
            "volume_ma_5",
            "volume_ma_20",
            "volume_ratio",
            "return_1",
            "return_3",
            "return_5",
            "volatility_5"
        ]

        available_columns = [
            column
            for column in feature_columns
            if column in df.columns
        ]

        feature_df = df[
            available_columns
        ].copy()

        # ----------------------------------------------------
        # Rename timestamp
        #
        # KEEP IT AS UNIX SECONDS.
        # ----------------------------------------------------

        feature_df = feature_df.rename(
            columns={
                "timestamp": "entry_timestamp",
                "close": "feature_entry_ltp"
            }
        )

        feature_df["entry_timestamp"] = pd.to_numeric(
            feature_df["entry_timestamp"],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Remove invalid timestamps
        # ----------------------------------------------------

        feature_df = feature_df.dropna(
            subset=["entry_timestamp"]
        )

        # ----------------------------------------------------
        # Avoid duplicate feature timestamps
        # ----------------------------------------------------

        feature_df = feature_df.drop_duplicates(
            subset=["entry_timestamp"],
            keep="last"
        )

        # ----------------------------------------------------
        # MERGE TECHNICAL FEATURES
        # ----------------------------------------------------

        trades = trades.merge(
            feature_df,
            on="entry_timestamp",
            how="left"
        )

        # ----------------------------------------------------
        # MERGE LTQ FEATURES
        # ----------------------------------------------------

        trades = merge_ltq_features(
            trades,
            ltq_df
        )

        # ----------------------------------------------------
        # ADD TO DATASET
        # ----------------------------------------------------

        all_trades.append(
            trades
        )

        print(
            f"Completed trades: {len(trades)}"
        )

    # ========================================================
    # CHECK DATA
    # ========================================================

    if not all_trades:

        print(
            "\nNo completed trades were found."
        )

        return pd.DataFrame()

    # ========================================================
    # COMBINE ALL STOCKS
    # ========================================================

    dataset = pd.concat(
        all_trades,
        ignore_index=True
    )

    # ========================================================
    # CLEAN TECHNICAL DATA
    # ========================================================

    required_columns = [
        "SMMA20",
        "SMMA120",
        "volume_ratio",
        "return_1",
        "return_3",
        "volatility_5"
    ]

    existing_required_columns = [
        column
        for column in required_columns
        if column in dataset.columns
    ]

    dataset = dataset.dropna(
        subset=existing_required_columns
    )

    # ========================================================
    # SORT DATASET
    # ========================================================

    dataset = dataset.sort_values(
        by=[
            "symbol",
            "entry_timestamp"
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # LTQ COVERAGE
    # ========================================================

    ltq_columns = [
        "ltq",
        "ltq_avg_2m",
        "ltq_avg_5m",
        "ltq_ratio",
        "ltq_zscore",
        "ltq_spike"
    ]

    print("\n" + "=" * 60)
    print("LTQ DATA COVERAGE")
    print("=" * 60)

    total = len(dataset)

    for column in ltq_columns:

        if column in dataset.columns:

            available = (
                dataset[column]
                .notna()
                .sum()
            )

            percentage = (
                available / total * 100
                if total > 0
                else 0
            )

            print(
                f"{column}: "
                f"{available}/{total} "
                f"({percentage:.1f}%)"
            )

        else:

            print(
                f"{column}: COLUMN NOT FOUND"
            )

    # ========================================================
    # OVERALL DATASET SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Total rows: {len(dataset)}"
    )

    if "symbol" in dataset.columns:

        print("\nRows by symbol:")

        print(
            dataset["symbol"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    if "entry_timestamp" in dataset.columns:

        entry_dt = unix_to_utc(
            dataset["entry_timestamp"]
        )

        print(
            "\nTrade time range:",
            entry_dt.min(),
            "->",
            entry_dt.max()
        )

    return dataset


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(dataset):

    os.makedirs(
        "data",
        exist_ok=True
    )

    file_path = OUTPUT_FILE

    dataset.to_csv(
        file_path,
        index=False
    )

    print(
        f"\nDataset saved to: "
        f"{file_path}"
    )

    print(
        f"Rows saved: {len(dataset)}"
    )

    print(
        f"Columns saved: {len(dataset.columns)}"
    )

    return file_path