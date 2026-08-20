import pandas as pd
import numpy as np


def calculate_smma(series, period):
    """
    Calculate Smoothed Moving Average (SMMA).

    SMMA:
        First value = SMA
        Following values:
        SMMA = ((Previous SMMA * (period - 1)) + Current Price) / period
    """

    series = pd.Series(series, dtype="float64")

    smma = pd.Series(
        np.nan,
        index=series.index,
        dtype="float64"
    )

    if len(series) < period:
        return smma

    # Initial SMA
    smma.iloc[period - 1] = series.iloc[:period].mean()

    # Subsequent SMMA values
    for i in range(period, len(series)):

        smma.iloc[i] = (
            (smma.iloc[i - 1] * (period - 1))
            + series.iloc[i]
        ) / period

    return smma


def detect_crossovers(df):
    """
    Detect SMMA 20 / SMMA 120 crossovers.

    BUY:
        SMMA20 crosses above SMMA120

    SELL:
        SMMA20 crosses below SMMA120
    """

    df = df.copy()

    # Calculate SMMAs
    df["SMMA20"] = calculate_smma(
        df["close"],
        20
    )

    df["SMMA120"] = calculate_smma(
        df["close"],
        120
    )

    # Difference
    df["SMMA_DIFF"] = (
        df["SMMA20"] -
        df["SMMA120"]
    )

    # Previous difference
    df["PREV_DIFF"] = df["SMMA_DIFF"].shift(1)

    # BUY crossover
    df["BUY_SIGNAL"] = (
        (df["PREV_DIFF"] <= 0) &
        (df["SMMA_DIFF"] > 0)
    )

    # SELL crossover
    df["SELL_SIGNAL"] = (
        (df["PREV_DIFF"] >= 0) &
        (df["SMMA_DIFF"] < 0)
    )

    return df


def get_crossover_events(df):
    """
    Return only rows where a crossover occurred.
    """

    crossover_mask = (
        df["BUY_SIGNAL"] |
        df["SELL_SIGNAL"]
    )

    events = df[crossover_mask].copy()

    events["SIGNAL"] = np.where(
        events["BUY_SIGNAL"],
        "BUY",
        "SELL"
    )

    return events