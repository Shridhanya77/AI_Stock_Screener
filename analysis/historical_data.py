import os
import time
import pandas as pd
from datetime import datetime, timedelta

from fyers.market_data import get_history


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOLS = [
    "NSE:SBIN-EQ",
    "NSE:RELIANCE-EQ",
    "NSE:TCS-EQ",
    "NSE:INFY-EQ",
    "NSE:HDFCBANK-EQ",
    "NSE:ICICIBANK-EQ",
    "NSE:AXISBANK-EQ",
    "NSE:KOTAKBANK-EQ",
    "NSE:LT-EQ",
]

RANGE_FROM = "2026-01-01"
RANGE_TO = "2026-08-18"

RESOLUTION = "5"

OUTPUT_DIR = os.path.join(
    "data",
    "historical"
)

# FYERS maximum allowed range is 100 days.
# We use 90 days to stay safely below the limit.
CHUNK_DAYS = 90


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DOWNLOAD ONE STOCK
# ============================================================

def download_symbol(
    access_token,
    symbol,
    range_from=RANGE_FROM,
    range_to=RANGE_TO
):

    print()
    print("=" * 60)
    print(f"DOWNLOADING: {symbol}")
    print("=" * 60)

    start_date = datetime.strptime(
        range_from,
        "%Y-%m-%d"
    )

    end_date = datetime.strptime(
        range_to,
        "%Y-%m-%d"
    )

    all_candles = []

    current_start = start_date

    chunk_number = 1

    while current_start <= end_date:

        current_end = min(
            current_start + timedelta(
                days=CHUNK_DAYS
            ),
            end_date
        )

        chunk_from = current_start.strftime(
            "%Y-%m-%d"
        )

        chunk_to = current_end.strftime(
            "%Y-%m-%d"
        )

        print()
        print(
            f"Chunk {chunk_number}: "
            f"{chunk_from} -> {chunk_to}"
        )

        try:

            response = get_history(
                access_token,
                symbol,
                resolution=RESOLUTION,
                range_from=chunk_from,
                range_to=chunk_to
            )

            print(
                "API STATUS:",
                response.get("s")
            )

            if response.get("s") != "ok":

                print("ERROR:")
                print(response)

            else:

                candles = response.get(
                    "candles",
                    []
                )

                print(
                    "Candles received:",
                    len(candles)
                )

                all_candles.extend(
                    candles
                )

        except Exception as e:

            print(
                "EXCEPTION:",
                str(e)
            )

        # Move to next chunk.
        # +1 day prevents overlapping requests.
        current_start = (
            current_end +
            timedelta(days=1)
        )

        chunk_number += 1

        # Small delay between API requests
        time.sleep(0.5)

    # ========================================================
    # CHECK DATA
    # ========================================================

    if not all_candles:

        print()
        print(
            f"NO DATA RECEIVED FOR {symbol}"
        )

        return False

    # ========================================================
    # CONVERT TO DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        all_candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    # ========================================================
    # SORT BY TIMESTAMP
    # ========================================================

    df = df.sort_values(
        by="timestamp"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SAVE FILE
    # ========================================================

    filename = (
        symbol
        .replace(":", "_")
        .replace("/", "_")
        + ".csv"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    df.to_csv(
        output_path,
        index=False
    )

    print()
    print("-" * 60)
    print("DOWNLOAD SUCCESSFUL")
    print("Symbol:", symbol)
    print("Total candles:", len(df))
    print("Saved to:", output_path)
    print("-" * 60)

    return True


# ============================================================
# DOWNLOAD ALL SYMBOLS
# ============================================================

def download_all(
    access_token
):

    print()
    print("=" * 70)
    print("FYERS HISTORICAL DATA DOWNLOADER")
    print("=" * 70)

    print(
        f"Date range: {RANGE_FROM} -> {RANGE_TO}"
    )

    print(
        f"Resolution: {RESOLUTION} minutes"
    )

    print(
        f"Chunk size: {CHUNK_DAYS} days"
    )

    print("=" * 70)

    successful = 0
    failed = 0

    for symbol in SYMBOLS:

        try:

            result = download_symbol(
                access_token,
                symbol
            )

            if result:
                successful += 1
            else:
                failed += 1

        except Exception as e:

            print()
            print(
                f"FAILED: {symbol}"
            )

            print(
                "ERROR:",
                str(e)
            )

            failed += 1

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(
        "Successful:",
        successful
    )

    print(
        "Failed:",
        failed
    )

    print()
    print(
        "Historical data directory:"
    )

    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )

    print("=" * 70)

    return successful, failed