from fyers_apiv3.FyersWebsocket import data_ws
import os
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo
import csv


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
# Store latest tick for each symbol
latest_ticks = {}

# CSV file
CSV_FILE = "data/live_ticks.csv"

# Global WebSocket object
fyers = None


# ============================================================
# SAVE TICK
# ============================================================

def save_tick(symbol, ltp, ltq, timestamp):

    os.makedirs("data", exist_ok=True)

    file_exists = os.path.exists(CSV_FILE)

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([
                "timestamp",
                "symbol",
                "ltp",
                "ltq"
            ])

        writer.writerow([
            timestamp,
            symbol,
            ltp,
            ltq
        ])

        file.flush()


# ============================================================
# FYERS WEBSOCKET MESSAGE
# ============================================================

def onmessage(message):

    print("\nLIVE TICK:")
    print(message)

    # Make sure message is a dictionary
    if not isinstance(message, dict):
        return

    # Get symbol
    symbol = message.get("symbol")

    if not symbol:
        return

    # Get LTP
    ltp = message.get("ltp")

    # Get LTQ
    ltq = message.get("last_traded_qty")

    # FYERS exchange feed timestamp
    exchange_time = message.get("exch_feed_time")

    # Ignore messages without market data
    if ltp is None and ltq is None:
        return

    # ========================================================
    # TIMESTAMP
    # ========================================================

    if exchange_time:

        try:

            timestamp = datetime.fromtimestamp(
                exchange_time,
                tz=ZoneInfo("Asia/Kolkata")
            ).isoformat()

        except Exception:

            timestamp = datetime.now(
                tz=ZoneInfo("Asia/Kolkata")
            ).isoformat()

    else:

        timestamp = datetime.now(
            tz=ZoneInfo("Asia/Kolkata")
        ).isoformat()

    # ========================================================
    # STORE LATEST TICK
    # ========================================================

    latest_ticks[symbol] = {

        "timestamp": timestamp,

        "exchange_time": exchange_time,

        "ltp": ltp,

        "ltq": ltq
    }

    # ========================================================
    # SAVE EVERY MARKET TICK
    # ========================================================

    save_tick(
        symbol,
        ltp,
        ltq,
        timestamp
    )

    print(
        f"NEW MARKET TICK | "
        f"{symbol} | "
        f"LTP={ltp} | "
        f"LTQ={ltq} | "
        f"EXCH_TIME={exchange_time}"
    )


# ============================================================
# WEBSOCKET ERROR
# ============================================================

def onerror(message):

    print("\nWEBSOCKET ERROR:")
    print(message)


# ============================================================
# WEBSOCKET CLOSE
# ============================================================

def onclose(message):

    print("\nWEBSOCKET CLOSED:")
    print(message)


# ============================================================
# WEBSOCKET CONNECT
# ============================================================

def onopen():

    print("\n================================")
    print("FYERS WEBSOCKET CONNECTED")
    print("================================")

    symbols = [
        "NSE:SBIN-EQ",
        "NSE:RELIANCE-EQ",
        "NSE:TCS-EQ",
        "NSE:INFY-EQ",
        "NSE:HDFCBANK-EQ",
        "NSE:ICICIBANK-EQ",
        "NSE:AXISBANK-EQ",
        "NSE:ITC-EQ"
    ]

    fyers.subscribe(
        symbols=symbols,
        data_type="SymbolUpdate"
    )

    print("\nSubscribed to:")

    for symbol in symbols:
        print("  ", symbol)

    print(
        "\nTotal symbols subscribed:",
        len(symbols)
    )

    print("\n================================")


# ============================================================
# START WEBSOCKET
# ============================================================

def start_websocket(access_token):

    global fyers

    # Check Client ID
    if not CLIENT_ID:

        raise ValueError(
            "FYERS_CLIENT_ID is missing from .env"
        )

    # Check access token
    if not access_token:

        raise ValueError(
            "FYERS access token is missing."
        )

    # FYERS WebSocket token format
    websocket_token = (
        CLIENT_ID
        + ":"
        + access_token
    )

    # Create WebSocket
    fyers = data_ws.FyersDataSocket(

        access_token=websocket_token,

        log_path="",

        litemode=False,

        write_to_file=False,

        reconnect=True,

        on_connect=onopen,

        on_close=onclose,

        on_error=onerror,

        on_message=onmessage
    )

    print("\nStarting FYERS WebSocket...")

    # Connect
    fyers.connect()