from fyers_apiv3 import fyersModel
import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("FYERS_CLIENT_ID")


def get_fyers_client(access_token):
    if not CLIENT_ID:
        raise ValueError("FYERS_CLIENT_ID is missing from .env")

    if not access_token:
        raise ValueError("FYERS access token is missing")

    return fyersModel.FyersModel(
        client_id=CLIENT_ID,
        token=access_token,
        is_async=False,
        log_path=""
    )


def get_history(
    access_token,
    symbol,
    resolution="5",
    range_from="2026-08-14",
    range_to="2026-08-18"
):

    fyers = get_fyers_client(access_token)

    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": range_from,
        "range_to": range_to,
        "cont_flag": "1"
    }

    print(
        f"Fetching history: {symbol} "
        f"{range_from} -> {range_to}"
    )

    response = fyers.history(data=data)

    return response