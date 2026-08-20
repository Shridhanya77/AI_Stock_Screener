import os
from fyers.websocket_data import start_websocket


TOKEN_FILE = "access_token.txt"


def load_access_token():

    if not os.path.exists(TOKEN_FILE):
        raise FileNotFoundError(
            "access_token.txt not found. "
            "Please authenticate with FYERS first."
        )

    with open(TOKEN_FILE, "r", encoding="utf-8") as file:
        access_token = file.read().strip()

    if not access_token:
        raise ValueError(
            "access_token.txt is empty."
        )

    return access_token


if __name__ == "__main__":

    print("=" * 60)
    print("FYERS WEBSOCKET TEST")
    print("=" * 60)

    access_token = load_access_token()

    print("Access token loaded.")
    print("Starting WebSocket...")

    start_websocket(access_token)