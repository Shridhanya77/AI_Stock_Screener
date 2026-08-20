from flask import Flask, redirect, request, session, render_template
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv
from fyers.market_data import get_history
from analysis.historical_data import download_all
import plotly.graph_objects as go

import os
import pandas as pd
import threading

from analysis.smma import (
    detect_crossovers,
    get_crossover_events
)

from analysis.trade_analyzer import evaluate_trades

from analysis.dataset_builder import (
    build_dataset,
    save_dataset
)

from fyers.websocket_data import start_websocket

from analysis.ltq_features import generate_ltq_features


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.secret_key = "ai-stock-screener-local-secret"


# ============================================================
# FYERS CONFIGURATION
# ============================================================

CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
SECRET_KEY = os.getenv("FYERS_SECRET_KEY")
REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI")


# ============================================================
# GLOBAL WEBSOCKET STATUS
# ============================================================

app.websocket_running = False


# ============================================================
# DEBUG INFORMATION
# ============================================================

print("=" * 60)
print("FYERS AI STOCK SCREENER")
print("=" * 60)

print("CLIENT ID:", CLIENT_ID)
print("REDIRECT URI:", REDIRECT_URI)
print("SECRET LOADED:", bool(SECRET_KEY))

print("=" * 60)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    connected = bool(session.get("access_token"))

    # --------------------------------------------------------
    # Connection status
    # --------------------------------------------------------

    if connected:
        connection_status = "CONNECTED"
        connection_class = "connected"
    else:
        connection_status = "NOT CONNECTED"
        connection_class = "disconnected"

    # --------------------------------------------------------
    # Dashboard
    # --------------------------------------------------------

    return render_template(
        "dashboard.html",
        connected=connected,
        connection_status=connection_status,
        connection_class=connection_class
    )

    connected = bool(session.get("access_token"))

    if connected:

        connection_status = """
        <p style="color:green;">
            <b>✓ FYERS Connected</b>
        </p>
        """

        buttons = """
        <a href="/profile">
            <button>Test FYERS API</button>
        </a>

        <a href="/test-data">
            <button>Get Market Data</button>
        </a>

        <a href="/smma">
            <button>SMMA Analysis</button>
        </a>

        <a href="/trades">
            <button>Trade Analysis</button>
        </a>

        <a href="/build-dataset">
            <button>Build ML Dataset</button>
        </a>

        <a href="/start-live">
            <button>Start Live Data</button>
        </a>

        <a href="/ltq-analysis">
            <button>LTQ Analysis</button>
        </a>

        <a href="/token-status">
            <button>Token Status</button>
        </a>

        <a href="/logout">
            <button style="background:#d32f2f;">
                Logout
            </button>
        </a>
        """

    else:

        connection_status = """
        <p style="color:red;">
            <b>✗ FYERS Not Connected</b>
        </p>
        """

        buttons = """
        <a href="/login">
            <button>Connect FYERS</button>
        </a>
        """

    return f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>AI Stock Screener</title>

        <style>

            body {{
                font-family: Arial, sans-serif;
                background: #f5f7fa;
                text-align: center;
                padding-top: 60px;
            }}

            .container {{
                background: white;
                width: 650px;
                margin: auto;
                padding: 40px;
                border-radius: 12px;
                box-shadow:
                    0 4px 15px rgba(0,0,0,0.1);
            }}

            h1 {{
                color: #222;
            }}

            p {{
                color: #555;
            }}

            button {{
                background: #1976d2;
                color: white;
                border: none;
                padding: 12px 20px;
                margin: 6px;
                font-size: 15px;
                border-radius: 6px;
                cursor: pointer;
            }}

            button:hover {{
                opacity: 0.85;
            }}

        </style>

    </head>

    <body>

        <div class="container">

            <h1>AI Stock Screener</h1>

            <p>FYERS API Authentication</p>

            <p>
                Non-Trading / Market Data Application
            </p>

            {connection_status}

            <br>

            {buttons}

        </div>

    </body>

    </html>
    """


# ============================================================
# FYERS LOGIN
# ============================================================

@app.route("/login")
def login():

    print("\n" + "=" * 60)
    print("STARTING FYERS AUTHENTICATION")
    print("=" * 60)

    print("CLIENT ID:", CLIENT_ID)
    print("REDIRECT URI:", REDIRECT_URI)
    print("SECRET LOADED:", bool(SECRET_KEY))

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    if not CLIENT_ID:

        return """
        <h2>Configuration Error</h2>

        <p>FYERS_CLIENT_ID is missing.</p>

        <p>Check your .env file.</p>
        """

    if not SECRET_KEY:

        return """
        <h2>Configuration Error</h2>

        <p>FYERS_SECRET_KEY is missing.</p>

        <p>Check your .env file.</p>
        """

    if not REDIRECT_URI:

        return """
        <h2>Configuration Error</h2>

        <p>FYERS_REDIRECT_URI is missing.</p>

        <p>Check your .env file.</p>
        """

    try:

        # ----------------------------------------------------
        # Create FYERS session
        # ----------------------------------------------------

        session_model = fyersModel.SessionModel(

            client_id=CLIENT_ID,

            secret_key=SECRET_KEY,

            redirect_uri=REDIRECT_URI,

            response_type="code",

            grant_type="authorization_code",

            state="ai_stock_screener"
        )

        # ----------------------------------------------------
        # Generate authentication URL
        # ----------------------------------------------------

        login_url = session_model.generate_authcode()

        print("\nFYERS LOGIN URL GENERATED")
        print(login_url)

        print("=" * 60)

        return redirect(login_url)

    except Exception as e:

        print("\nERROR WHILE GENERATING LOGIN URL")
        print(str(e))

        return f"""
        <h2>FYERS Login Error</h2>

        <pre>{str(e)}</pre>

        <br>

        <a href="/">
            Back to Home
        </a>
        """


# ============================================================
# FYERS CALLBACK
# ============================================================

@app.route("/callback")
def callback():

    print("\n" + "=" * 60)
    print("FYERS CALLBACK RECEIVED")
    print("=" * 60)

    # --------------------------------------------------------
    # Get authorization code
    # --------------------------------------------------------

    auth_code = request.args.get("auth_code")

    state = request.args.get("state")

    error = request.args.get("error")

    error_message = request.args.get("error_message")

    print("AUTH CODE RECEIVED:", bool(auth_code))
    print("STATE:", state)

    # --------------------------------------------------------
    # Handle FYERS error
    # --------------------------------------------------------

    if error:

        print("FYERS ERROR:", error)
        print("FYERS ERROR MESSAGE:", error_message)

        return f"""
        <h1>FYERS Authentication Failed</h1>

        <p>
            <b>Error:</b> {error}
        </p>

        <p>
            <b>Error Message:</b>
            {error_message}
        </p>

        <br>

        <a href="/">
            Try Again
        </a>
        """

    # --------------------------------------------------------
    # Check auth code
    # --------------------------------------------------------

    if not auth_code:

        print("NO AUTH_CODE RECEIVED")

        return """
        <h1>Authentication Failed</h1>

        <p>
            No auth_code received from FYERS.
        </p>

        <p>
            Start authentication from the home page.
        </p>

        <br>

        <a href="/">
            Back to Home
        </a>
        """

    print("AUTH CODE RECEIVED SUCCESSFULLY")

    # ========================================================
    # EXCHANGE AUTH CODE FOR ACCESS TOKEN
    # ========================================================

    try:

        print("\nGenerating FYERS access token...")

        # ----------------------------------------------------
        # Create session model
        # ----------------------------------------------------

        session_model = fyersModel.SessionModel(

            client_id=CLIENT_ID,

            secret_key=SECRET_KEY,

            redirect_uri=REDIRECT_URI,

            response_type="code",

            grant_type="authorization_code"
        )

        # ----------------------------------------------------
        # Set authorization code
        # ----------------------------------------------------

        session_model.set_token(auth_code)

        # ----------------------------------------------------
        # Generate token
        # ----------------------------------------------------

        response = session_model.generate_token()

        print("\nFYERS TOKEN RESPONSE:")

        # Do NOT print the complete token
        if isinstance(response, dict):

            safe_response = dict(response)

            if "access_token" in safe_response:

                token = safe_response["access_token"]

                safe_response["access_token"] = (
                    token[:10] + "..."
                    if token
                    else None
                )

            print(safe_response)

        else:

            print(response)

        # ----------------------------------------------------
        # Validate response
        # ----------------------------------------------------

        if not isinstance(response, dict):

            return f"""
            <h2>Token Generation Failed</h2>

            <pre>{response}</pre>

            <a href="/">
                Try Again
            </a>
            """

        if "access_token" not in response:

            return f"""
            <h2>Access Token Generation Failed</h2>

            <p>
                FYERS did not return an access token.
            </p>

            <pre>{response}</pre>

            <br>

            <a href="/">
                Try Again
            </a>
            """

        # ====================================================
        # STORE FRESH TOKEN IN FLASK SESSION
        # ====================================================

        access_token = response["access_token"]

        session["access_token"] = access_token

        print("\nACCESS TOKEN RECEIVED SUCCESSFULLY")

        print(
            "Token length:",
            len(access_token)
        )

        print(
            "Token stored in Flask session."
        )

        print("=" * 60)

        # ----------------------------------------------------
        # Success page
        # ----------------------------------------------------

        return """
        <!DOCTYPE html>

        <html>

        <head>

            <title>FYERS Connected</title>

            <style>

                body {
                    font-family: Arial;
                    background: #f5f7fa;
                    text-align: center;
                    padding-top: 100px;
                }

                .container {
                    background: white;
                    width: 550px;
                    margin: auto;
                    padding: 40px;
                    border-radius: 12px;
                    box-shadow:
                        0 4px 15px rgba(0,0,0,0.1);
                }

                .success {
                    color: green;
                }

                button {
                    background: #1976d2;
                    color: white;
                    border: none;
                    padding: 12px 25px;
                    font-size: 16px;
                    border-radius: 6px;
                    cursor: pointer;
                }

            </style>

        </head>

        <body>

            <div class="container">

                <h1 class="success">
                    ✓ FYERS Connected Successfully
                </h1>

                <p>
                    Authorization completed successfully.
                </p>

                <p>
                    Fresh access token received.
                </p>

                <p>
                    The token has been stored in the
                    Flask session.
                </p>

                <br>

                <a href="/profile">

                    <button>
                        Test FYERS API
                    </button>

                </a>

                <a href="/test-data">

                    <button>
                        Test Market Data
                    </button>

                </a>

            </div>

        </body>

        </html>
        """

    except Exception as e:

        print("\nERROR GENERATING ACCESS TOKEN")

        print(str(e))

        return f"""
        <h1>Access Token Error</h1>

        <pre>{str(e)}</pre>

        <br>

        <a href="/">
            Try Again
        </a>
        """


# ============================================================
# TEST FYERS PROFILE API
# ============================================================

@app.route("/profile")
def profile():

    access_token = session.get("access_token")

    # --------------------------------------------------------
    # Check token
    # --------------------------------------------------------

    if not access_token:

        return """
        <h2>Not Connected</h2>

        <p>
            Please connect your FYERS account first.
        </p>

        <a href="/">
            Go Home
        </a>
        """

    try:

        print("\nTesting FYERS Profile API...")

        # ----------------------------------------------------
        # Create FYERS API object
        # ----------------------------------------------------

        fyers = fyersModel.FyersModel(

            client_id=CLIENT_ID,

            token=access_token,

            is_async=False,

            log_path=""
        )

        # ----------------------------------------------------
        # Get profile
        # ----------------------------------------------------

        response = fyers.get_profile()

        print("\nFYERS PROFILE RESPONSE:")
        print(response)

        # ----------------------------------------------------
        # Check response
        # ----------------------------------------------------

        if response.get("s") == "error":

            return f"""
            <h1>FYERS API Authentication Error</h1>

            <pre>{response}</pre>

            <br>

            <a href="/logout">
                Clear Session
            </a>

            <br><br>

            <a href="/">
                Home
            </a>
            """

        return f"""
        <!DOCTYPE html>

        <html>

        <head>

            <title>FYERS API Test</title>

        </head>

        <body>

            <h1>FYERS API Test</h1>

            <h2 style="color:green;">
                Connection Successful ✓
            </h2>

            <h3>API Response:</h3>

            <pre>{response}</pre>

            <br>

            <a href="/">
                Back to Home
            </a>

        </body>

        </html>
        """

    except Exception as e:

        print("\nFYERS API ERROR:")

        print(str(e))

        return f"""
        <h1>FYERS API Error</h1>

        <pre>{str(e)}</pre>

        <br>

        <a href="/">
            Back to Home
        </a>
        """


# ============================================================
# TEST HISTORICAL MARKET DATA
# ============================================================

@app.route("/test-data")
def test_data():

    access_token = session.get("access_token")

    # --------------------------------------------------------
    # Check token
    # --------------------------------------------------------

    if not access_token:

        return """
        <h2>FYERS Not Connected</h2>

        <a href="/login">
            Connect FYERS
        </a>
        """

    symbol = "NSE:SBIN-EQ"

    print("\n" + "=" * 60)
    print("TESTING HISTORICAL MARKET DATA")
    print("=" * 60)

    print("Symbol:", symbol)

    print(
        "Token length:",
        len(access_token)
    )

    try:

        response = get_history(

            access_token,

            symbol,

            resolution="5"
        )

        print("\nHISTORY RESPONSE:")

        print(response)

        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        if response.get("s") != "ok":

            return f"""
            <h1>FYERS Market Data Error</h1>

            <pre>{response}</pre>

            <br>

            <p>
                Your Flask session token may be expired.
            </p>

            <a href="/logout">
                Logout / Clear Token
            </a>

            <br><br>

            <a href="/">
                Home
            </a>
            """

        # ----------------------------------------------------
        # Candles
        # ----------------------------------------------------

        candles = response.get(
            "candles",
            []
        )

        return f"""
        <!DOCTYPE html>

        <html>

        <head>

            <title>FYERS Market Data</title>

        </head>

        <body>

            <h1>FYERS Market Data</h1>

            <h2>{symbol}</h2>

            <p>
                Status:
                <b style="color:green;">
                    {response.get("s")}
                </b>
            </p>

            <p>
                Candles received:
                <b>{len(candles)}</b>
            </p>

            <h3>First 10 Candles</h3>

            <pre>{candles[:10]}</pre>

            <br>

            <a href="/smma">
                Run SMMA Analysis
            </a>

            <br><br>

            <a href="/">
                Home
            </a>

        </body>

        </html>
        """

    except Exception as e:

        print("\nMARKET DATA ERROR:")

        print(str(e))

        return f"""
        <h1>Market Data Error</h1>

        <pre>{str(e)}</pre>

        <br>

        <a href="/">
            Home
        </a>
        """


# ============================================================
# SMMA ANALYSIS
# ============================================================

@app.route("/smma")
def smma_analysis():

    access_token = session.get("access_token")

    if not access_token:

        return """
        <h2>FYERS Not Connected</h2>

        <a href="/login">
            Connect FYERS
        </a>
        """

    symbol = "NSE:SBIN-EQ"

    # --------------------------------------------------------
    # Get historical data
    # --------------------------------------------------------

    response = get_history(

        access_token,

        symbol,

        resolution="5"
    )

    if response.get("s") != "ok":

        return f"""
        <h2>FYERS Data Error</h2>

        <pre>{response}</pre>

        <br>

        <a href="/logout">
            Clear Session
        </a>
        """

    candles = response.get(
        "candles",
        []
    )

    if not candles:

        return """
        <h2>No historical candles returned.</h2>
        """

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

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
    # Calculate SMMA
    # --------------------------------------------------------

    df = detect_crossovers(df)

    # --------------------------------------------------------
    # Get crossover events
    # --------------------------------------------------------

    events = get_crossover_events(df)

    # --------------------------------------------------------
    # Latest rows
    # --------------------------------------------------------

    latest = df.tail(20)

    # --------------------------------------------------------
    # Event table
    # --------------------------------------------------------

    event_columns = [
        "timestamp",
        "close",
        "SMMA20",
        "SMMA120",
        "SIGNAL"
    ]

    event_columns = [
        c
        for c in event_columns
        if c in events.columns
    ]

    event_html = events[
        event_columns
    ].tail(30).to_html(
        index=False
    )

    # --------------------------------------------------------
    # Latest table
    # --------------------------------------------------------

    latest_columns = [
        "timestamp",
        "close",
        "SMMA20",
        "SMMA120",
        "BUY_SIGNAL",
        "SELL_SIGNAL"
    ]

    latest_columns = [
        c
        for c in latest_columns
        if c in latest.columns
    ]

    latest_html = latest[
        latest_columns
    ].to_html(
        index=False
    )

    return f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>SMMA Analysis</title>

        <style>

            body {{
                font-family: Arial;
                padding: 30px;
                background: #f5f7fa;
            }}

            .card {{
                background: white;
                padding: 25px;
                margin-bottom: 25px;
                border-radius: 10px;
                box-shadow:
                    0 2px 10px rgba(0,0,0,0.1);
            }}

            table {{
                border-collapse: collapse;
                width: 100%;
            }}

            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}

            th {{
                background: #eee;
            }}

        </style>

    </head>

    <body>

        <h1>AI Stock Screener</h1>

        <h2>SMMA Crossover Analysis</h2>

        <div class="card">

            <h3>{symbol}</h3>

            <p>
                SMMA Fast:
                <b>20</b>
            </p>

            <p>
                SMMA Slow:
                <b>120</b>
            </p>

            <p>
                Candles:
                <b>{len(df)}</b>
            </p>

        </div>

        <div class="card">

            <h2>Crossover Events</h2>

            {event_html}

        </div>

        <div class="card">

            <h2>Latest Market Data</h2>

            {latest_html}

        </div>

        <br>

        <a href="/">
            Home
        </a>

    </body>

    </html>
    """


# ============================================================
# TRADE ANALYSIS
# ============================================================

@app.route("/trades")
def trade_analysis():

    access_token = session.get("access_token")

    if not access_token:

        return """
        <h2>FYERS Not Connected</h2>

        <a href="/login">
            Connect FYERS
        </a>
        """

    symbol = "NSE:SBIN-EQ"

    # --------------------------------------------------------
    # Historical data
    # --------------------------------------------------------

    response = get_history(

        access_token,

        symbol,

        resolution="5"
    )

    if response.get("s") != "ok":

        return f"""
        <h2>FYERS Data Error</h2>

        <pre>{response}</pre>
        """

    candles = response.get(
        "candles",
        []
    )

    if not candles:

        return """
        <h2>No historical data available.</h2>
        """

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

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
    # SMMA
    # --------------------------------------------------------

    df = detect_crossovers(df)

    # --------------------------------------------------------
    # Evaluate trades
    # --------------------------------------------------------

    trades = evaluate_trades(df)

    if trades.empty:

        return """
        <h2>No completed trades found.</h2>

        <p>
        More historical data is required to evaluate
        crossover trades.
        </p>

        <br>

        <a href="/smma">
            SMMA Analysis
        </a>
        """

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_trades = len(trades)

    profitable_trades = (
        trades["profitable"] == 1
    ).sum()

    losing_trades = (
        trades["profitable"] == 0
    ).sum()

    total_pnl = trades["pnl"].sum()

    win_rate = (
        profitable_trades /
        total_trades
    ) * 100

    # --------------------------------------------------------
    # Table
    # --------------------------------------------------------

    table_columns = [
        "signal",
        "entry_timestamp",
        "entry_ltp",
        "exit_timestamp",
        "exit_ltp",
        "pnl",
        "profitable"
    ]

    table_columns = [
        c
        for c in table_columns
        if c in trades.columns
    ]

    table = trades[
        table_columns
    ].copy()

    if "profitable" in table.columns:

        table["result"] = table[
            "profitable"
        ].map({
            1: "PROFIT",
            0: "LOSS"
        })

        table = table.drop(
            columns=["profitable"]
        )

    table_html = table.to_html(
        index=False
    )

    return f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>Trade Analysis</title>

        <style>

            body {{
                font-family: Arial;
                background: #f5f7fa;
                padding: 30px;
            }}

            .container {{
                max-width: 1200px;
                margin: auto;
            }}

            .cards {{
                display: flex;
                gap: 20px;
                margin-bottom: 30px;
            }}

            .card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                flex: 1;
                box-shadow:
                    0 2px 10px rgba(0,0,0,0.1);
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
            }}

            th, td {{
                padding: 10px;
                border: 1px solid #ddd;
                text-align: center;
            }}

            th {{
                background: #eee;
            }}

        </style>

    </head>

    <body>

    <div class="container">

        <h1>AI Stock Screener</h1>

        <h2>SMMA Trade Analysis</h2>

        <div class="cards">

            <div class="card">
                <h3>Total Trades</h3>
                <h2>{total_trades}</h2>
            </div>

            <div class="card">
                <h3>Profitable Trades</h3>
                <h2>{profitable_trades}</h2>
            </div>

            <div class="card">
                <h3>Losing Trades</h3>
                <h2>{losing_trades}</h2>
            </div>

            <div class="card">
                <h3>Win Rate</h3>
                <h2>{win_rate:.2f}%</h2>
            </div>

            <div class="card">
                <h3>Total P/L</h3>
                <h2>₹{total_pnl:.2f}</h2>
            </div>

        </div>

        <h2>Completed Trades</h2>

        {table_html}

        <br>

        <a href="/">
            Home
        </a>

    </div>

    </body>

    </html>
    """


# ============================================================
# BUILD ML DATASET
# ============================================================

@app.route("/build-dataset")
def build_dataset_route():

    access_token = session.get("access_token")

    if not access_token:

        return """
        <h2>FYERS Not Connected</h2>

        <a href="/login">
            Connect FYERS
        </a>
        """

    print("\n")
    print("=" * 60)
    print("BUILDING ML DATASET")
    print("=" * 60)

    try:

        dataset = build_dataset(
            access_token
        )

        if dataset.empty:

            return """
            <h2>Dataset Could Not Be Created</h2>

            <p>
            No completed crossover trades were found.
            </p>
            """

        file_path = save_dataset(
            dataset
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_trades = len(dataset)

        profitable = (
            dataset["profitable"] == 1
        ).sum()

        losing = (
            dataset["profitable"] == 0
        ).sum()

        total_pnl = dataset["pnl"].sum()

        win_rate = (
            profitable /
            total_trades
        ) * 100

        # ----------------------------------------------------
        # Stock statistics
        # ----------------------------------------------------

        stock_stats = (

            dataset

            .groupby("symbol")

            .agg(
                trades=("pnl", "count"),
                profitable=("profitable", "sum"),
                pnl=("pnl", "sum")
            )

            .reset_index()
        )

        stock_stats["win_rate"] = (

            stock_stats["profitable"] /
            stock_stats["trades"]

        ) * 100

        stock_html = stock_stats.to_html(
            index=False
        )

        # ----------------------------------------------------
        # Dataset preview
        # ----------------------------------------------------

        preview_columns = [

            "symbol",
            "signal",
            "entry_ltp",
            "exit_ltp",
            "pnl",
            "profitable",
            "smma_spread_pct",
            "volume_ratio",
            "return_3",
            "volatility_5"
        ]

        preview_columns = [

            c
            for c in preview_columns
            if c in dataset.columns

        ]

        preview_html = dataset[
            preview_columns
        ].tail(30).to_html(
            index=False
        )

        return f"""
        <!DOCTYPE html>

        <html>

        <head>

            <title>ML Dataset</title>

            <style>

                body {{
                    font-family: Arial;
                    background: #f5f7fa;
                    padding: 30px;
                }}

                .container {{
                    max-width: 1400px;
                    margin: auto;
                }}

                .cards {{
                    display: flex;
                    gap: 20px;
                    margin-bottom: 30px;
                }}

                .card {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    flex: 1;
                    box-shadow:
                        0 2px 10px rgba(0,0,0,0.1);
                }}

                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: white;
                    margin-bottom: 30px;
                }}

                th, td {{
                    padding: 8px;
                    border: 1px solid #ddd;
                    text-align: center;
                }}

                th {{
                    background: #eee;
                }}

            </style>

        </head>

        <body>

        <div class="container">

            <h1>AI Stock Screener</h1>

            <h2>ML Dataset Builder</h2>

            <div class="cards">

                <div class="card">
                    <h3>Total Trades</h3>
                    <h2>{total_trades}</h2>
                </div>

                <div class="card">
                    <h3>Profitable</h3>
                    <h2>{profitable}</h2>
                </div>

                <div class="card">
                    <h3>Losses</h3>
                    <h2>{losing}</h2>
                </div>

                <div class="card">
                    <h3>Win Rate</h3>
                    <h2>{win_rate:.2f}%</h2>
                </div>

                <div class="card">
                    <h3>Total P/L</h3>
                    <h2>₹{total_pnl:.2f}</h2>
                </div>

            </div>

            <h2>Stock-wise Statistics</h2>

            {stock_html}

            <h2>Dataset Preview</h2>

            {preview_html}

            <p>
                Dataset saved as:
                <b>{file_path}</b>
            </p>

            <br>

            <a href="/">
                Home
            </a>

        </div>

        </body>

        </html>
        """

    except Exception as e:

        print("\nDATASET ERROR:")
        print(str(e))

        return f"""
        <h1>Dataset Builder Error</h1>

        <pre>{str(e)}</pre>

        <br>

        <a href="/">
            Home
        </a>
        """


# ============================================================
# START LIVE WEBSOCKET
# ============================================================

@app.route("/start-live")
def start_live():

    access_token = session.get("access_token")

    if not access_token:

        return """
        <h2>FYERS Not Connected</h2>

        <a href="/login">
            Connect FYERS
        </a>
        """

    try:

        # ----------------------------------------------------
        # Prevent duplicate WebSocket
        # ----------------------------------------------------

        if getattr(
            app,
            "websocket_running",
            False
        ):

            return """
            <h1>
                Live Market Data Already Running
            </h1>

            <p>
                FYERS WebSocket is already running.
            </p>

            <p>
                Symbol:
                <b>NSE:SBIN-EQ</b>
            </p>

            <p>
                Data file:
            </p>

            <pre>
data/live_ticks.csv
            </pre>

            <br>

            <a href="/ltq-analysis">
                View LTQ Analysis
            </a>
            """

        # ----------------------------------------------------
        # Set status
        # ----------------------------------------------------

        app.websocket_running = True

        # ----------------------------------------------------
        # Start WebSocket thread
        # ----------------------------------------------------

        thread = threading.Thread(

            target=start_websocket,

            args=(access_token,),

            daemon=True
        )

        thread.start()

        return """
        <h1>
            Live Market Data Started
        </h1>

        <p>
            FYERS WebSocket is starting...
        </p>

        <p>
            Symbol:
            <b>NSE:SBIN-EQ</b>
        </p>

        <p>
            LTP and LTQ will be saved to:
        </p>

        <pre>
data/live_ticks.csv
        </pre>

        <br>

        <a href="/ltq-analysis">
            View LTQ Analysis
        </a>

        <br><br>

        <a href="/">
            Home
        </a>
        """

    except Exception as e:

        app.websocket_running = False

        print("\nWEBSOCKET ERROR:")
        print(str(e))

        return f"""
        <h2>WebSocket Error</h2>

        <pre>{str(e)}</pre>
        """


# ============================================================
# LTQ ANALYSIS
# ============================================================

@app.route("/ltq-analysis")
def ltq_analysis():

    try:

        df = generate_ltq_features()

        if df.empty:

            return """
            <h2>No LTQ Data Available.</h2>

            <p>
            Start the live WebSocket first.
            </p>

            <a href="/start-live">
                Start Live Data
            </a>
            """

        latest = df.tail(50)

        columns = [

            "timestamp",
            "symbol",
            "ltp",
            "ltq",
            "ltq_avg_2m",
            "ltq_avg_5m",
            "ltq_ratio",
            "ltq_change_pct",
            "ltq_zscore",
            "ltp_change_pct",
            "ltq_spike",
            "price_direction"
        ]

        columns = [

            c
            for c in columns
            if c in latest.columns

        ]

        table = latest[
            columns
        ].to_html(
            index=False
        )

        last = df.iloc[-1]

        return f"""
        <!DOCTYPE html>

        <html>

        <head>

            <title>LTQ Analysis</title>

        </head>

        <body>

        <h1>AI Stock Screener</h1>

        <h2>Real-Time LTQ Analysis</h2>

        <h3>Current LTP</h3>

        <p>
            ₹{last["ltp"]:.2f}
        </p>

        <h3>Current LTQ</h3>

        <p>
            {last["ltq"]:.0f}
        </p>

        <h3>2-Min LTQ Average</h3>

        <p>
            {last["ltq_avg_2m"]:.2f}
        </p>

        <h3>5-Min LTQ Average</h3>

        <p>
            {last["ltq_avg_5m"]:.2f}
        </p>

        <h3>LTQ Ratio</h3>

        <p>
            {last["ltq_ratio"]:.2f}
        </p>

        <h2>Latest LTQ Data</h2>

        {table}

        <br>

        <a href="/">
            Home
        </a>

        </body>

        </html>
        """

    except Exception as e:

        print("\nLTQ ANALYSIS ERROR:")
        print(str(e))

        return f"""
        <h2>LTQ Analysis Error</h2>

        <pre>{str(e)}</pre>
        """


# ============================================================
# TOKEN STATUS
# ============================================================

@app.route("/token-status")
def token_status():

    access_token = session.get(
        "access_token"
    )

    if not access_token:

        return """
        <h1>Token Status</h1>

        <p style="color:red;">
            ✗ No access token in Flask session.
        </p>

        <a href="/login">
            Connect FYERS
        </a>
        """

    # --------------------------------------------------------
    # Mask token
    # --------------------------------------------------------

    if len(access_token) > 15:

        masked_token = (
            access_token[:10]
            + "..."
            + access_token[-5:]
        )

    else:

        masked_token = "***"

    return f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>Token Status</title>

    </head>

    <body>

        <h1>FYERS Token Status</h1>

        <p style="color:green;">
            <b>✓ Token exists</b>
        </p>

        <p>
            Token length:
            <b>{len(access_token)}</b>
        </p>

        <p>
            Token:
            <b>{masked_token}</b>
        </p>

        <p>
            Token is stored in the Flask session.
        </p>

        <br>

        <a href="/profile">
            Test FYERS API
        </a>

        <br><br>

        <a href="/test-data">
            Test Market Data
        </a>

        <br><br>

        <a href="/">
            Home
        </a>

    </body>

    </html>
    """
@app.route("/download-historical")
def download_historical():

    access_token = session.get("access_token")

    if not access_token:

        return """
        <h2>FYERS not connected</h2>

        <p>
        Please connect your FYERS account first.
        </p>

        <a href="/login">
            Connect FYERS
        </a>
        """

    try:

        print()
        print("=" * 70)
        print("STARTING HISTORICAL DATA DOWNLOAD")
        print("=" * 70)

        successful, failed = download_all(
            access_token
        )

        return f"""
        <!DOCTYPE html>

        <html>

        <head>
            <title>Historical Data</title>
        </head>

        <body>

            <h1>
                Historical Data Download
            </h1>

            <h2>
                Download Complete ✓
            </h2>

            <p>
                Successful stocks:
                <b>{successful}</b>
            </p>

            <p>
                Failed stocks:
                <b>{failed}</b>
            </p>

            <p>
                Data saved to:
            </p>

            <pre>
data/historical/
            </pre>

            <br>

            <a href="/smma">
                Go to SMMA Analysis
            </a>

        </body>

        </html>
        """

    except Exception as e:

        print()
        print("HISTORICAL DATA ERROR:")
        print(str(e))

        return f"""
        <h2>Historical Data Error</h2>

        <pre>{str(e)}</pre>

        <br>

        <a href="/">
            Home
        </a>
        """


# ============================================================
# DASHBOARD CHART
# ============================================================

@app.route("/dashboard-chart")
def dashboard_chart():

    access_token = session.get("access_token")

    if not access_token:

        return """
        <h2>FYERS Not Connected</h2>

        <a href="/login">
            Connect FYERS
        </a>
        """

    symbol = "NSE:SBIN-EQ"

    try:

        response = get_history(
            access_token,
            symbol,
            resolution="5"
        )

        if response.get("s") != "ok":

            return f"""
            <h2>Market Data Error</h2>
            <pre>{response}</pre>
            """

        candles = response.get("candles", [])

        if not candles:

            return """
            <h2>No market data available.</h2>
            """

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

        df = detect_crossovers(df)

        # Last 100 candles
        df = df.tail(100).copy()

        # Convert timestamp
        df["datetime"] = pd.to_datetime(
            df["timestamp"],
            unit="s"
        )

        fig = go.Figure()

        # Price
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["close"],
                mode="lines",
                name="LTP"
            )
        )

        # SMMA 20
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["SMMA20"],
                mode="lines",
                name="SMMA 20"
            )
        )

        # SMMA 120
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["SMMA120"],
                mode="lines",
                name="SMMA 120"
            )
        )

        # BUY signals
        buy = df[df["BUY_SIGNAL"]]

        fig.add_trace(
            go.Scatter(
                x=buy["datetime"],
                y=buy["close"],
                mode="markers",
                name="BUY",
                marker=dict(
                    size=10,
                    symbol="triangle-up"
                )
            )
        )

        # SELL signals
        sell = df[df["SELL_SIGNAL"]]

        fig.add_trace(
            go.Scatter(
                x=sell["datetime"],
                y=sell["close"],
                mode="markers",
                name="SELL",
                marker=dict(
                    size=10,
                    symbol="triangle-down"
                )
            )
        )

        fig.update_layout(
            title="SBIN - SMMA Crossover Analysis",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_white",
            height=600
        )

        chart = fig.to_html(
            full_html=False
        )

        return f"""
        <!DOCTYPE html>

        <html>

        <head>

            <title>SMMA Chart</title>

            <style>

                body {{
                    font-family: Arial;
                    background: #f4f6f9;
                    padding: 30px;
                }}

                .container {{
                    max-width: 1400px;
                    margin: auto;
                    background: white;
                    padding: 25px;
                    border-radius: 12px;
                }}

                a {{
                    text-decoration: none;
                }}

                button {{
                    padding: 12px 20px;
                    background: #2563eb;
                    color: white;
                    border: none;
                    border-radius: 6px;
                }}

            </style>

        </head>

        <body>

        <div class="container">

            <h1>AI Stock Screener</h1>

            <h2>SMMA Crossover Chart</h2>

            <p>
                Symbol:
                <b>NSE:SBIN-EQ</b>
            </p>

            <p>
                Strategy:
                <b>SMMA 20 / SMMA 120</b>
            </p>

            {chart}

            <br>

            <a href="/">
                <button>
                    Back to Dashboard
                </button>
            </a>

        </div>

        </body>

        </html>
        """

    except Exception as e:

        return f"""
        <h2>Chart Error</h2>

        <pre>{str(e)}</pre>

        <br>

        <a href="/">
            Back to Dashboard
        </a>
        """
# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    print("\nFlask session cleared.")

    return """
    <h2>FYERS Session Cleared</h2>

    <p>
        The local Flask session and access token
        have been cleared.
    </p>

    <br>

    <a href="/login">
        Connect FYERS Again
    </a>

    <br><br>

    <a href="/">
        Home
    </a>
    """


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("AI STOCK SCREENER - FYERS DATA APPLICATION")
    print("=" * 60)

    print("\nApplication URL:")
    print("https://127.0.0.1:5000")

    print("\nCallback URL:")
    print("https://127.0.0.1:5000/callback")

    print("\nNon-trading / Data-only application")

    print("=" * 60)

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        ssl_context="adhoc"
    )