from flask import Flask, request

app = Flask(__name__)


@app.route("/", methods=["GET"])
@app.route("/callback", methods=["GET"])
def callback():
    auth_code = request.args.get("auth_code")
    state = request.args.get("state")

    print("\n===================================")
    print("FYERS CALLBACK RECEIVED")
    print("===================================")

    print("Auth Code:", auth_code)
    print("State:", state)

    if auth_code:
        with open("auth_code.txt", "w") as f:
            f.write(auth_code)

        return """
        <html>
        <body>
            <h2>✅ FYERS Authentication Successful!</h2>
            <p>Auth code received successfully.</p>
            <p>You can close this browser window.</p>
        </body>
        </html>
        """

    return """
    <html>
    <body>
        <h2>⚠️ Callback received, but no auth_code was found.</h2>
    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        ssl_context="adhoc"
    )