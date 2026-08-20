import os
import webbrowser
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

# Load credentials from .env
load_dotenv()

CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
SECRET_KEY = os.getenv("FYERS_SECRET_KEY")
REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI")

# Check credentials
if not CLIENT_ID or not SECRET_KEY or not REDIRECT_URI:
    raise ValueError(
        "Missing FYERS credentials. Check your .env file."
    )

# Create FYERS session
session = fyersModel.SessionModel(
    client_id=CLIENT_ID,
    secret_key=SECRET_KEY,
    redirect_uri=REDIRECT_URI,
    response_type="code",
    grant_type="authorization_code"
)

# Generate login URL
auth_url = session.generate_authcode()

print("\n====================================")
print("FYERS AUTHENTICATION")
print("====================================\n")

print("Open this URL in your browser:\n")
print(auth_url)

# Open browser automatically
webbrowser.open(auth_url)

print("\nLogin to FYERS and authorize the application.")
print("After authorization, FYERS will redirect you to your Redirect URI.")
print("\nCopy the value of 'auth_code' from that URL.")

auth_code = input("\nEnter auth_code: ").strip()

# Exchange auth code for access token
session.set_token(auth_code)

response = session.generate_token()

print("\nFYERS RESPONSE:")
print(response)

if response.get("s") == "ok" and "access_token" in response:

    access_token = response["access_token"]

    with open("access_token.txt", "w") as file:
        file.write(access_token)

    print("\n====================================")
    print("SUCCESS!")
    print("Access token generated.")
    print("Saved to access_token.txt")
    print("====================================")

else:

    print("\n====================================")
    print("AUTHENTICATION FAILED")
    print("====================================")
    print(response)