import os
import secrets
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

load_dotenv()

client_id = os.getenv("FYERS_APP_ID")
secret_key = os.getenv("FYERS_SECRET_ID")
redirect_uri = os.getenv("FYERS_REDIRECT_URI")

state = secrets.token_urlsafe(16)

session = fyersModel.SessionModel(
    client_id=client_id,
    secret_key=secret_key,
    redirect_uri=redirect_uri,
    response_type="code",
    grant_type="authorization_code",
    state=state
)

auth_url = session.generate_authcode()

print("\nAUTH URL:")
print(auth_url)