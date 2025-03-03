import jwt
import time

api_key = "devkey"
api_secret = "secret"

# Define the claims for the token
claims = {
    # Required claims
    "iss": api_key,  # API Key
    "sub": "test-identity",  # User identity
    "jti": str(int(time.time())),  # Unique token ID
    "exp": int(time.time()) + 3600,  # Expiration (1 hour from now)
    "nbf": int(time.time()),  # Not valid before current time
    
    # Video grant
    "video": {
        "room": "agent_room",
        "roomJoin": True,
        "canPublish": True,
        "canSubscribe": True,
        "canPublishData": True
    },
    
    # Optional user info
    "name": "test-user",
    "metadata": ""
}

# Generate the JWT token
token = jwt.encode(claims, api_secret, algorithm='HS256')
print(f"LiveKit Token: {token}")
