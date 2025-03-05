from datetime import datetime, timedelta
import jwt
import time

def create_permanent_token(
    api_key: str = "devkey",
    api_secret: str = "secret",
    room_name: str = "playground",
    identity: str = "playground-user"
) -> str:
    """Create a permanent LiveKit token"""
    
    # Set expiration to 10 years from now
    exp = int((datetime.now() + timedelta(days=3650)).timestamp())
    
    # Define token claims
    claims = {
        "iss": api_key,  # Issuer
        "sub": identity,  # Subject (user identity)
        "jti": str(int(time.time())),  # JWT ID
        "exp": exp,  # Expiration
        "name": "Playground User",
        "video": {
            "room": room_name,
            "roomCreate": True,
            "roomJoin": True,
            "roomAdmin": True,
            "roomList": True,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
            "canPublishSources": ["camera", "microphone", "screen_share"]
        }
    }
    
    # Generate token
    token = jwt.encode(claims, api_secret, algorithm="HS256")
    return token

if __name__ == "__main__":
    # Generate token
    token = create_permanent_token()
    print("\nGenerated permanent LiveKit token:")
    print(token)
    print("\nThis token will be valid for 10 years.")
