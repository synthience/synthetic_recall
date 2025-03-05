import jwt
import time
from typing import List, Optional

def create_token(
    api_key: str,
    api_secret: str,
    room_name: str,
    identity: str,
    ttl: int = 3600,  # 1 hour
    metadata: Optional[str] = None,
    name: Optional[str] = None,
    can_publish: bool = True,
    can_subscribe: bool = True,
    can_publish_data: bool = True
) -> str:
    """
    Create a LiveKit access token
    """
    now = int(time.time())
    
    grants = {
        "video": {
            "room": room_name,
            "roomJoin": True,
            "canPublish": can_publish,
            "canSubscribe": can_subscribe,
            "canPublishData": can_publish_data,
        }
    }
    
    if metadata:
        grants["metadata"] = metadata
    if name:
        grants["name"] = name
        
    payload = {
        "iss": api_key,  # Issuer
        "sub": identity,  # Subject (participant identity)
        "exp": now + ttl,  # Expiration
        "nbf": now,       # Not Before
        "iat": now,       # Issued At
        "jti": f"{room_name}_{identity}_{now}",  # JWT ID
        "video": grants["video"]
    }
    
    return jwt.encode(payload, api_secret, algorithm="HS256")
