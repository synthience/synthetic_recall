import jwt
import time

# Token claims for playground
claims = {
    'iss': 'devkey',  # API Key
    'sub': 'playground-user',  # User identity
    'jti': str(int(time.time())),  # Unique token ID
    'exp': int(time.time()) + 86400,  # Token expiration (24 hours)
    'nbf': int(time.time()),  # Not valid before current time
    'room': 'playground',
    'roomJoin': True,
    'roomCreate': True,
    'roomAdmin': True,
    'roomList': True,
    'canPublish': True,
    'canSubscribe': True,
    'canPublishData': True,
    'hidden': False,
    'recorder': False,
    'name': 'Playground User',
    'metadata': ''
}

# Generate JWT token
token = jwt.encode(claims, 'secret', algorithm='HS256')
print(token)
