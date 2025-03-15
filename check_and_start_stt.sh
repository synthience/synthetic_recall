#!/bin/bash

# Check if STT service is running
echo "Checking if STT service is running..."
if pgrep -f "python /workspace/project/server/STT_server.py" > /dev/null; then
    echo "STT service is already running."
else
    # Check if NeMo installation is done
    if pgrep -f "pip install git" > /dev/null; then
        echo "Dependencies still installing. Please wait..."
    else
        echo "Starting STT service..."
        cd /workspace/project
        python /workspace/project/server/STT_server.py &
        echo "STT service started."
    fi
fi

# Check if STT service is listening on port 5002
echo "Checking if STT service is listening on port 5002..."
# Use Python to check if port is open
python -c 'import socket; s=socket.socket(); result=s.connect_ex(("localhost", 5002)); print("STT service is listening on port 5002." if result==0 else "STT service is not listening on port 5002 yet."); s.close()'
