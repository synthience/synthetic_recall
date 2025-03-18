#!/usr/bin/env python
# Utility to check and start the emotion analyzer container

import os
import sys
import subprocess
import time
import json
import argparse
import socket

def check_port_open(host, port):
    """Check if a port is open on the host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        result = s.connect_ex((host, port))
        return result == 0

def check_container_running(container_id=None):
    """Check if the emotion analyzer container is running."""
    try:
        # Use docker ps to check for the container
        if container_id:
            cmd = f"docker ps -q --filter id={container_id}"
        else:
            cmd = "docker ps -q --filter name=emotion-analyzer"
        
        result = subprocess.check_output(cmd, shell=True).decode().strip()
        return bool(result)
    except subprocess.CalledProcessError:
        return False

def start_container(use_compose=True):
    """Start the emotion analyzer container."""
    try:
        if use_compose:
            # Use docker-compose to start the container
            cmd = "docker-compose -f docker-compose.emotion.yml up -d"
            print("Starting emotion analyzer using docker-compose...")
        else:
            # Start container directly
            cmd = "docker run -d --name emotion-analyzer -p 5007:5007 -p 8007:8007 lucid-recall-dist-emotion-analyzer:latest"
            print("Starting emotion analyzer container directly...")
            
        subprocess.check_call(cmd, shell=True)
        print("Container started successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to start container: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Check and start the emotion analyzer container")
    parser.add_argument("--no-compose", action="store_true", help="Don't use docker-compose, run container directly")
    parser.add_argument("--container-id", help="Specific container ID to check")
    args = parser.parse_args()
    
    # Check if the WebSocket port is already open
    if check_port_open('localhost', 5007):
        print("Emotion analyzer is already running (port 5007 is open).")
        return
    
    # Check if the container is already running
    if check_container_running(args.container_id):
        print("Emotion analyzer container is running, but port 5007 is not responding.")
        print("The container might be starting up or having issues.")
        return
    
    # Container is not running, start it
    if start_container(not args.no_compose):
        # Wait for the container to start
        max_attempts = 10
        for i in range(max_attempts):
            print(f"Waiting for emotion analyzer to start (attempt {i+1}/{max_attempts})...")
            time.sleep(2)
            if check_port_open('localhost', 5007):
                print("Emotion analyzer is now running!")
                print("WebSocket API: ws://localhost:5007/analyze")
                print("Web interface: http://localhost:8007")
                return
        
        print("Timed out waiting for emotion analyzer to start.")
    else:
        print("Failed to start emotion analyzer container.")

if __name__ == "__main__":
    main()
