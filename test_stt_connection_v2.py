#!/usr/bin/env python
"""
Simple test script for STT Docker service connection.

Run this script to test connection to various STT Docker service endpoints.
This will help diagnose "did not receive a valid HTTP response" errors.
"""

import os
import sys
import json
import asyncio
import logging
import argparse
import ssl
import socket
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("stt_connection_test")

# Import websockets
try:
    import websockets
except ImportError:
    logger.error("websockets package is not installed. Please install it with 'pip install websocket-client'")
    sys.exit(1)

async def test_connection(url, ignore_ssl=False, verbose=False):
    """Test connection to a websocket URL.
    
    Args:
        url: WebSocket URL to test
        ignore_ssl: Whether to ignore SSL verification
        verbose: Whether to print verbose output
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    logger.info(f"Testing connection to {url}")
    
    # Parse URL
    parsed_url = urlparse(url)
    protocol = parsed_url.scheme
    host = parsed_url.netloc
    path = parsed_url.path
    
    if protocol not in ["ws", "wss"]:
        logger.error(f"Invalid protocol: {protocol}. URL must start with ws:// or wss://")
        return False
        
    # Test basic socket connection first
    try:
        if ":" in host:
            hostname, port = host.split(":")
            port = int(port)
        else:
            hostname = host
            port = 443 if protocol == "wss" else 80
            
        logger.info(f"Checking if {hostname}:{port} is reachable...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((hostname, port))
        sock.close()
        
        if result == 0:
            logger.info(f"✓ Host {hostname}:{port} is reachable")
        else:
            logger.error(f"✗ Host {hostname}:{port} is not reachable (error code: {result})")
            logger.error("This indicates a network connectivity or firewall issue.")
            return False
    except Exception as e:
        logger.error(f"Error checking host reachability: {e}")
        return False
        
    # Create SSL context if needed
    ssl_context = None
    if protocol == "wss" and ignore_ssl:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        logger.info("Using SSL context with verification disabled")
        
    # Now try WebSocket connection
    try:
        logger.info(f"Attempting WebSocket connection to {url}...")
        async with websockets.connect(
            url, 
            ping_interval=20, 
            ssl=ssl_context if protocol == "wss" else None
        ) as websocket:
            # Send a ping message
            await websocket.send(json.dumps({"type": "ping"}))
            logger.info("Sent ping message, waiting for response...")
            
            # Wait for response with timeout
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            logger.info(f"✓ Received response: {response}")
            logger.info(f"✓ Successfully connected to {url}")
            return True
    except Exception as e:
        logger.error(f"✗ WebSocket connection failed: {e}")
        logger.error("This could be due to:")
        logger.error("  1. The server is not accepting WebSocket connections on this endpoint")
        logger.error("  2. The server is expecting a different protocol or message format")
        logger.error("  3. The server is running but the WebSocket service is not active")
        return False

async def test_multiple_endpoints():
    """Test multiple common STT endpoints."""
    results = []
    
    # Common ports and paths for STT services
    endpoints = [
        "ws://localhost:5002/ws/transcribe",  # Common NeMo STT Docker default
        "ws://localhost:8000/ws/transcribe",  # Alternative port
        "ws://localhost:8765/ws/transcribe",  # WebSocket default port
        "ws://localhost:8080/ws/transcribe",  # Another common port
    ]
    
    # Add any environment-defined endpoints
    docker_endpoint = os.environ.get("NEMO_DOCKER_ENDPOINT")
    if docker_endpoint:
        endpoints.append(docker_endpoint)
        
    docker_fallback = os.environ.get("NEMO_DOCKER_ENDPOINT_FALLBACK")
    if docker_fallback:
        endpoints.append(docker_fallback)
    
    # Test each endpoint
    for endpoint in endpoints:
        success = await test_connection(endpoint, ignore_ssl=True)
        results.append((endpoint, success))
        print("\n" + "-"*50 + "\n")
    
    # Print summary
    print("\n===== SUMMARY =====\n")
    working_endpoints = []
    for endpoint, success in results:
        status = "✓ WORKING" if success else "✗ FAILED"
        print(f"{status}: {endpoint}")
        if success:
            working_endpoints.append(endpoint)
    
    if working_endpoints:
        print("\n===== RECOMMENDED CONFIGURATION =====\n")
        print(f"Add this to your environment variables or .env file:\n")
        print(f"NEMO_DOCKER_ENDPOINT={working_endpoints[0]}")
        if len(working_endpoints) > 1:
            print(f"NEMO_DOCKER_ENDPOINT_FALLBACK={working_endpoints[1]}")
            
    return bool(working_endpoints)

async def main():
    parser = argparse.ArgumentParser(description="Test connection to Docker STT service")
    parser.add_argument("--url", type=str, help="WebSocket URL to test")
    parser.add_argument("--scan", action="store_true", help="Scan multiple common endpoints")
    parser.add_argument("--ignore-ssl", action="store_true", help="Ignore SSL verification for wss:// URLs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.url:
        await test_connection(args.url, ignore_ssl=args.ignore_ssl, verbose=args.verbose)
    elif args.scan or not args.url:
        await test_multiple_endpoints()
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
