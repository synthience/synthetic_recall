#!/usr/bin/env python3
import socket
import sys

def check_port(host, port):
    """Check if a port is open on a given host"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)  # 2 second timeout
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <host> <port>")
        sys.exit(1)
    
    host = sys.argv[1]
    port = int(sys.argv[2])
    
    if check_port(host, port):
        print(f"Port {port} is open on {host}")
    else:
        print(f"Port {port} is closed on {host}")
