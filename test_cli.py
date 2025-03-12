#!/usr/bin/env python3

import asyncio
from lucidia_cli import LucidiaReflectionClient

async def test_cli():
    client = LucidiaReflectionClient()
    
    # Test connection
    print("Connecting to services...")
    connected = await client.connect()
    print(f"Connection status: {'Connected' if connected else 'Failed to connect'}")
    
    if connected:
        try:
            # Test parameter endpoints
            print("\nTesting parameter endpoints...")
            params = await client.get_parameters()
            if params:
                print(f"Successfully retrieved {len(params)} parameters")
                
                # Try getting a specific parameter
                first_param = next(iter(params.keys()))
                print(f"\nGetting parameter: {first_param}")
                param_data = await client.get_parameter(first_param)
                print(f"Parameter data: {param_data}")
            else:
                print("Failed to retrieve parameters")
            
            # Test system state
            print("\nFetching system state...")
            state = await client.fetch_system_state()
            print(f"System state fetch: {'Successful' if state else 'Failed'}")
            
        finally:
            # Disconnect
            await client.disconnect()
            print("\nDisconnected from services")

if __name__ == "__main__":
    asyncio.run(test_cli())
