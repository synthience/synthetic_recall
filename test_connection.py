"""
Simple script to test the connection to the Memory Core API.
"""
import asyncio
import aiohttp
import sys

async def test_connection(url="http://localhost:5010"):
    print(f"Testing connection to {url}...")
    
    # Test basic connectivity
    try:
        async with aiohttp.ClientSession() as session:
            # Test the root endpoint first
            print("Testing root endpoint...")
            async with session.get(f"{url}/") as response:
                if response.status == 200:
                    print("✅ Root endpoint is accessible")
                    data = await response.json()
                    print(f"  Response: {data}")
                else:
                    print(f"❌ Root endpoint returned status {response.status}")
                    return False
            
            # Test the health endpoint
            print("\nTesting health endpoint...")
            async with session.get(f"{url}/health") as response:
                if response.status == 200:
                    print("✅ Health endpoint is accessible")
                    data = await response.json()
                    print(f"  Status: {data.get('status')}")
                    print(f"  Memory count: {data.get('memory_count')}")
                    print(f"  Vector index state: {data.get('vector_index_state')}")
                    print(f"  Storage path: {data.get('storage_path')}")
                else:
                    print(f"❌ Health endpoint returned status {response.status}")
                    return False
            
            # Test the stats endpoint
            print("\nTesting stats endpoint...")
            async with session.get(f"{url}/stats") as response:
                if response.status == 200:
                    print("✅ Stats endpoint is accessible")
                    data = await response.json()
                    # Pretty print some key stats
                    if data.get("success"):
                        memory_stats = data.get("memory", {})
                        vector_stats = data.get("vector_index", {})
                        print(f"  Total memories: {memory_stats.get('total_memories')}")
                        print(f"  Storage path: {memory_stats.get('storage_path')}")
                        print(f"  Vector index count: {vector_stats.get('count')}")
                    else:
                        print(f"  Error: {data.get('error')}")
                else:
                    print(f"❌ Stats endpoint returned status {response.status}")
                    return False
                    
            return True
            
    except aiohttp.ClientConnectorError as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5010"
    success = asyncio.run(test_connection(url))
    if success:
        print("\n✅ All tests passed! The server is accessible.")
    else:
        print("\n❌ Some tests failed. The server might not be accessible.")
        sys.exit(1)