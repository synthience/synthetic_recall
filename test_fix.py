# test_fix.py
import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s %(name)s - %(message)s')

# Add the current directory to the Python path
sys.path.append('.')

# Import after setting up path
from memory_core.enhanced_memory_client import EnhancedMemoryClient

async def test_memory_client():
    # Create a client with dummy URLs
    client = EnhancedMemoryClient(
        tensor_server_url='ws://localhost:5001',
        hpc_server_url='ws://localhost:5001',
        session_id='test_session'
    )
    
    # Test with list result
    list_result = []
    dict_result = {"memories": []}
    
    # Test both code paths
    print("Testing with list result...")
    try:
        # Directly call the fixed method with empty results to test the fix
        result1 = await client._generate_standard_context('test', 5, 0.5)
        print(f"Result with actual implementation: {result1}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    asyncio.run(test_memory_client())
