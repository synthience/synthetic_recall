# examples/personal_details_demo.py

import asyncio
import logging
from memory_core.enhanced_memory_client import EnhancedMemoryClient

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def main():
    # Create memory client
    client = EnhancedMemoryClient(
        tensor_server_url="ws://localhost:8765",
        hpc_server_url="ws://localhost:8766",
        session_id="demo_session",
        user_id="demo_user"
    )
    
    # Example messages with personal details
    messages = [
        "Hi, my name is Alex Johnson.",
        "I live in Seattle, Washington.",
        "I work as a software engineer.",
        "My wife is Sarah and we have two children.",
        "Lucidia, can you remember my details?",  # Should not detect Lucidia as a name
        "My name is not Bob.",  # Should handle negation
    ]
    
    print("\n=== Personal Details Detection Demo ===\n")
    
    # Process each message
    for i, message in enumerate(messages):
        print(f"\nMessage {i+1}: {message}")
        
        # Detect and store personal details
        details_found = await client.detect_and_store_personal_details(message)
        
        if details_found:
            print("✓ Personal details detected!")
        else:
            print("✗ No personal details detected")
    
    # Display all stored personal details
    print("\n=== Stored Personal Details ===\n")
    for category, detail in client.personal_details.items():
        print(f"{category.capitalize()}: {detail['value']} (significance: {detail['significance']})")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
