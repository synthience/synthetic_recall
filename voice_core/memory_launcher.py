"""Script to launch the memory agent."""

import asyncio
import argparse
import logging
import os
import sys

from voice_core.agent3 import LucidiaMemoryAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory_launcher")

async def run_memory_agent(config_path=None):
    """Run the memory agent with the given configuration."""
    memory_agent = None
    try:
        # Configure the memory agent
        tensor_server_url = os.environ.get("TENSOR_SERVER_URL", "ws://localhost:5001")
        hpc_server_url = os.environ.get("HPC_SERVER_URL", "ws://localhost:5005")
        
        # Create and start the agent
        memory_agent = LucidiaMemoryAgent(
            tensor_server_url=tensor_server_url,
            hpc_server_url=hpc_server_url,
            config_path=config_path
        )
        
        await memory_agent.start()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Memory agent interrupted")
    finally:
        if memory_agent:
            await memory_agent.stop()

def main():
    """Parse command line arguments and run the memory agent."""
    parser = argparse.ArgumentParser(description="Lucidia Memory Agent")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    asyncio.run(run_memory_agent(args.config))

if __name__ == "__main__":
    main()
