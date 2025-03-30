#!/usr/bin/env python

"""
Runner for Titans integration tests.

This script executes integration tests for the Titans variants in the Lucidia system.
Note: Each variant must be tested separately by setting the TITANS_VARIANT environment
variable in the CCE Docker container before running this script.

Example for testing MAC variant:
   1. In your docker-compose.yml file, set environment for cce service:
      environment:
        - TITANS_VARIANT=MAC
   2. Restart the CCE container: docker-compose restart cce
   3. Run this script with: python run_titans_integration_tests.py MAC
"""

import asyncio
import logging
import os
import sys
from typing import Optional

from tests.integration.test_titans_integration import (
    test_base_variant,
    test_mac_variant, 
    test_mag_variant,
    test_mal_variant,
    test_all_variants_sequentially
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("titans_integration_tests.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Set service URLs
CCE_URL = os.environ.get('CCE_URL', 'http://localhost:8002')
MEMORY_CORE_URL = os.environ.get('MEMORY_CORE_URL', 'http://localhost:5010')  
NEURAL_MEMORY_URL = os.environ.get('NEURAL_MEMORY_URL', 'http://localhost:8001')

# Export URLs as environment variables for tests
os.environ['CCE_URL'] = CCE_URL
os.environ['MEMORY_CORE_URL'] = MEMORY_CORE_URL
os.environ['NEURAL_MEMORY_URL'] = NEURAL_MEMORY_URL

async def run_selected_tests(variant: Optional[str] = None):
    """Run selected tests for a specific variant.
    
    Args:
        variant: The variant to test ('NONE', 'MAC', 'MAG', 'MAL', or None for all)
    """
    try:
        variant = (variant or 'ALL').upper()
        print("="*75)
        print("TITANS INTEGRATION TESTS".center(75))
        print("="*75)
        print(f"Testing variant(s): {variant}")
        print("Environment configuration:")
        print(f"  CCE_URL: {CCE_URL}")
        print(f"  MEMORY_CORE_URL: {MEMORY_CORE_URL}")
        print(f"  NEURAL_MEMORY_URL: {NEURAL_MEMORY_URL}")
        print()
        
        print("⚠️  IMPORTANT: Ensure Docker services are running with correct variant")
        print("   When testing a specific variant, make sure the CCE container")
        print("   has the TITANS_VARIANT environment variable set accordingly.")
        print()
        
        if variant == 'ALL':
            logger.warning(
                "Testing ALL variants sequentially is not recommended. "
                "The CCE container should be restarted with the appropriate "
                "TITANS_VARIANT environment variable between tests."
            )
            await test_all_variants_sequentially()
        elif variant == 'NONE':
            print("TESTING NONE VARIANT (BASELINE)".center(75, '='))
            await test_base_variant()
            print("\n✅ NONE variant test completed successfully\n")
        elif variant == 'MAC':
            print("TESTING MAC VARIANT (MEMORY-ATTENDED COMPUTATION)".center(75, '='))
            await test_mac_variant()
            print("\n✅ MAC variant test completed successfully\n")
        elif variant == 'MAG':
            print("TESTING MAG VARIANT (MEMORY-ATTENDED GATES)".center(75, '='))
            await test_mag_variant()
            print("\n✅ MAG variant test completed successfully\n")
        elif variant == 'MAL':
            print("TESTING MAL VARIANT (MEMORY-AUGMENTED LEARNING)".center(75, '='))
            await test_mal_variant()
            print("\n✅ MAL variant test completed successfully\n")
        else:
            logger.error(f"Unknown variant: {variant}")
            print(f"❌ Unknown variant: {variant}. Must be one of: ALL, NONE, MAC, MAG, MAL")
    except Exception as e:
        logger.exception(f"Error running tests: {e}")
        print(f"\n❌ Tests failed with error: {e}")
        print(f"📋 Check the log file for details: titans_integration_tests.log")


if __name__ == "__main__":
    # Get the variant to test from command-line argument
    variant_to_test = sys.argv[1] if len(sys.argv) > 1 else 'ALL'
    
    asyncio.run(run_selected_tests(variant_to_test))
