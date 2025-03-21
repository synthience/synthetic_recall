import asyncio
import numpy as np
import torch
import traceback
import logging
import sys
from server.qr_calculator import UnifiedQuickRecallCalculator

# Configure logger with a StreamHandler that uses sys.stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clear any existing handlers
for handler in logger.handlers:
    logger.removeHandler(handler)

# Add our custom handler
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

async def test_dimension_mismatch():
    print("\n" + "=" * 80)
    print("TESTING DIMENSION MISMATCH HANDLING")
    print("=" * 80 + "\n")
    
    # Create calculator instance
    print("\n=== INITIALIZING CALCULATOR ===\n")
    calc = UnifiedQuickRecallCalculator()
    
    # Create embeddings with different dimensions
    print("\n=== CREATING TEST EMBEDDINGS ===\n")
    emb1 = np.random.random(384)  # 384-dim embedding
    emb2 = np.random.random(768)  # 768-dim embedding
    
    print(f"Embedding 1: {type(emb1)} with shape {emb1.shape}")
    print(f"Embedding 2: {type(emb2)} with shape {emb2.shape}\n")
    
    # Set momentum buffer with mixed dimensions
    momentum_data = [emb1, emb2]
    print(f"Setting momentum buffer with {len(momentum_data)} embeddings of different dimensions")
    calc.set_external_momentum(momentum_data)
    
    print("\n" + "=" * 80)
    print("TESTING QR CALCULATOR WITH MIXED DIMENSION EMBEDDINGS")
    print("=" * 80 + "\n")
    
    # First, test the vector alignment utility directly
    try:
        print("\n=== TESTING VECTOR ALIGNMENT UTILITY ===\n")
        aligned_1, aligned_2, common_dim = calc._align_vectors_for_comparison(emb1, emb2)
        print(f"Original shapes: {emb1.shape}, {emb2.shape}")
        print(f"Aligned shapes: {aligned_1.shape}, {aligned_2.shape}")
        print(f"Common dimension: {common_dim}")
        print("Vector alignment: SUCCESS")
    except Exception as e:
        print(f"Error in vector alignment test: {e}")
        traceback.print_exc()
    
    tests = [
        ("overlap", calc._calculate_overlap),
        ("surprise", calc._calculate_surprise),
        ("diversity", calc._calculate_diversity),
        ("r_geometry", calc._calculate_r_geometry),
        ("causal_novelty", calc._calculate_causal_novelty),
        ("self_organization", calc._calculate_self_organization)
    ]
    
    print("\n=== TESTING INDIVIDUAL CALCULATION METHODS ===\n")
    for name, func in tests:
        try:
            print(f"Testing {name} calculation...")
            result1 = await func(emb1, {})
            result2 = await func(emb2, {})
            print(f"{name} score for 384-dim embedding: {result1}")
            print(f"{name} score for 768-dim embedding: {result2}")
            print(f"{name} calculation: SUCCESS\n")
        except Exception as e:
            print(f"Error in {name} calculation: {e}")
            traceback.print_exc()
            print(f"{name} calculation: FAILED\n")
    
    try:
        # Test a full HPC-QR calculation
        print("\n=== TESTING COMPLETE HPC-QR CALCULATION ===\n")
        calc.set_mode("hpc_qr")
        result1 = await calc.calculate(emb1)
        result2 = await calc.calculate(emb2)
        print(f"Complete QR score for 384-dim embedding: {result1}")
        print(f"Complete QR score for 768-dim embedding: {result2}")
        print("Complete HPC-QR calculation: SUCCESS")
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 80 + "\n")
    except Exception as e:
        print(f"Error in complete calculation: {e}")
        traceback.print_exc()
        print("Complete HPC-QR calculation: FAILED")

async def test_align_vectors_stress():
    print("\n" + "=" * 80)
    print("STRESS TESTING VECTOR ALIGNMENT LOGGING")
    print("=" * 80 + "\n")
    
    # Create calculator instance
    calc = UnifiedQuickRecallCalculator()
    
    # Create embeddings with different dimensions
    emb1 = np.random.random(384)  # 384-dim embedding
    emb2 = np.random.random(768)  # 768-dim embedding
    
    # Perform many alignments to test logging optimization
    print(f"Performing 50 vector alignments with dimension mismatch")
    for i in range(50):
        _, _, _ = calc._align_vectors_for_comparison(emb1, emb2)
    
    print(f"Total warning logs issued: {calc.dim_mismatch_warnings}")
    print(f"Warning limit reached: {calc.dim_mismatch_logged}")
    print("\nStress test complete - verify that warning logs were capped properly")

if __name__ == "__main__":
    # Run the async tests
    asyncio.run(test_dimension_mismatch())
    asyncio.run(test_align_vectors_stress())
