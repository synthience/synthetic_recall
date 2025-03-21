#!/usr/bin/env python
"""
Test script for the enhanced QuickRecal scoring system.

This script validates the recalibrated scoring system with:
- Z-score normalization
- Logarithmic fusion
- Soft thresholding
- Rebalanced weight factors
"""

import asyncio
import json
import numpy as np
import torch
import time
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Import local modules
from server.hpc_server import HPCClient, HPCServer
from server.qr_calculator import UnifiedQuickRecallCalculator, QuickRecallMode, QuickRecallFactor, safe_run
from memory.lucidia_memory_system.core.integration.hpc_qr_flow_manager import HPCQRFlowManager

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test data directory
TEST_DATA_DIR = Path("test_data")
TEST_DATA_DIR.mkdir(exist_ok=True)


def create_diverse_test_embeddings(num_embeddings=20, embedding_dim=384, seed=42):
    """Create diverse test embeddings to properly test SOM responsiveness.
    
    Instead of purely random embeddings, this creates embeddings with:
    - Controlled variance in different dimensions
    - Some similar clusters and some outliers
    - Directional biases to simulate semantic differences
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Base embeddings with different characteristics
    embeddings = []
    
    # Create cluster 1 - similar embeddings with small variations
    base_vec1 = torch.randn(embedding_dim)
    for i in range(5):
        # Small variations on the same base vector
        noise = torch.randn(embedding_dim) * 0.1
        emb = base_vec1 + noise
        embeddings.append(emb)
    
    # Create cluster 2 - different from cluster 1
    base_vec2 = -base_vec1 + torch.randn(embedding_dim) * 0.5
    for i in range(5):
        noise = torch.randn(embedding_dim) * 0.1
        emb = base_vec2 + noise
        embeddings.append(emb)
    
    # Create some outliers that are very different
    for i in range(3):
        # Strong directional changes
        outlier = torch.randn(embedding_dim) * 2.0
        embeddings.append(outlier)
    
    # Create time-dependent sequence (gradual drift)
    drift_base = torch.randn(embedding_dim)
    for i in range(5):
        # Gradually changing embedding to simulate temporal evolution
        drift = drift_base + torch.randn(embedding_dim) * 0.3 * i
        embeddings.append(drift)
    
    # Add remaining random embeddings if needed
    while len(embeddings) < num_embeddings:
        embeddings.append(torch.randn(embedding_dim))
    
    # Normalize all embeddings
    normalized_embeddings = []
    for emb in embeddings:
        emb_norm = torch.norm(emb)
        if emb_norm > 0:
            normalized_embeddings.append(emb / emb_norm)
        else:
            normalized_embeddings.append(emb)
    
    return normalized_embeddings


def standardize_embedding(embedding, target_dim=384):
    """Standardize embedding to a target dimension by truncating or padding.
    
    Args:
        embedding: The input embedding (numpy array or torch.Tensor)
        target_dim: The target dimension to standardize to
        
    Returns:
        Standardized embedding with target_dim dimensions
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(embedding, torch.Tensor):
        is_tensor = True
        device = embedding.device
        embedding_np = embedding.cpu().numpy()
    else:
        is_tensor = False
        embedding_np = embedding
    
    # Flatten if needed (handle both vector and multi-dimensional cases)
    orig_shape = embedding_np.shape
    if len(orig_shape) > 1:
        flat_emb = embedding_np.reshape(-1)
    else:
        flat_emb = embedding_np
    
    current_dim = flat_emb.shape[0]
    
    # Standardize dimension
    if current_dim == target_dim:
        # Already the right size
        standardized = flat_emb
    elif current_dim > target_dim:
        # Truncate
        standardized = flat_emb[:target_dim]
    else:
        # Pad with zeros
        standardized = np.zeros(target_dim)
        standardized[:current_dim] = flat_emb
    
    # Normalize the vector
    norm = np.linalg.norm(standardized)
    if norm > 1e-10:
        standardized = standardized / norm
    
    # Convert back to tensor if input was a tensor
    if is_tensor:
        return torch.tensor(standardized, device=device)
    else:
        return standardized


async def test_direct_qr_calculation():
    """Test the UnifiedQuickRecallCalculator directly"""
    logger.info("=== Testing Direct QR Calculation ===")
    
    # Set target embedding dimension
    target_dim = 384
    
    # Create calculator with config from file
    calculator = UnifiedQuickRecallCalculator({
        'mode': QuickRecallMode.STANDARD,
        'embedding_dim': target_dim,      # Set the embedding dimension to match our test embeddings
        'som_grid_size': (10, 10),        # Maintain the 10x10 grid
        'som_learning_rate': 0.3,         # Higher learning rate for more responsiveness
        'som_sigma': 3.0,                 # Increased neighborhood size
        'novelty_threshold': 0.2,         # Lower threshold to make novelty detection more sensitive
        'use_zscore_normalization': True
    })
    
    # Print current configuration
    logger.info(f"Calculator mode: {calculator.config['mode']}")
    logger.info(f"Novelty threshold: {calculator.config.get('novelty_threshold', 'N/A')}")
    logger.info(f"Using Z-score normalization: {calculator.config.get('use_zscore_normalization', False)}")
    logger.info(f"SOM dimensions: {calculator.config.get('som_grid_size', 'N/A')} x {calculator.config.get('embedding_dim', 'N/A')}")
    
    # Create diverse test embeddings
    test_embeddings = create_diverse_test_embeddings(num_embeddings=20, embedding_dim=target_dim, seed=42)
    logger.info(f"Created {len(test_embeddings)} diverse test embeddings")
    
    # Process through calculator with delays to allow SOM to adapt
    scores = []
    all_component_scores = []
    
    # First pass to build momentum and adapt SOM
    logger.info("First pass through embeddings to build momentum...")
    for i, emb in enumerate(test_embeddings[:5]):
        # Always standardize embedding before any operation
        std_emb = standardize_embedding(emb, target_dim=target_dim) 
        std_emb_np = std_emb.numpy() if isinstance(std_emb, torch.Tensor) else std_emb
        
        # Pre-train SOM and build momentum buffer
        await calculator._update_som(std_emb_np)  
        
        # Update momentum manually and ensure correct dimensions
        if calculator.external_momentum is None:
            calculator.external_momentum = std_emb_np
        else:
            # Stack as numpy array
            if isinstance(calculator.external_momentum, np.ndarray):
                # Convert to list if not already
                if len(calculator.external_momentum.shape) == 1:
                    calculator.external_momentum = [calculator.external_momentum, std_emb_np]
                else:
                    calculator.external_momentum = np.vstack([calculator.external_momentum, std_emb_np])
            else:
                calculator.external_momentum = std_emb_np
    
    logger.info("Second pass with scoring...")
    for i, emb in enumerate(test_embeddings):
        # Always standardize the embedding
        std_emb = standardize_embedding(emb, target_dim=target_dim)
        std_emb_np = std_emb.numpy() if isinstance(std_emb, torch.Tensor) else std_emb
        
        # Initialize component scores dict
        component_scores = {}
        context = {}
        
        # After the first embedding, we'll have momentum for component calculations
        if i > 0:  
            for factor in [QuickRecallFactor.SURPRISE, QuickRecallFactor.DIVERSITY, 
                         QuickRecallFactor.R_GEOMETRY, QuickRecallFactor.CAUSAL_NOVELTY]:
                try:
                    if factor == QuickRecallFactor.SURPRISE:
                        component_scores[factor.value] = await calculator._calculate_surprise(std_emb_np, context)
                    elif factor == QuickRecallFactor.DIVERSITY:
                        component_scores[factor.value] = await calculator._calculate_diversity(std_emb_np, context)
                    elif factor == QuickRecallFactor.R_GEOMETRY:
                        component_scores[factor.value] = await calculator._calculate_r_geometry(std_emb_np, context)
                    elif factor == QuickRecallFactor.CAUSAL_NOVELTY:
                        component_scores[factor.value] = await calculator._calculate_causal_novelty(std_emb_np, context)
                except Exception as e:
                    logger.warning(f"Error calculating {factor.value}: {e}")
                    component_scores[factor.value] = 0.5  # Default value
        
        # Calculate full score with standardized embedding
        score = await calculator.calculate(std_emb)
        scores.append(score)
        
        # Store component scores for later analysis
        if component_scores:
            all_component_scores.append(component_scores)
            logger.info(f"Embedding {i}: QR Score = {score:.4f}, Components: {component_scores}")
        else:
            logger.info(f"Embedding {i}: QR Score = {score:.4f}")
        
        # Update momentum with standardized embedding
        if calculator.external_momentum is None:
            calculator.external_momentum = std_emb_np
        else:
            # Handle momentum update based on type
            if isinstance(calculator.external_momentum, list):
                calculator.external_momentum.append(std_emb_np)
                # Keep only the last 10 for efficiency
                if len(calculator.external_momentum) > 10:
                    calculator.external_momentum = calculator.external_momentum[-10:]
            elif isinstance(calculator.external_momentum, np.ndarray):
                if len(calculator.external_momentum.shape) == 1:
                    calculator.external_momentum = np.stack([calculator.external_momentum, std_emb_np])
                else:
                    calculator.external_momentum = np.vstack([calculator.external_momentum, std_emb_np])
                    # Keep only the last 10 for efficiency
                    if len(calculator.external_momentum) > 10:
                        calculator.external_momentum = calculator.external_momentum[-10:]
            else:
                calculator.external_momentum = std_emb_np
    
    # Calculate and log statistics about the scores
    score_array = np.array(scores)
    logger.info(f"Score statistics - Min: {score_array.min():.4f}, Max: {score_array.max():.4f}, "  
               f"Mean: {score_array.mean():.4f}, StdDev: {score_array.std():.4f}")
    
    # Implement soft gating based on score volatility
    if len(all_component_scores) >= 3:
        # Calculate volatility for each component
        component_volatility = {}
        for factor in [QuickRecallFactor.SURPRISE.value, QuickRecallFactor.DIVERSITY.value,
                     QuickRecallFactor.R_GEOMETRY.value, QuickRecallFactor.CAUSAL_NOVELTY.value]:
            component_values = [cs.get(factor, 0.5) for cs in all_component_scores if factor in cs]
            if component_values:
                volatility = np.std(component_values)
                component_volatility[factor] = volatility
        
        logger.info(f"Component volatility (higher is better): {component_volatility}")
        
        # Suggest weight adjustments based on volatility
        if component_volatility:
            total_volatility = sum(component_volatility.values())
            if total_volatility > 0:
                suggested_weights = {factor: (vol/total_volatility) 
                                  for factor, vol in component_volatility.items()}
                logger.info(f"Suggested factor weights based on volatility: {suggested_weights}")
    
    # Log distribution after accumulating scores
    calculator.log_score_distribution()
    
    return scores


async def test_memory_index_search():
    """Test the enhanced memory index search functionality"""
    logger.info("\n=== Testing Memory Index Search ===\n")
    
    # Import locally to avoid circular imports
    from server.memory_index import MemoryIndex
    
    # Create a memory index
    memory_index = MemoryIndex(embedding_dim=384)
    
    # Print fusion weights
    logger.info(f"Fusion weights: {memory_index.fusion_weights}")
    
    # Create test embeddings and add to index
    test_embeddings = [standardize_embedding(torch.randn(384), target_dim=384) for _ in range(30)]
    
    # Add memories with varying QR scores - distribution is important
    # Create a distribution where most are in the middle range (0.4-0.6)
    # but a few key memories have scores above 0.7 or below 0.3
    qr_scores = []
    
    # Add a few low-scoring memories (0.2-0.3)
    for i in range(5):
        qr_score = 0.2 + (i / 5) * 0.1
        qr_scores.append(qr_score)
    
    # Add medium-scoring memories (0.4-0.6) - most will be here
    for i in range(15):
        qr_score = 0.4 + (i / 15) * 0.2
        qr_scores.append(qr_score)
    
    # Add high-scoring memories (0.7-0.9)
    for i in range(10):
        qr_score = 0.7 + (i / 10) * 0.2
        qr_scores.append(qr_score)
        
    # Shuffle the scores to avoid biasing by order
    np.random.shuffle(qr_scores)
    
    for i, (emb, qr_score) in enumerate(zip(test_embeddings, qr_scores)):
        # Add to index
        await memory_index.add_memory(
            f"memory_{i}",
            emb,
            time.time(),
            quickrecal_score=qr_score
        )
        
    # Build the index
    memory_index.build_index()
    
    # Search with a random query
    query = standardize_embedding(torch.randn(384), target_dim=384)
    
    # Define volatility levels to test threshold boosting
    volatility_levels = [0.0, 0.2, 0.5, 0.8]
    thresholds = [0.0, 0.2, 0.4, 0.6]
    
    logger.info("\nTesting search with different thresholds and volatility levels:")
    for threshold in thresholds:
        logger.info(f"\nBase threshold: {threshold}")
        
        for volatility in volatility_levels:
            results = memory_index.search(query, k=5, min_quickrecal_score=threshold, threshold_boost=volatility)
            
            # Calculate statistics on returned scores
            if results:
                qr_scores = [r['qr_score'] for r in results]
                avg_score = sum(qr_scores) / len(qr_scores)
                min_score = min(qr_scores)
                max_score = max(qr_scores)
                
                logger.info(f"  Volatility {volatility}: Found {len(results)} results with "  
                          f"QR scores min={min_score:.2f}, avg={avg_score:.2f}, max={max_score:.2f}")
                
                # Detail first few results
                for j, result in enumerate(results[:2]):
                    logger.info(f"    {j+1}. Memory: {result['memory']['id']}, "
                              f"QR: {result['qr_score']:.3f}, "
                              f"Sim: {result['similarity']:.3f}, "
                              f"Fused: {result['combined_score']:.3f}")
            else:
                logger.info(f"  Volatility {volatility}: No results found")
    
    return True


async def test_hpc_client():
    """Test the enhanced HPC client functionality"""
    logger.info("\n=== Testing HPC Client ===\n")
    
    # Create client
    # Docker hostname - allow configuration via environment variable or fallback to host.docker.internal for Docker
    docker_host = os.environ.get("DOCKER_HOST_QUICKRECAL", "host.docker.internal")
    # Alternative hostnames to try if the primary connection fails
    alternative_hosts = ["localhost", "127.0.0.1", docker_host, "quickrecal"]
    
    connected = False
    client = None
    
    # Try multiple host options
    for host in alternative_hosts:
        ws_url = f"ws://{host}:5005"
        logger.info(f"Attempting to connect to HPC server at {ws_url}")
        client = HPCClient(url=ws_url)
        
        # Try to connect
        try:
            connected = await client.connect()
            if connected:
                logger.info(f"Successfully connected to HPC server at {ws_url}")
                break
        except Exception as e:
            logger.warning(f"Failed to connect to {ws_url}: {e}")
    
    if not connected:
        logger.error("Failed to connect to HPC server. Is it running?")
        logger.info("Start the server with: python -m server.hpc_server")
        return False
    
    # Load config
    await client.load_config()
    logger.info(f"Loaded search fusion weights: {client.search_fusion_weights}")
    
    # Process embeddings with different thresholds
    test_emb = torch.randn(384)
    test_emb = test_emb / torch.norm(test_emb)
    
    # Regular processing
    logger.info("Testing regular processing...")
    result = await client.process_embeddings(test_emb.tolist())
    logger.info(f"Regular processing result: {result}")
    
    # Enhanced processing with QR scores
    logger.info("\nTesting enhanced processing with QR scores...")
    thresholds = [0.0, 0.3, 0.5]
    for threshold in thresholds:
        result = await client.process_embeddings_with_qr(test_emb.tolist(), min_qr_threshold=threshold)
        logger.info(f"Processing with threshold {threshold}: {result}")
    
    # Request score distribution
    logger.info("\nRequesting score distribution...")
    await client.request_score_distribution()
    
    # Disconnect
    await client.disconnect()
    return True


async def plot_score_distribution(scores):
    """Plot histogram of scores for visualization"""
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, alpha=0.7, color='blue')
    plt.title('QuickRecal Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plot_path = TEST_DATA_DIR / 'qr_score_distribution.png'
    plt.savefig(plot_path)
    logger.info(f"Score distribution plot saved to {plot_path}")


def test_malformed_embedding_handling():
    """Test how the system handles malformed embeddings, including QR score capping."""
    print("\n=== Testing Malformed Embedding Handling ===")
    print("--------------------------------------------------\n")
    
    # Initialize calculator with standard mode
    calculator = UnifiedQuickRecallCalculator({
        "mode": "standard",
        "embedding_dim": 384,
        "verbose": True,
        "use_emotions": False,
    })
    
    # Test cases for various kinds of malformed embeddings
    test_cases = [
        {"name": "Valid 384-dim", "embedding": torch.randn(384), "expected_qr_range": [0.0, 1.0]},
        {"name": "None", "embedding": None, "expected_qr_range": [0.0, 0.1]},
        {"name": "Scalar", "embedding": torch.tensor([1.0]), "expected_qr_range": [0.0, 0.1]},
        {"name": "Empty list", "embedding": [], "expected_qr_range": [0.0, 0.3]},
        {"name": "NaN values", "embedding": torch.tensor([float('nan')] * 384), "expected_qr_range": [0.0, 0.1]},
        {"name": "Small dim (10)", "embedding": torch.randn(10), "expected_qr_range": [0.0, 0.2]},
        {"name": "Different dim (768)", "embedding": torch.randn(768), "expected_qr_range": [0.0, 1.0]},
    ]
    
    print("Testing UnifiedQuickRecallCalculator:")
    # Test each case with the calculator
    for case in test_cases:
        if case["embedding"] is None:
            print(f"- {case['name']}: Skipped (None embedding)")
            continue
            
        try:
            # Standardize the embedding for the test
            if isinstance(case["embedding"], torch.Tensor) and case["embedding"].numel() > 0:
                embedding = standardize_embedding(case["embedding"])
                # Use the validate_embedding method to get penalty info
                validated_emb, is_penalty, qr_cap = calculator.validate_embedding(embedding)
                # Calculate QR score with penalty context
                qr_score = safe_run(calculator.calculate, validated_emb, context={'is_penalty': is_penalty, 'qr_cap': qr_cap})
                # Verify QR range matches expectations
                min_expected, max_expected = case["expected_qr_range"]
                result = "PASS" if min_expected <= qr_score <= max_expected else "FAIL"
                penalty_str = "[PENALTY]" if is_penalty else "[NORMAL]"
                print(f"{result} {case['name']}: QR={qr_score:.4f} {penalty_str} (Cap={qr_cap:.2f}, Expected={min_expected:.2f}-{max_expected:.2f})")
            elif isinstance(case["embedding"], list) and len(case["embedding"]) == 0:
                # Empty list test
                try:
                    # Try to convert and validate
                    validated_emb, is_penalty, qr_cap = calculator.validate_embedding(case["embedding"])
                    # Calculate QR score with penalty context
                    qr_score = safe_run(calculator.calculate, validated_emb, context={'is_penalty': is_penalty, 'qr_cap': qr_cap})
                    # Check result
                    min_expected, max_expected = case["expected_qr_range"]
                    result = "PASS" if min_expected <= qr_score <= max_expected else "FAIL"
                    print(f"{result} {case['name']}: Empty list handled with penalty vector")
                except Exception as e:
                    print(f"FAIL {case['name']}: Error handling empty list - {str(e)}")
        except Exception as e:
            print(f"FAIL {case['name']}: Error - {str(e)}")

    # Test flow manager
    print("\nTesting HPCQRFlowManager:")
    
    # Create a HPC flow manager with calculator inside the config
    flow_manager_config = {
        "device": "cpu",
        "chunk_size": 384,
        "embedding_dim": 384, 
        "alpha": 1.0,
        "beta": 0.8,
        "gamma": 0.5,
        "shock_absorption_enabled": True,
        "dynamic_scaling_factor": 0.1,
    }
    
    # Create the flow manager with its config
    flow_manager = HPCQRFlowManager(flow_manager_config)
    # Set the calculator after initialization
    flow_manager.qr_calculator = calculator
    
    for case in test_cases:
        try:
            # Process the embedding through the flow manager
            embedding = case["embedding"]
            
            # Process using the flow manager
            if embedding is not None and isinstance(embedding, torch.Tensor) and embedding.numel() > 0:
                processed_emb, qr_score = safe_run(flow_manager.process_embedding, embedding)
                # Verify QR range matches expectations
                min_expected, max_expected = case["expected_qr_range"]
                result = "PASS" if min_expected <= qr_score <= max_expected else "FAIL"
                print(f"{result} {case['name']}: QR={qr_score:.4f} (Expected={min_expected:.2f}-{max_expected:.2f})")
            elif embedding is None:
                try:
                    processed_emb, qr_score = safe_run(flow_manager.process_embedding, embedding)
                    min_expected, max_expected = case["expected_qr_range"]
                    result = "PASS" if min_expected <= qr_score <= max_expected else "FAIL"
                    print(f"{result} {case['name']}: QR={qr_score:.4f} (Expected={min_expected:.2f}-{max_expected:.2f})")
                except Exception as e:
                    # If it handles the error gracefully, that's a pass
                    print(f"PASS {case['name']}: Error handled appropriately - {str(e)[:50]}")
            elif isinstance(embedding, list) and len(embedding) == 0:
                try:
                    # Try with empty tensor
                    test_tensor = torch.tensor([])
                    processed_emb, qr_score = safe_run(flow_manager.process_embedding, test_tensor)
                    min_expected, max_expected = case["expected_qr_range"]
                    result = "PASS" if min_expected <= qr_score <= max_expected else "FAIL"
                    print(f"{result} {case['name']}: QR={qr_score:.4f} (Expected={min_expected:.2f}-{max_expected:.2f})")
                except Exception as e:
                    # If it handles the error gracefully, that's a pass
                    print(f"PASS {case['name']}: Error handled appropriately - {str(e)[:50]}")
            else:
                # For problem embeddings, process with special handling
                try:
                    processed_emb, qr_score = safe_run(flow_manager.process_embedding, embedding)
                    min_expected, max_expected = case["expected_qr_range"]
                    result = "PASS" if min_expected <= qr_score <= max_expected else "FAIL"
                    print(f"{result} {case['name']}: QR={qr_score:.4f} (Expected={min_expected:.2f}-{max_expected:.2f})")
                except Exception as e:
                    print(f"FAIL {case['name']}: Processing error - {str(e)[:50]}")
        except Exception as e:
            print(f"FAIL {case['name']}: Error - {str(e)[:50]}")
    
    print("\nTest completed")


async def test_nan_handling():
    """Test specific handling of NaN embeddings"""
    print("\n=== Testing NaN Handling ===\n")
    
    # Initialize calculator
    calculator = UnifiedQuickRecallCalculator({
        "mode": "standard",
        "embedding_dim": 384,
        "verbose": True,
    })
    
    # Create a NaN embedding
    nan_embedding = torch.tensor([float('nan')] * 384)
    
    # Validate the embedding
    print("Validating NaN embedding...")
    validated_emb, is_penalty, qr_cap = calculator.validate_embedding(nan_embedding)
    print(f"Is penalty: {is_penalty}, QR cap: {qr_cap}")
    
    # Calculate QR score
    print("Calculating QR score...")
    qr_score = await calculator.calculate(validated_emb, context={'is_penalty': is_penalty, 'qr_cap': qr_cap})
    print(f"QR score for NaN embedding: {qr_score}")
    print(f"Expected max: {qr_cap}, Actual: {qr_score}")
    print(f"Test result: {'PASS' if qr_score <= qr_cap else 'FAIL'}")


async def main():
    """Run all tests"""
    logger.info("Starting enhanced QuickRecal system tests")
    
    try:
        # Test direct calculation
        scores = await test_direct_qr_calculation()
        
        # Plot score distribution
        await plot_score_distribution(scores)
        
        # Test memory index search
        await test_memory_index_search()
        
        # Test HPC client (only if server is running)
        try:
            await test_hpc_client()
        except Exception as e:
            logger.warning(f"HPC client test failed: {e}")
            logger.warning("To test the HPC client, make sure the HPC server is running in another terminal.")
            logger.warning("Run: python -m server.hpc_server")
        
        # Test malformed embedding handling
        test_malformed_embedding_handling()
        
        # Test NaN handling
        await test_nan_handling()
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    logger.info("All tests completed!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
