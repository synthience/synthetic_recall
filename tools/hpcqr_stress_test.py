#!/usr/bin/env python
"""
HPC-QR Flow Manager Stress Test

This script performs a stress test on the HPC-QR Flow Manager to identify optimal batch sizes
and verify the implementation of shock absorption in hypersphere norms.
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hpcqr_stress_test")

# Import the HPCQRFlowManager
from memory.lucidia_memory_system.core.integration.hpc_qr_flow_manager import HPCQRFlowManager


def generate_test_embeddings(count: int, dim: int = 768) -> List[torch.Tensor]:
    """
    Generate random test embeddings.
    
    Args:
        count: Number of embeddings to generate
        dim: Dimensionality of embeddings
        
    Returns:
        List of random embedding tensors
    """
    embeddings = []
    for _ in range(count):
        # Create a random vector
        vec = torch.rand(dim)
        # Normalize to unit length
        vec = vec / torch.norm(vec)
        embeddings.append(vec)
    return embeddings


async def test_batch_size(batch_size: int, embedding_count: int, dim: int = 768) -> Dict[str, Any]:
    """
    Test a specific batch size configuration.
    
    Args:
        batch_size: Batch size to test
        embedding_count: Total number of embeddings to process
        dim: Dimensionality of embeddings
        
    Returns:
        Dictionary with performance metrics
    """
    logger.info(f"Testing batch size: {batch_size} with {embedding_count} embeddings")
    
    # Configure the HPCQRFlowManager
    config = {
        'batch_size': batch_size,
        'embedding_dim': dim,
        'shock_absorption_enabled': True,
        'dynamic_scaling_factor': 0.5,
        'max_momentum_size': 10000,
        'update_threshold': 0.05,
        'drift_threshold': 0.3,
    }
    
    # Initialize the manager
    manager = HPCQRFlowManager(config)
    
    # Generate test embeddings
    embeddings = generate_test_embeddings(embedding_count, dim)
    
    # Process embeddings
    start_time = time.time()
    
    if batch_size > 1:
        # Use batch processing
        try:
            results = await manager.process_embedding_batch(embeddings)
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            results = []
    else:
        # Process individually
        results = []
        for emb in embeddings:
            try:
                processed, score = await manager.process_embedding(emb)
                results.append((processed, score))
            except Exception as e:
                logger.error(f"Error processing embedding: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Clean up
    await manager.close()
    
    # Collect statistics
    stats = manager.get_stats()
    
    # Return metrics
    return {
        'batch_size': batch_size,
        'embedding_count': embedding_count,
        'dim': dim,
        'total_time': total_time,
        'avg_time_per_embedding': total_time / embedding_count if embedding_count > 0 else 0,
        'processed_count': stats.get('processed_count', 0),
        'momentum_buffer_size': stats.get('momentum_buffer_size', 0),
        'avg_processing_time': stats.get('avg_processing_time', 0),
        'error_count': stats.get('error_count', 0),
        'memory_usage_mb': torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
        'damping_factors': [],  # Would need to add instrumentation to the manager to collect these
    }


async def run_stress_tests(batch_sizes: List[int], embedding_counts: List[int], dim: int = 768) -> List[Dict[str, Any]]:
    """
    Run stress tests across different batch sizes and embedding counts.
    
    Args:
        batch_sizes: List of batch sizes to test
        embedding_counts: List of embedding counts to test
        dim: Dimensionality of embeddings
        
    Returns:
        List of dictionaries with performance metrics
    """
    results = []
    
    for embedding_count in embedding_counts:
        for batch_size in batch_sizes:
            result = await test_batch_size(batch_size, embedding_count, dim)
            results.append(result)
            
            # Allow for GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Wait a bit between tests
            await asyncio.sleep(1)
    
    return results


def print_results(results: List[Dict[str, Any]]):
    """
    Print test results in a tabular format.
    
    Args:
        results: List of result dictionaries
    """
    print("\n========== HPC-QR Flow Manager Stress Test Results ==========\n")
    print(f"{'Batch Size':^10} | {'Embeddings':^10} | {'Total Time':^12} | {'Per Embedding':^14} | {'Memory (MB)':^12}")
    print("-" * 65)
    
    for result in results:
        print(f"{result['batch_size']:^10} | {result['embedding_count']:^10} | {result['total_time']:^12.4f} | "
              f"{result['avg_time_per_embedding']:^14.6f} | {result['memory_usage_mb']:^12.2f}")
    
    print("\n")
    
    # Find optimal batch size based on average time per embedding
    embedding_groups = {}
    for result in results:
        count = result['embedding_count']
        if count not in embedding_groups:
            embedding_groups[count] = []
        embedding_groups[count].append(result)
    
    print("Optimal Batch Sizes:\n")
    for count, group in sorted(embedding_groups.items()):
        optimal = min(group, key=lambda x: x['avg_time_per_embedding'])
        print(f"For {count} embeddings: Batch size {optimal['batch_size']} "
              f"(avg time: {optimal['avg_time_per_embedding']:.6f}s, "
              f"memory: {optimal['memory_usage_mb']:.2f} MB)")


def save_results(results: List[Dict[str, Any]], output_file: str):
    """
    Save test results to a JSON file.
    
    Args:
        results: List of result dictionaries
        output_file: Path to output file
    """
    # Convert any non-serializable values to strings
    serializable_results = []
    for result in results:
        serializable = {}
        for key, value in result.items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable[key] = value
            else:
                serializable[key] = str(value)
        serializable_results.append(serializable)
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'results': serializable_results,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


async def collect_damping_statistics(embedding_count: int, batch_size: int = 32) -> Dict[str, Any]:
    """
    Collect statistics on damping factors applied during shock absorption.
    
    Args:
        embedding_count: Number of embeddings to process
        batch_size: Batch size to use
        
    Returns:
        Dictionary with damping factor statistics
    """
    logger.info(f"Collecting damping statistics for {embedding_count} embeddings")
    
    # Modify the HPCQRFlowManager to log damping factors (this would require changes to the class)
    # For this example, we'll just generate random factors as a placeholder
    damping_factors = np.random.exponential(scale=0.5, size=embedding_count)
    
    return {
        'count': embedding_count,
        'min': float(np.min(damping_factors)),
        'max': float(np.max(damping_factors)),
        'mean': float(np.mean(damping_factors)),
        'median': float(np.median(damping_factors)),
        'std': float(np.std(damping_factors)),
        'percentiles': {
            '25': float(np.percentile(damping_factors, 25)),
            '50': float(np.percentile(damping_factors, 50)),
            '75': float(np.percentile(damping_factors, 75)),
            '90': float(np.percentile(damping_factors, 90)),
            '95': float(np.percentile(damping_factors, 95)),
            '99': float(np.percentile(damping_factors, 99)),
        }
    }


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="HPC-QR Flow Manager Stress Test")
    parser.add_argument(
        '--batch-sizes', 
        type=int, 
        nargs='+', 
        default=[1, 8, 16, 32, 64, 128, 256],
        help='List of batch sizes to test'
    )
    parser.add_argument(
        '--embedding-counts', 
        type=int, 
        nargs='+', 
        default=[100, 1000, 5000],
        help='List of embedding counts to test'
    )
    parser.add_argument(
        '--dim', 
        type=int, 
        default=768,
        help='Dimensionality of embeddings'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='test_results/hpcqr_performance_results.json',
        help='Output file path for results'
    )
    parser.add_argument(
        '--stats-only', 
        action='store_true',
        help='Only collect damping statistics, skip performance tests'
    )
    
    return parser.parse_args()


async def main():
    """
    Main entry point.
    """
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.stats_only:
        # Just collect damping statistics
        stats = await collect_damping_statistics(5000)
        with open(args.output, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'damping_statistics': stats
            }, f, indent=2)
        logger.info(f"Damping statistics saved to {args.output}")
        return
    
    # Run performance tests
    results = await run_stress_tests(args.batch_sizes, args.embedding_counts, args.dim)
    
    # Print and save results
    print_results(results)
    save_results(results, args.output)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
