#!/usr/bin/env python
"""
Test script for the HPCQRFlowManager integration with UnifiedQuickRecallCalculator

This script validates the functionality and performance of the integrated components
through a series of tests and benchmarks.

Usage:
    python tests/test_hpc_qr_integration.py --benchmark --mode standard --device cuda
"""

import os
import sys
import asyncio
import time
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.lucidia_memory_system.core.integration.hpc_qr_flow_manager import HPCQRFlowManager
from server.qr_calculator import UnifiedQuickRecallCalculator, QuickRecallMode, QuickRecallFactor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_hpc_qr_integration")


class TestHPCQRIntegration:
    """Test suite for HPCQRFlowManager integration with UnifiedQuickRecallCalculator"""
    
    def __init__(self, config=None):
        """
        Initialize test suite
        
        Args:
            config: Configuration for HPCQRFlowManager
        """
        self.config = config or {
            'embedding_dim': 384,
            'batch_size': 16,
            'device': 'cpu',
            'alpha': 0.4,
            'beta': 0.3,
            'gamma': 0.2,
            'delta': 0.1,
            'max_momentum_size': 100
        }
        self.flow_manager = HPCQRFlowManager(self.config)
        self.results = {}
        
    def generate_test_embeddings(self, count=20, dim=None) -> List[torch.Tensor]:
        """
        Generate random embeddings for testing
        
        Args:
            count: Number of embeddings to generate
            dim: Embedding dimension (uses config if None)
            
        Returns:
            List of random embeddings
        """
        if dim is None:
            dim = self.config['embedding_dim']
            
        device = torch.device(self.config['device'])
        return [torch.randn(dim, device=device) for _ in range(count)]
        
    async def test_process_single_embedding(self):
        """
        Test processing a single embedding
        
        Returns:
            Test result dictionary
        """
        logger.info("Testing single embedding processing...")
        start_time = time.time()
        
        # Generate test embeddings
        test_embeddings = self.generate_test_embeddings(count=5)
        results = []
        
        # Process each embedding
        for i, emb in enumerate(test_embeddings):
            item_start = time.time()
            adjusted, score = await self.flow_manager.process_embedding(emb)
            item_time = time.time() - item_start
            
            # Verify results
            assert adjusted.shape == emb.shape, f"Shape mismatch: {adjusted.shape} != {emb.shape}"
            assert 0 <= score <= 1, f"Score {score} outside valid range [0,1]"
            
            results.append({
                "index": i,
                "score": float(score),
                "time_ms": item_time * 1000,
                "is_valid": True
            })
            
        elapsed = time.time() - start_time
        
        test_result = {
            "test": "process_single_embedding",
            "passed": True,
            "time_total_ms": elapsed * 1000,
            "time_avg_ms": (elapsed * 1000) / len(test_embeddings),
            "results": results
        }
        
        self.results["single_embedding"] = test_result
        logger.info(f"Single embedding test complete: {len(results)} embeddings in {elapsed:.2f}s")
        
        return test_result
    
    async def test_process_batch(self):
        """
        Test batch processing of embeddings
        
        Returns:
            Test result dictionary
        """
        logger.info("Testing batch processing...")
        start_time = time.time()
        
        # Generate test embeddings
        batch_sizes = [2, 8, 16]
        batch_results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size {batch_size}...")
            test_embeddings = self.generate_test_embeddings(count=batch_size)
            
            # Process batch
            batch_start = time.time()
            results = await self.flow_manager.process_embedding_batch(test_embeddings)
            batch_time = time.time() - batch_start
            
            # Verify results
            assert len(results) == batch_size, f"Result count mismatch: {len(results)} != {batch_size}"
            
            score_sum = 0
            for i, (adjusted, score) in enumerate(results):
                assert adjusted.shape == test_embeddings[i].shape, f"Shape mismatch at index {i}"
                assert 0 <= score <= 1, f"Score {score} outside valid range [0,1]"
                score_sum += score
            
            batch_results[batch_size] = {
                "time_ms": batch_time * 1000,
                "avg_score": score_sum / batch_size,
                "throughput": batch_size / batch_time  # embeddings per second
            }
            
        elapsed = time.time() - start_time
        
        test_result = {
            "test": "process_batch",
            "passed": True,
            "time_total_ms": elapsed * 1000,
            "batch_results": batch_results
        }
        
        self.results["batch_processing"] = test_result
        logger.info(f"Batch processing test complete in {elapsed:.2f}s")
        
        return test_result
    
    async def test_momentum_buffer_sync(self):
        """
        Test that momentum buffer is properly synced with calculator
        
        Returns:
            Test result dictionary
        """
        logger.info("Testing momentum buffer synchronization...")
        start_time = time.time()
        
        # Process some embeddings to populate momentum buffer
        test_embeddings = self.generate_test_embeddings(count=10)
        for emb in test_embeddings:
            await self.flow_manager.process_embedding(emb)
        
        # Verify buffer is synchronized
        tensor_count = len(self.flow_manager.momentum_buffer) 
        external_count = len(self.flow_manager.qr_calculator.external_momentum)
        
        synchronized = tensor_count == external_count
        assert synchronized, f"Buffer sizes don't match: {tensor_count} != {external_count}"
        
        elapsed = time.time() - start_time
        
        test_result = {
            "test": "momentum_buffer_sync",
            "passed": synchronized,
            "time_total_ms": elapsed * 1000,
            "buffer_size": tensor_count,
            "external_buffer_size": external_count
        }
        
        self.results["momentum_sync"] = test_result
        logger.info(f"Momentum sync test: {'PASSED' if synchronized else 'FAILED'}")
        
        return test_result
    
    async def test_config_propagation(self):
        """
        Test that configuration updates propagate correctly
        
        Returns:
            Test result dictionary
        """
        logger.info("Testing configuration propagation...")
        start_time = time.time()
        
        # Update configuration
        new_config = {
            'alpha': 0.5,  # Changed from 0.4
            'beta': 0.25,  # Changed from 0.3
            'gamma': 0.15,  # Changed from 0.2
            'delta': 0.1,   # Unchanged
            'calculator_mode': 'custom',
            'factor_weights': {
                'R_GEOMETRY': 0.6,
                'CAUSAL_NOVELTY': 0.3,
                'SELF_ORG': 0.05,
                'OVERLAP': 0.05
            }
        }
        
        # Save original values
        original_alpha = self.flow_manager.config['alpha']
        original_beta = self.flow_manager.config['beta']
        original_mode = self.flow_manager.qr_calculator.config['mode']
        
        # Update configuration
        self.flow_manager.update_calculator_config(new_config)
        
        # Verify configuration was updated
        config_updated = (
            self.flow_manager.config['alpha'] == new_config['alpha'] and
            self.flow_manager.config['beta'] == new_config['beta'] and
            self.flow_manager.qr_calculator.config['alpha'] == new_config['alpha'] and
            self.flow_manager.qr_calculator.config['beta'] == new_config['beta'] and
            self.flow_manager.qr_calculator.config['mode'] == QuickRecallMode.CUSTOM
        )
        
        # Verify factor weights were updated
        weights_updated = (
            self.flow_manager.qr_calculator.factor_weights[QuickRecallFactor.R_GEOMETRY] == 0.6 and
            self.flow_manager.qr_calculator.factor_weights[QuickRecallFactor.CAUSAL_NOVELTY] == 0.3
        )
        
        elapsed = time.time() - start_time
        
        test_result = {
            "test": "config_propagation",
            "passed": config_updated and weights_updated,
            "time_total_ms": elapsed * 1000,
            "config_updated": config_updated,
            "weights_updated": weights_updated,
            "original": {
                "alpha": original_alpha,
                "beta": original_beta,
                "mode": original_mode.value if original_mode else "none"
            },
            "new": {
                "alpha": self.flow_manager.config['alpha'],
                "beta": self.flow_manager.config['beta'],
                "mode": self.flow_manager.qr_calculator.config['mode'].value
            }
        }
        
        self.results["config_propagation"] = test_result
        logger.info(f"Configuration propagation test: {'PASSED' if config_updated and weights_updated else 'FAILED'}")
        
        return test_result
    
    async def benchmark_modes(self, embeddings_count=50):
        """
        Benchmark different calculator modes
        
        Args:
            embeddings_count: Number of embeddings to use for benchmarking
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking modes with {embeddings_count} embeddings...")
        
        # Generate test embeddings
        test_embeddings = self.generate_test_embeddings(count=embeddings_count)
        
        # Define batch sizes and modes to test
        batch_sizes = [1, 8, 32]
        modes = [QuickRecallMode.STANDARD, QuickRecallMode.EFFICIENT, 
                 QuickRecallMode.PRECISE, QuickRecallMode.HPC_QR]
        
        # Run benchmark
        benchmark_results = await self.flow_manager.benchmark_modes(
            test_embeddings=test_embeddings,
            batch_sizes=batch_sizes,
            modes_to_test=modes
        )
        
        # Store results
        self.results["benchmark"] = benchmark_results
        
        # Print summary
        logger.info("Benchmark results summary:")
        for mode, results in benchmark_results.items():
            logger.info(f"Mode: {mode}")
            for batch_size, metrics in results.items():
                logger.info(f"  {batch_size}: {metrics['throughput_eps']:.2f} embeddings/sec, "
                           f"avg: {metrics['time_per_embedding_avg_ms']:.2f}ms, "
                           f"p95: {metrics['time_per_embedding_p95_ms']:.2f}ms")
        
        return benchmark_results
    
    async def run_all_tests(self, run_benchmark=False):
        """
        Run all tests
        
        Args:
            run_benchmark: Whether to run benchmarking tests
            
        Returns:
            Dictionary with all test results
        """
        logger.info("Running all tests...")
        start_time = time.time()
        
        # Run basic tests
        await self.test_process_single_embedding()
        await self.test_process_batch()
        await self.test_momentum_buffer_sync()
        await self.test_config_propagation()
        
        # Run benchmarks if requested
        if run_benchmark:
            await self.benchmark_modes()
            
            # Get detailed performance stats
            self.results["performance_stats"] = self.flow_manager.get_detailed_performance_stats()
        
        elapsed = time.time() - start_time
        logger.info(f"All tests completed in {elapsed:.2f}s")
        
        # Add summary
        self.results["summary"] = {
            "total_time_ms": elapsed * 1000,
            "all_passed": all(r.get("passed", False) for r in self.results.values() if "passed" in r),
            "test_count": sum(1 for r in self.results.values() if "passed" in r),
            "device": self.config["device"]
        }
        
        return self.results
    
    def save_results(self, output_file=None):
        """
        Save test results to file
        
        Args:
            output_file: Path to output file (default: benchmark_results_{timestamp}.md)
            
        Returns:
            Path to output file
        """
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"benchmark_results/benchmark_summary_{timestamp}.md"
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Format results as markdown
        markdown = f"# HPCQRFlowManager Integration Test Results\n\n"
        markdown += f"Test run: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary
        if "summary" in self.results:
            summary = self.results["summary"]
            markdown += f"## Summary\n\n"
            markdown += f"- Total time: {summary['total_time_ms'] / 1000:.2f}s\n"
            markdown += f"- All tests passed: {summary['all_passed']}\n"
            markdown += f"- Test count: {summary['test_count']}\n"
            markdown += f"- Device: {summary['device']}\n\n"
        
        # Individual test results
        for test_name, result in self.results.items():
            if test_name == "summary" or test_name == "benchmark" or test_name == "performance_stats":
                continue
                
            markdown += f"## {test_name}\n\n"
            if "passed" in result:
                markdown += f"- Passed: {result['passed']}\n"
            if "time_total_ms" in result:
                markdown += f"- Time: {result['time_total_ms'] / 1000:.4f}s\n"
            
            # Add test-specific details
            if test_name == "momentum_sync":
                markdown += f"- Buffer size: {result['buffer_size']}\n"
                markdown += f"- External buffer size: {result['external_buffer_size']}\n"
            elif test_name == "config_propagation":
                markdown += f"- Config updated: {result['config_updated']}\n"
                markdown += f"- Weights updated: {result['weights_updated']}\n"
                markdown += f"- Original: {result['original']}\n"
                markdown += f"- New: {result['new']}\n"
            
            markdown += "\n"
        
        # Benchmark results
        if "benchmark" in self.results:
            markdown += f"## Benchmark Results\n\n"
            
            # Create table
            markdown += f"| Mode | Batch Size | Throughput (emb/s) | Avg Time (ms) | P95 Time (ms) | Avg Score |\n"
            markdown += f"|------|------------|-------------------|--------------|--------------|-----------|\n"
            
            for mode, results in self.results["benchmark"].items():
                for batch_key, metrics in results.items():
                    batch_size = batch_key.split("_")[1]  # Extract size from "batch_X"
                    markdown += (f"| {mode} | {batch_size} | {metrics['throughput_eps']:.2f} | "
                               f"{metrics['time_per_embedding_avg_ms']:.2f} | "
                               f"{metrics['time_per_embedding_p95_ms']:.2f} | "
                               f"{metrics['avg_score']:.4f} |\n")
            
            markdown += "\n"
        
        # Performance statistics
        if "performance_stats" in self.results:
            stats = self.results["performance_stats"]
            markdown += f"## System Performance\n\n"
            
            # CPU info
            if "cpu" in stats:
                cpu = stats["cpu"]
                markdown += f"### CPU\n\n"
                for key, value in cpu.items():
                    markdown += f"- {key}: {value}\n"
                markdown += "\n"
            
            # GPU info
            if "gpu" in stats:
                gpu = stats["gpu"]
                markdown += f"### GPU\n\n"
                markdown += f"- Count: {gpu.get('count', 0)}\n"
                markdown += f"- Memory allocated: {gpu.get('memory_allocated_mb', 0):.2f} MB\n"
                markdown += f"- Memory reserved: {gpu.get('memory_reserved_mb', 0):.2f} MB\n\n"
                
                # Per-device info
                if "devices" in gpu:
                    markdown += f"#### Devices\n\n"
                    for device_id, device_info in gpu["devices"].items():
                        markdown += f"**Device {device_id}**: {device_info.get('name', 'Unknown')}\n"
                        markdown += f"- Memory allocated: {device_info.get('memory_allocated_mb', 0):.2f} MB\n"
                        if 'gpu_utilization_percent' in device_info:
                            markdown += f"- GPU utilization: {device_info['gpu_utilization_percent']}%\n"
                        markdown += "\n"
        
        # Write to file
        with open(output_file, "w") as f:
            f.write(markdown)
            
        logger.info(f"Results saved to {output_file}")
        return output_file


async def main():
    """
    Main entry point for test script
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test HPCQRFlowManager integration")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarking tests")
    parser.add_argument("--mode", choices=["standard", "efficient", "precise", "hpc_qr"], 
                        default="hpc_qr", help="Calculator mode to use")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", 
                        help="Device to use for calculation")
    parser.add_argument("--output", help="Output file for test results")
    args = parser.parse_args()
    
    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    
    # Set up configuration
    config = {
        'embedding_dim': 384,
        'batch_size': 16,
        'device': args.device,
        'alpha': 0.4,
        'beta': 0.3,
        'gamma': 0.2,
        'delta': 0.1,
        'max_momentum_size': 100,
        'calculator_mode': args.mode
    }
    
    # Initialize test suite
    logger.info(f"Initializing test suite with {args.device} device and {args.mode} calculator mode")
    test_suite = TestHPCQRIntegration(config)
    
    # Run validation
    validation = test_suite.flow_manager.validate_configuration()
    if not validation["valid"]:
        logger.warning("Configuration validation failed:")
        for warning in validation["warnings"]:
            logger.warning(f"- {warning}")
        for recommendation in validation["recommendations"]:
            logger.info(f"Recommendation: {recommendation}")
    
    # Run tests
    await test_suite.run_all_tests(run_benchmark=args.benchmark)
    
    # Save results
    output_file = test_suite.save_results(args.output)
    logger.info(f"Test complete, results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
