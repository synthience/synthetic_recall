#!/usr/bin/env python
"""
GPU-Optimized TensorServer

This script enhances the TensorServer with GPU optimization features specifically
for testing the shock absorption mechanism in the HPC-QR Flow Manager.
"""

import os
import sys
import time
import json
import logging
import argparse
import asyncio
import torch
import torch.cuda as cuda
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gpu_optimized_tensor_server")

# Import the server components
from server.tensor_server import TensorServer
from memory.lucidia_memory_system.core.integration.hpc_qr_flow_manager import HPCQRFlowManager


class GPUOptimizedTensorServer(TensorServer):
    """
    GPU-Optimized extension of the TensorServer with enhanced monitoring and optimization
    for the HPC-QR Flow Manager shock absorption testing.
    """
    
    def __init__(self, config=None):
        """Initialize the GPU-Optimized TensorServer"""
        super().__init__(config)
        
        # Additional GPU monitoring attributes
        self.gpu_stats = {
            'memory_allocated': 0,
            'memory_reserved': 0,
            'memory_cached': 0,
            'last_refresh': 0,
            'peak_memory': 0,
            'utilization': 0
        }
        
        # Configure HPC-QR Flow Manager
        self.hpcqr_config = {
            'batch_size': config.get('batch_size', 32),
            'embedding_dim': config.get('embedding_dim', 768),
            'shock_absorption_enabled': True,
            'dynamic_scaling_factor': config.get('dynamic_scaling_factor', 0.5),
            'max_momentum_size': config.get('max_momentum_size', 10000),
            'update_threshold': config.get('update_threshold', 0.05),
            'drift_threshold': config.get('drift_threshold', 0.3),
        }
        
        self.flow_manager = HPCQRFlowManager(self.hpcqr_config)
        
        # Set up GPU monitoring if available
        if cuda.is_available():
            logger.info(f"GPU detected: {cuda.get_device_name(0)}")
            # Start monitoring in background
            self._start_gpu_monitoring()
        else:
            logger.warning("No GPU detected, running in CPU mode")
    
    def _start_gpu_monitoring(self, interval_seconds=5):
        """Start background GPU monitoring"""
        async def _monitor_gpu():
            while True:
                self.update_gpu_stats()
                await asyncio.sleep(interval_seconds)
        
        # Run the monitoring as a background task
        asyncio.create_task(_monitor_gpu())
    
    def update_gpu_stats(self):
        """Update GPU statistics"""
        if not cuda.is_available():
            return
        
        # Memory stats
        self.gpu_stats['memory_allocated'] = cuda.memory_allocated() / (1024 ** 2)  # MB
        self.gpu_stats['memory_reserved'] = cuda.memory_reserved() / (1024 ** 2)  # MB
        self.gpu_stats['memory_cached'] = 0  # Not directly accessible in PyTorch
        
        # Peak memory
        current_peak = cuda.max_memory_allocated() / (1024 ** 2)  # MB
        if current_peak > self.gpu_stats['peak_memory']:
            self.gpu_stats['peak_memory'] = current_peak
        
        # Update refresh timestamp
        self.gpu_stats['last_refresh'] = time.time()
        
        # Log the stats
        logger.debug(f"GPU Memory Allocated: {self.gpu_stats['memory_allocated']:.2f} MB, "
                     f"Peak: {self.gpu_stats['peak_memory']:.2f} MB")
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics"""
        # Refresh stats before returning
        self.update_gpu_stats()
        return self.gpu_stats
    
    async def optimize_batch_size(self, embedding_counts: List[int] = [100, 500, 1000], 
                          batch_sizes: List[int] = [1, 8, 16, 32, 64, 128]):
        """Find optimal batch size for current GPU setup"""
        logger.info("Running batch size optimization test...")
        
        from tools.hpcqr_stress_test import run_stress_tests
        
        results = await run_stress_tests(batch_sizes, embedding_counts)
        
        # Find optimal batch sizes by embedding count
        optimal_batch_sizes = {}
        for result in results:
            count = result['embedding_count']
            if count not in optimal_batch_sizes:
                optimal_batch_sizes[count] = result
            elif result['avg_time_per_embedding'] < optimal_batch_sizes[count]['avg_time_per_embedding']:
                optimal_batch_sizes[count] = result
        
        # Log results
        logger.info("Optimal batch sizes:")
        for count, result in sorted(optimal_batch_sizes.items()):
            logger.info(f"For {count} embeddings: Batch size {result['batch_size']} "
                       f"(avg time: {result['avg_time_per_embedding']:.6f}s)")
        
        # Return the overall optimal batch size (for the largest embedding count)
        if optimal_batch_sizes:
            max_count = max(optimal_batch_sizes.keys())
            optimal_size = optimal_batch_sizes[max_count]['batch_size']
            
            # Update the flow manager config
            self.hpcqr_config['batch_size'] = optimal_size
            
            return optimal_size
        return None
    
    async def process_with_shock_absorption(self, embeddings: List[torch.Tensor]) -> List[Tuple[torch.Tensor, float]]:
        """Process embeddings with shock absorption"""
        # Ensure GPU memory stats are updated before processing
        self.update_gpu_stats()
        
        start_time = time.time()
        
        # Use the flow manager for processing
        results = await self.flow_manager.process_embedding_batch(embeddings)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(embeddings)} embeddings in {processing_time:.4f}s "
                   f"({processing_time/len(embeddings):.6f}s per embedding)")
        
        # Update GPU stats after processing
        self.update_gpu_stats()
        
        return results
    
    async def close(self):
        """Clean up resources"""
        await self.flow_manager.close()
        
        # Clear GPU cache
        if cuda.is_available():
            cuda.empty_cache()
        
        logger.info("GPU-Optimized TensorServer resources cleaned up")


async def run_server(host='0.0.0.0', port=5001, config=None):
    """Run the GPU-Optimized TensorServer"""
    # Default configuration
    if config is None:
        config = {
            'batch_size': 32,
            'embedding_dim': 768,
            'dynamic_scaling_factor': 0.5,
            'max_momentum_size': 10000
        }
    
    # Initialize server
    server = GPUOptimizedTensorServer(config)
    
    # Run optimization test to find best batch size
    optimal_batch_size = await server.optimize_batch_size()
    logger.info(f"Using optimal batch size: {optimal_batch_size}")
    
    # TODO: Implement HTTP API for remote access if needed
    
    # Keep server running
    try:
        logger.info(f"Server running, ready for processing. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(60)  # Keep alive
            server.update_gpu_stats()  # Periodic GPU monitoring
            
            # Log current stats
            stats = server.get_gpu_stats()
            logger.info(f"Current GPU Memory: {stats['memory_allocated']:.2f} MB, "
                       f"Peak: {stats['peak_memory']:.2f} MB")
    
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    finally:
        await server.close()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GPU-Optimized TensorServer")
    parser.add_argument(
        '--host', 
        type=str, 
        default='0.0.0.0',
        help='Host to bind the server to'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=5001,
        help='Port to bind the server to'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Initial batch size (will be optimized during startup)'
    )
    parser.add_argument(
        '--dynamic-scaling-factor', 
        type=float, 
        default=0.5,
        help='Dynamic scaling factor for shock absorption'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_arguments()
    
    config = {
        'batch_size': args.batch_size,
        'dynamic_scaling_factor': args.dynamic_scaling_factor
    }
    
    await run_server(args.host, args.port, config)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
