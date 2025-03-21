#!/usr/bin/env python
"""
GPU Resource Manager

This module provides utilities for GPU resource management and monitoring,
specifically optimized for the HPC-QR Flow Manager shock absorption testing.
"""

import os
import sys
import time
import math
import json
import logging
import argparse
import subprocess
from typing import Dict, List, Any, Optional, Tuple

try:
    import torch
    import torch.cuda as cuda
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gpu_resource_manager")


class GPUStats:
    """Class to collect and track GPU statistics"""
    
    def __init__(self):
        """Initialize GPU statistics tracking"""
        self.has_gpu = False
        self.device_count = 0
        self.current_device = 0
        self.devices = []
        self.stats_history = []
        self.history_limit = 100  # Keep last 100 stats snapshots
        
        self._initialize()
    
    def _initialize(self):
        """Initialize GPU detection"""
        if not HAS_TORCH:
            logger.warning("PyTorch not available, GPU stats will be limited")
            return
        
        self.has_gpu = cuda.is_available()
        if not self.has_gpu:
            logger.warning("No CUDA-compatible GPU detected")
            return
        
        self.device_count = cuda.device_count()
        self.current_device = cuda.current_device()
        
        # Collect information about each device
        for i in range(self.device_count):
            cuda.set_device(i)
            device_info = {
                'index': i,
                'name': cuda.get_device_name(i),
                'capability': f"{cuda.get_device_capability(i)[0]}.{cuda.get_device_capability(i)[1]}",
                'total_memory': cuda.get_device_properties(i).total_memory / (1024 ** 2),  # MB
            }
            self.devices.append(device_info)
        
        # Reset to original device
        cuda.set_device(self.current_device)
        
        logger.info(f"Detected {self.device_count} GPU(s)")
        for device in self.devices:
            logger.info(f"GPU {device['index']}: {device['name']} (Compute {device['capability']}) "
                      f"with {device['total_memory']:.2f} MB memory")
    
    def update_stats(self) -> Dict[str, Any]:
        """Update GPU statistics"""
        if not self.has_gpu or not HAS_TORCH:
            return None
        
        stats = {
            'timestamp': time.time(),
            'devices': []
        }
        
        # Get stats for each device
        for i in range(self.device_count):
            cuda.set_device(i)
            
            # Memory stats in MB
            allocated = cuda.memory_allocated(i) / (1024 ** 2)
            reserved = cuda.memory_reserved(i) / (1024 ** 2)
            max_allocated = cuda.max_memory_allocated(i) / (1024 ** 2)
            
            device_stats = {
                'index': i,
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'max_allocated_mb': max_allocated,
                'utilization_memory': allocated / self.devices[i]['total_memory'] * 100,
            }
            stats['devices'].append(device_stats)
        
        # Reset to original device
        cuda.set_device(self.current_device)
        
        # Add to history and trim if needed
        self.stats_history.append(stats)
        if len(self.stats_history) > self.history_limit:
            self.stats_history = self.stats_history[-self.history_limit:]
        
        return stats
    
    def get_latest_stats(self) -> Dict[str, Any]:
        """Get the most recent stats (updating first)"""
        return self.update_stats()
    
    def get_stats_history(self) -> List[Dict[str, Any]]:
        """Get the history of stats"""
        return self.stats_history
    
    def get_memory_trend(self, device_index: int = 0) -> Dict[str, List[float]]:
        """Get memory usage trend for a specific device"""
        if not self.stats_history:
            return {'timestamps': [], 'allocated': [], 'reserved': [], 'max_allocated': []}
        
        trend = {
            'timestamps': [],
            'allocated': [],
            'reserved': [],
            'max_allocated': []
        }
        
        for stats in self.stats_history:
            trend['timestamps'].append(stats['timestamp'])
            if device_index < len(stats['devices']):
                device = stats['devices'][device_index]
                trend['allocated'].append(device['allocated_mb'])
                trend['reserved'].append(device['reserved_mb'])
                trend['max_allocated'].append(device['max_allocated_mb'])
            else:
                # Fill with zeros if device index is out of range
                trend['allocated'].append(0)
                trend['reserved'].append(0)
                trend['max_allocated'].append(0)
        
        return trend
    
    def clear_memory(self, device_index: Optional[int] = None):
        """Clear cached memory on specified device or all devices"""
        if not self.has_gpu or not HAS_TORCH:
            return
        
        if device_index is not None:
            if 0 <= device_index < self.device_count:
                cuda.set_device(device_index)
                cuda.empty_cache()
                logger.info(f"Cleared memory cache for GPU {device_index}")
            else:
                logger.warning(f"Invalid device index: {device_index}")
        else:
            # Clear all devices
            for i in range(self.device_count):
                cuda.set_device(i)
                cuda.empty_cache()
            logger.info("Cleared memory cache for all GPUs")
        
        # Reset to original device
        cuda.set_device(self.current_device)
    
    def reset_peak_stats(self, device_index: Optional[int] = None):
        """Reset peak memory stats"""
        if not self.has_gpu or not HAS_TORCH:
            return
        
        if device_index is not None:
            if 0 <= device_index < self.device_count:
                cuda.set_device(device_index)
                cuda.reset_peak_memory_stats(device_index)
                logger.info(f"Reset peak memory stats for GPU {device_index}")
            else:
                logger.warning(f"Invalid device index: {device_index}")
        else:
            # Reset all devices
            for i in range(self.device_count):
                cuda.set_device(i)
                cuda.reset_peak_memory_stats(i)
            logger.info("Reset peak memory stats for all GPUs")
        
        # Reset to original device
        cuda.set_device(self.current_device)


def print_gpu_info(detailed: bool = False):
    """Print GPU information to console"""
    stats = GPUStats()
    
    if not stats.has_gpu:
        print("No CUDA-compatible GPU detected")
        return
    
    print(f"\n{'=' * 60}")
    print(f"{'GPU INFORMATION':^60}")
    print(f"{'=' * 60}\n")
    
    for i, device in enumerate(stats.devices):
        print(f"GPU {i}: {device['name']}")
        print(f"Compute Capability: {device['capability']}")
        print(f"Total Memory: {device['total_memory']:.2f} MB")
        
        if detailed:
            # Get current stats
            current_stats = stats.get_latest_stats()
            if current_stats and i < len(current_stats['devices']):
                dev_stats = current_stats['devices'][i]
                print(f"Memory Allocated: {dev_stats['allocated_mb']:.2f} MB")
                print(f"Memory Reserved: {dev_stats['reserved_mb']:.2f} MB")
                print(f"Max Memory Allocated: {dev_stats['max_allocated_mb']:.2f} MB")
                print(f"Memory Utilization: {dev_stats['utilization_memory']:.2f}%")
        
        print()
    
    # PyTorch build info
    if HAS_TORCH:
        print("PyTorch build information:")
        print(f"Version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
            print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
            print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print()


def optimize_memory_settings():
    """Optimize GPU memory settings for better performance"""
    if not HAS_TORCH or not cuda.is_available():
        logger.warning("Cannot optimize memory settings - GPU not available")
        return
    
    # Enable cuDNN benchmark for optimized performance
    torch.backends.cudnn.benchmark = True
    logger.info("Enabled cuDNN benchmark mode for optimized performance")
    
    # Optimize memory allocation
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    
    # Set preferred memory allocation mechanism to avoid fragmentation
    if hasattr(torch.cuda, 'memory_stats'):
        # This is an advanced feature in newer PyTorch versions
        logger.info("Advanced memory management available")
    
    logger.info("Memory optimization settings applied")


def monitor_gpu_usage(interval_sec: int = 5, duration_sec: int = 60, output_file: Optional[str] = None):
    """Monitor GPU usage for a specified duration"""
    stats = GPUStats()
    
    if not stats.has_gpu:
        logger.warning("No GPU available for monitoring")
        return
    
    start_time = time.time()
    end_time = start_time + duration_sec
    
    logger.info(f"Starting GPU monitoring for {duration_sec} seconds with {interval_sec} second intervals")
    
    try:
        while time.time() < end_time:
            current_stats = stats.update_stats()
            
            # Print current stats
            for i, device_stats in enumerate(current_stats['devices']):
                print(f"GPU {i}: Memory Used: {device_stats['allocated_mb']:.2f} MB / "
                      f"{stats.devices[i]['total_memory']:.2f} MB "
                      f"({device_stats['utilization_memory']:.2f}%)")
            
            # Sleep until next update
            time.sleep(interval_sec)
    
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    
    # Export results if requested
    if output_file:
        history = stats.get_stats_history()
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"GPU stats history saved to {output_file}")
    
    return stats.get_stats_history()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GPU Resource Manager")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display GPU information')
    info_parser.add_argument(
        '--detailed', 
        action='store_true',
        help='Show detailed GPU information'
    )
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor GPU usage')
    monitor_parser.add_argument(
        '--interval', 
        type=int, 
        default=5,
        help='Monitoring interval in seconds'
    )
    monitor_parser.add_argument(
        '--duration', 
        type=int, 
        default=60,
        help='Monitoring duration in seconds'
    )
    monitor_parser.add_argument(
        '--output', 
        type=str,
        help='Output file for monitoring data (JSON format)'
    )
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize GPU memory settings')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear GPU memory cache')
    clear_parser.add_argument(
        '--device', 
        type=int, 
        default=None,
        help='Device index to clear (default: all devices)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    if args.command == 'info':
        print_gpu_info(args.detailed)
    
    elif args.command == 'monitor':
        monitor_gpu_usage(args.interval, args.duration, args.output)
    
    elif args.command == 'optimize':
        optimize_memory_settings()
    
    elif args.command == 'clear':
        stats = GPUStats()
        stats.clear_memory(args.device)
        stats.reset_peak_stats(args.device)
    
    else:
        # Default to showing basic info
        print_gpu_info(False)


if __name__ == "__main__":
    main()
