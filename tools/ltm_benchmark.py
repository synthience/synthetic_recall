#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTM Benchmark - Real-world data testing for HPC-QR Flow Manager

This script benchmarks the HPC-QR Flow Manager using real-world data from LTM folders
to evaluate performance and scalability on GPU-accelerated environments.
"""

import os
import json
import time
import argparse
import asyncio
import logging
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ltm_benchmark")

# Import HPC-QR Flow Manager
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.append(sys_path)

import sys
from memory.lucidia_memory_system.core.integration.hpc_qr_flow_manager import HPCQRFlowManager
from tools.gpu_resource_manager import GPUStats


class LTMBenchmark:
    """Benchmark HPC-QR Flow Manager with real-world LTM data"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize benchmark"""
        self.config = config
        self.ltm_path = config.get('ltm_path')
        self.batch_sizes = config.get('batch_sizes', [16, 32, 64, 128])
        self.output_dir = config.get('output_dir', 'benchmark_results')
        self.max_files = config.get('max_files', 1000)
        self.gpu_stats = GPUStats()
        
        # Configure HPC-QR based on provided config
        self.hpcqr_config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'batch_size': 64,  # Default batch size, will be changed during benchmarking
            'shock_absorption_enabled': True,
            'dynamic_scaling_factor': config.get('dynamic_scaling_factor', 0.5),
            'update_threshold': config.get('update_threshold', 0.05),
            'drift_threshold': config.get('drift_threshold', 0.3),
        }
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Results storage
        self.results = {}
        
        logger.info(f"Initialized LTM Benchmark with config: {config}")
        
    async def process_ltm_folder(self, folder_path: Path, batch_size: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Process files in an LTM folder with specified batch size"""
        metrics = {
            'total_files': 0,
            'total_conversations': 0,
            'total_messages': 0,
            'total_words': 0,
            'total_processed_time': 0.0,
            'max_gpu_memory_used': 0.0,
            'avg_time_per_message': 0.0,
            'avg_gpu_utilization': 0.0
        }
        
        gpu_metrics = []
        
        # Update HPC-QR config with current batch size
        self.hpcqr_config['batch_size'] = batch_size
        
        # Initialize HPCQRFlowManager with current config
        flow_manager = HPCQRFlowManager(self.hpcqr_config)
        
        # Find all JSON files in the folder
        json_files = list(folder_path.glob('**/*.json'))
        if len(json_files) > self.max_files:
            logger.info(f"Limiting to {self.max_files} files from {len(json_files)} available")
            json_files = json_files[:self.max_files]
        
        # Process files in batches
        for json_file in tqdm(json_files, desc="Processing LTM files"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract messages from various JSON formats
                messages = self._extract_messages(data)
                if not messages:
                    continue
                
                # Convert messages to text for embedding
                texts = [self._message_to_text(msg) for msg in messages]
                word_count = sum(len(text.split()) for text in texts)
                
                # Get embeddings for each message
                start_time = time.time()
                
                # Generate embeddings batch by batch
                embeddings = []
                try:
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i+batch_size]
                        # Use mock embeddings for testing, since we just want to benchmark the HPC-QR flow
                        # This will avoid TensorServer connectivity issues
                        batch_embeddings = [torch.randn(768, device=flow_manager.config['device']) for _ in batch_texts]
                        embeddings.extend(batch_embeddings)
                    
                    # Process embeddings through HPC-QR with batch processing
                    for i in range(0, len(embeddings), batch_size):
                        batch_embeddings = embeddings[i:i+batch_size]
                        _ = await flow_manager.process_embedding_batch(batch_embeddings)
                    
                    process_time = time.time() - start_time
                    
                    # Update metrics
                    metrics['total_files'] += 1
                    metrics['total_conversations'] += 1  # Assuming each file is a conversation
                    metrics['total_messages'] += len(messages)
                    metrics['total_words'] += word_count
                    metrics['total_processed_time'] += process_time
                    
                    # Capture GPU metrics
                    gpu_snapshot = self.gpu_stats.get_latest_stats()
                    gpu_metrics.append({
                        'timestamp': time.time(),
                        'file': str(json_file),
                        'process_time': process_time,
                        'message_count': len(messages),
                        'gpu_utilization': gpu_snapshot.get('devices', [{}])[0].get('utilization_memory', 0) if gpu_snapshot else 0,
                        'memory_used': gpu_snapshot.get('devices', [{}])[0].get('allocated_mb', 0) if gpu_snapshot else 0,
                        'memory_allocated': gpu_snapshot.get('devices', [{}])[0].get('reserved_mb', 0) if gpu_snapshot else 0
                    })
                    
                    # Update max GPU metrics
                    if gpu_snapshot and gpu_snapshot.get('devices'):
                        metrics['max_gpu_memory_used'] = max(
                            metrics['max_gpu_memory_used'], 
                            gpu_snapshot['devices'][0].get('allocated_mb', 0)
                        )
                except Exception as e:
                    logger.warning(f"Error processing batch: {e}")
                    continue
            except Exception as e:
                logger.warning(f"Error processing file {json_file}: {e}")
                continue
        
        # Calculate averages
        if metrics['total_messages'] > 0:
            metrics['avg_time_per_message'] = metrics['total_processed_time'] / metrics['total_messages']
        
        # Calculate average GPU utilization
        if gpu_metrics:
            metrics['avg_gpu_utilization'] = sum(m.get('gpu_utilization', 0) for m in gpu_metrics if m.get('gpu_utilization') is not None) / len(gpu_metrics)
        
        # Close the flow manager
        await flow_manager.close()
        
        return metrics, gpu_metrics
    
    def _extract_messages(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract messages from various JSON formats"""
        messages = []
        
        # Handle different formats
        if isinstance(data, list):
            # Format where the JSON is a list of messages
            messages = data
        elif 'messages' in data:
            # Format where messages are in a 'messages' field
            messages = data['messages']
        elif 'conversation' in data and isinstance(data['conversation'], list):
            # Format where messages are in a 'conversation' field
            messages = data['conversation']
        
        return messages
    
    def _message_to_text(self, message: Dict[str, Any]) -> str:
        """Convert a message dict to text for embedding"""
        text = ""
        
        # Handle different message formats
        if 'content' in message:
            text = message['content']
        elif 'text' in message:
            text = message['text']
        elif 'message' in message:
            text = message['message']
        
        # Include role if available
        if 'role' in message and text:
            text = f"{message['role']}: {text}"
        
        return text
    
    async def run_benchmark(self):
        """Run the benchmark and return results"""
        # Create output dir if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store results by batch size and folder
        all_results = {}
        for batch_size in self.batch_sizes:
            all_results[str(batch_size)] = {}
        
        # Run benchmark for each batch size
        for batch_size in self.batch_sizes:
            logger.info(f"Running benchmark with batch size {batch_size}...")
            
            # Create HPCQRFlowManager with current batch size
            self.hpcqr_config['batch_size'] = batch_size
            
            # Process LTM files by folder
            ltm_dir = Path(self.ltm_path)
            folder_paths = [f for f in ltm_dir.iterdir() if f.is_dir()]
            
            for folder_path in folder_paths:
                folder_name = folder_path.name
                logger.info(f"Processing folder: {folder_name}")
                
                # Process files in this folder with current batch size
                results, gpu_metrics = await self.process_ltm_folder(folder_path, batch_size)
                
                # Store results for this folder and batch size
                if str(batch_size) not in all_results:
                    all_results[str(batch_size)] = {}
                all_results[str(batch_size)][folder_name] = {
                    'metrics': results,
                    'gpu_metrics': gpu_metrics
                }
        
        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(self.output_dir, f"ltm_benchmark_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {results_file}")
        
        # Generate visualizations if there are any results
        if all_results and any(all_results.values()):
            folders_exist = False
            for bs in all_results:
                if all_results[bs]:
                    folders_exist = True
                    break
            
            if folders_exist:
                self._generate_visualizations(all_results)
            else:
                logger.warning("No folder data available for visualizations")
        else:
            logger.warning("No results available for visualizations")
        
        return all_results
    
    def _generate_visualizations(self, results: Dict[str, Any]):
        """Generate visualizations from benchmark results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Extract data for plotting
        batch_sizes = sorted([bs for bs in results.keys() if results[bs]])  # Only include batch sizes with data
        if not batch_sizes:
            logger.warning("No batch sizes with data available for visualization")
            return
        
        # Get folders that exist across all batch sizes for consistent comparison
        common_folders = set(results[batch_sizes[0]].keys())
        for bs in batch_sizes[1:]:
            common_folders = common_folders.intersection(set(results[bs].keys()))
        
        if not common_folders:
            logger.warning("No common folders across batch sizes for visualization")
            return
        
        folders = sorted(list(common_folders))
        logger.info(f"Generating visualizations for {len(folders)} folders and {len(batch_sizes)} batch sizes")
        
        # 1. Performance by batch size (avg time per message)
        plt.figure(figsize=(12, 8))
        for folder in folders:
            try:
                times = [results[bs][folder]['metrics']['avg_time_per_message'] for bs in batch_sizes
                        if folder in results[bs] and 'metrics' in results[bs][folder]]
                if len(times) == len(batch_sizes):  # Only plot if we have data for all batch sizes
                    plt.plot(batch_sizes, times, marker='o', label=folder)
            except (KeyError, TypeError) as e:
                logger.warning(f"Error plotting time data for folder {folder}: {e}")
        
        plt.xlabel('Batch Size')
        plt.ylabel('Average Time per Message (s)')
        plt.title('Performance by Batch Size')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"perf_by_batch_size_{timestamp}.png"))
        
        # 2. GPU Memory Usage by batch size
        plt.figure(figsize=(12, 8))
        for folder in folders:
            try:
                memory_usage = [results[bs][folder]['metrics']['max_gpu_memory_used'] for bs in batch_sizes
                              if folder in results[bs] and 'metrics' in results[bs][folder]]
                if len(memory_usage) == len(batch_sizes):  # Only plot if we have data for all batch sizes
                    plt.plot(batch_sizes, memory_usage, marker='o', label=folder)
            except (KeyError, TypeError) as e:
                logger.warning(f"Error plotting memory data for folder {folder}: {e}")
        
        plt.xlabel('Batch Size')
        plt.ylabel('Max GPU Memory Usage (MB)')
        plt.title('GPU Memory Usage by Batch Size')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"gpu_memory_by_batch_size_{timestamp}.png"))
        
        # Generate summary report
        self._generate_summary_report(results, timestamp)
    
    def _generate_summary_report(self, results: Dict[str, Any], timestamp: str):
        """Generate a markdown summary report"""
        report_file = os.path.join(self.output_dir, f"benchmark_summary_{timestamp}.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# HPC-QR Flow Manager Benchmark Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # System info
            f.write("## System Information\n\n")
            f.write(f"- Device: {self.hpcqr_config['device']}\n")
            if torch.cuda.is_available():
                f.write(f"- GPU: {torch.cuda.get_device_name(0)}\n")
                f.write(f"- CUDA Version: {torch.version.cuda}\n")
            f.write(f"- PyTorch Version: {torch.__version__}\n\n")
            
            # Overall performance summary
            f.write("## Overall Performance Summary\n\n")
            f.write("| Batch Size | Avg Time Per Message (s) | Max GPU Memory (MB) | Avg GPU Utilization (%) |\n")
            f.write("|------------|--------------------------|---------------------|-------------------------:|\n")
            
            # Get valid batch sizes and folders
            valid_batch_sizes = sorted([int(bs) for bs in results.keys() if results[bs]])
            if not valid_batch_sizes:
                f.write("\nNo valid benchmark data available for reporting.\n")
                return
            
            for bs in valid_batch_sizes:
                bs_str = str(bs)
                if not results[bs_str]:
                    continue
                
                avg_time = 0.0
                max_memory = 0.0
                avg_util = 0.0
                count = 0
                
                for folder in results[bs_str]:
                    if 'metrics' not in results[bs_str][folder]:
                        continue
                    
                    metrics = results[bs_str][folder]['metrics']
                    avg_time += metrics.get('avg_time_per_message', 0)
                    max_memory = max(max_memory, metrics.get('max_gpu_memory_used', 0))
                    avg_util += metrics.get('avg_gpu_utilization', 0)
                    count += 1
                
                if count > 0:
                    avg_time /= count
                    avg_util /= count
                
                f.write(f"| {bs:10d} | {avg_time:24.6f} | {max_memory:19.2f} | {avg_util:23.2f} |\n")
            
            # Results by folder
            f.write("\n## Results by Folder\n\n")
            
            for bs_str in sorted([str(bs) for bs in valid_batch_sizes]):
                f.write(f"### Batch Size: {bs_str}\n\n")
                f.write("| Folder | Messages | Avg Time/Msg (s) | Max GPU Memory (MB) | GPU Util (%) |\n")
                f.write("|--------|----------|-----------------|---------------------|-------------:|\n")
                
                if not results[bs_str]:
                    f.write("No data available for this batch size.\n\n")
                    continue
                
                for folder in sorted(results[bs_str].keys()):
                    if 'metrics' not in results[bs_str][folder]:
                        continue
                    
                    metrics = results[bs_str][folder]['metrics']
                    msg_count = metrics.get('total_messages', 0)
                    avg_time = metrics.get('avg_time_per_message', 0)
                    max_memory = metrics.get('max_gpu_memory_used', 0)
                    avg_util = metrics.get('avg_gpu_utilization', 0)
                    
                    f.write(f"| {folder} | {msg_count:8d} | {avg_time:15.6f} | {max_memory:19.2f} | {avg_util:11.2f} |\n")
                
                f.write("\n")
            
            logger.info(f"Summary report saved to {report_file}")


async def main():
    parser = argparse.ArgumentParser(description="LTM Benchmark for HPC-QR Flow Manager")
    parser.add_argument(
        "--ltm-path", 
        type=str, 
        default="memory/stored/ltm",
        help="Path to LTM folders"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=50,
        help="Maximum number of files to process per folder"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--dynamic-scaling-factor",
        type=float,
        default=0.5,
        help="Dynamic scaling factor for shock absorption"
    )
    parser.add_argument(
        "--update-threshold",
        type=float,
        default=0.05,
        help="Update threshold for shock absorption"
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.3,
        help="Drift threshold for embedding stability"
    )
    
    args = parser.parse_args()
    
    config = {
        'ltm_path': args.ltm_path,
        'batch_sizes': args.batch_sizes,
        'max_files': args.max_files,
        'output_dir': args.output_dir,
        'dynamic_scaling_factor': args.dynamic_scaling_factor,
        'update_threshold': args.update_threshold,
        'drift_threshold': args.drift_threshold,
    }
    
    benchmark = LTMBenchmark(config)
    await benchmark.run_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
