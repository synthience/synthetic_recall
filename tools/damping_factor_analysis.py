#!/usr/bin/env python
"""
Damping Factor Analysis for HPC-QR Flow Manager

This script tests and visualizes the damping factors applied during shock absorption
to provide insights into embedding surprise distributions.
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Any, Tuple, Optional

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture damping factors
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        "test_results/damping_factors.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("damping_analysis")

# Import the HPCQRFlowManager
from memory.lucidia_memory_system.core.integration.hpc_qr_flow_manager import HPCQRFlowManager


class DampingFactorLogger(logging.Handler):
    """Custom logging handler to capture damping factors"""
    
    def __init__(self):
        super().__init__()
        self.damping_data = []
        
    def emit(self, record):
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            if 'Applied damping factor:' in record.msg:
                try:
                    # Extract damping factor and surprise score
                    msg = record.msg
                    damping_str = msg.split('Applied damping factor:')[1].split('(')[0].strip()
                    surprise_str = msg.split('surprise score:')[1].split(')')[0].strip()
                    
                    damping = float(damping_str)
                    surprise = float(surprise_str)
                    
                    self.damping_data.append({
                        'damping_factor': damping,
                        'surprise_score': surprise,
                        'timestamp': time.time()
                    })
                except Exception as e:
                    pass
            
            if 'Damping factors - Min:' in record.msg:
                try:
                    # Extract batch damping statistics
                    msg = record.msg
                    min_val = float(msg.split('Min:')[1].split(',')[0].strip())
                    max_val = float(msg.split('Max:')[1].split(',')[0].strip())
                    mean_val = float(msg.split('Mean:')[1].split(',')[0].strip())
                    
                    # Extract application count
                    applied_str = msg.split('Applied to')[1].strip()
                    applied_count, total_count = map(int, applied_str.split('/'))
                    
                    self.damping_data.append({
                        'min_damping': min_val,
                        'max_damping': max_val,
                        'mean_damping': mean_val,
                        'applied_count': applied_count,
                        'total_count': total_count,
                        'timestamp': time.time(),
                        'is_batch': True
                    })
                except Exception as e:
                    pass


def generate_embedding_sets(count_per_set: int, num_sets: int, dim: int = 768) -> List[List[torch.Tensor]]:
    """
    Generate sets of test embeddings with different characteristics.
    
    Args:
        count_per_set: Number of embeddings per set
        num_sets: Number of different sets to generate
        dim: Dimensionality of embeddings
        
    Returns:
        List of embedding sets with different distributions
    """
    embedding_sets = []
    
    # Set 1: Random uniform embeddings
    uniform_set = []
    for _ in range(count_per_set):
        vec = torch.rand(dim)
        vec = vec / torch.norm(vec)
        uniform_set.append(vec)
    embedding_sets.append(uniform_set)
    
    # Set 2: Clustered embeddings (low variance)
    clustered_set = []
    center = torch.rand(dim)
    center = center / torch.norm(center)
    for _ in range(count_per_set):
        # Add small perturbation to center
        noise = torch.randn(dim) * 0.1
        vec = center + noise
        vec = vec / torch.norm(vec)
        clustered_set.append(vec)
    embedding_sets.append(clustered_set)
    
    # Set 3: Embeddings with drift (simulating concept drift)
    drift_set = []
    start_vec = torch.rand(dim)
    start_vec = start_vec / torch.norm(start_vec)
    end_vec = torch.rand(dim)
    end_vec = end_vec / torch.norm(end_vec)
    
    for i in range(count_per_set):
        # Interpolate between start and end vectors
        t = i / (count_per_set - 1)
        vec = (1 - t) * start_vec + t * end_vec
        # Add small noise
        noise = torch.randn(dim) * 0.05
        vec = vec + noise
        vec = vec / torch.norm(vec)
        drift_set.append(vec)
    embedding_sets.append(drift_set)
    
    # Set 4: Mixed random embeddings
    mixed_set = []
    for _ in range(count_per_set):
        # Generate with higher variance
        vec = torch.randn(dim)  # Normal distribution instead of uniform
        vec = vec / torch.norm(vec)
        mixed_set.append(vec)
    embedding_sets.append(mixed_set)
    
    return embedding_sets[:num_sets]


async def run_damping_analysis(embedding_sets: List[List[torch.Tensor]],
                             batch_size: int = 32,
                             dynamic_scaling_factors: List[float] = [0.1, 0.5, 1.0]):
    """
    Run analysis on how damping factors are applied to different embedding sets.
    
    Args:
        embedding_sets: List of embedding sets to analyze
        batch_size: Batch size for processing
        dynamic_scaling_factors: List of scaling factors to test
    
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    for i, embeddings in enumerate(embedding_sets):
        set_name = f"set_{i+1}"
        results[set_name] = {}
        
        for scaling_factor in dynamic_scaling_factors:
            # Setup custom logger to capture damping factors
            damping_logger = DampingFactorLogger()
            logging.getLogger('memory.lucidia_memory_system.core.integration.hpc_qr_flow_manager').addHandler(damping_logger)
            
            # Configure the HPCQRFlowManager
            config = {
                'batch_size': batch_size,
                'embedding_dim': embeddings[0].shape[0],
                'shock_absorption_enabled': True,
                'dynamic_scaling_factor': scaling_factor,
                'max_momentum_size': 10000,
                'update_threshold': 0.01,  # Lower threshold for better visualization
                'drift_threshold': 0.3,
                'debug_logging': True
            }
            
            # Initialize the manager
            manager = HPCQRFlowManager(config)
            
            logger.info(f"Testing set {i+1} with scaling factor {scaling_factor}")
            
            # Process embeddings
            try:
                batch_results = await manager.process_embedding_batch(embeddings)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue
                
            # Clean up
            await manager.close()
            
            # Store results
            results[set_name][f"scaling_{scaling_factor}"] = {
                'damping_data': damping_logger.damping_data,
                'config': config,
                'stats': manager.get_stats()
            }
            
            # Remove the custom handler
            logging.getLogger('memory.lucidia_memory_system.core.integration.hpc_qr_flow_manager').removeHandler(damping_logger)
    
    return results


def visualize_results(results, output_dir="test_results"):
    """
    Create visualizations of damping factor analysis.
    
    Args:
        results: Results from run_damping_analysis
        output_dir: Directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for set_name, set_results in results.items():
        for scaling_key, data in set_results.items():
            damping_data = data['damping_data']
            
            # Convert to pandas DataFrame for easier analysis
            df = pd.DataFrame(damping_data)
            
            if not df.empty:
                # Create plots directory
                plot_dir = os.path.join(output_dir, 'plots')
                os.makedirs(plot_dir, exist_ok=True)
                
                # Individual embeddings plot
                individual_df = df[df.get('is_batch', False) == False].copy()
                if not individual_df.empty and 'damping_factor' in individual_df.columns and 'surprise_score' in individual_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(individual_df['surprise_score'], individual_df['damping_factor'], alpha=0.6)
                    plt.title(f'{set_name} - {scaling_key} - Damping vs Surprise')
                    plt.xlabel('Surprise Score')
                    plt.ylabel('Damping Factor')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f'{set_name}_{scaling_key}_damping_vs_surprise.png'))
                    plt.close()
                
                # Batch statistics plot
                batch_df = df[df.get('is_batch', False) == True].copy()
                if not batch_df.empty and 'mean_damping' in batch_df.columns and 'applied_count' in batch_df.columns:
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    
                    # Plot mean damping
                    ax1.set_xlabel('Batch')
                    ax1.set_ylabel('Mean Damping Factor', color='tab:blue')
                    ax1.plot(batch_df.index, batch_df['mean_damping'], 'o-', color='tab:blue')
                    ax1.tick_params(axis='y', labelcolor='tab:blue')
                    
                    # Plot application ratio on secondary axis
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('Application Ratio', color='tab:red')
                    ax2.plot(batch_df.index, batch_df['applied_count'] / batch_df['total_count'], 
                             's-', color='tab:red')
                    ax2.tick_params(axis='y', labelcolor='tab:red')
                    
                    plt.title(f'{set_name} - {scaling_key} - Batch Statistics')
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f'{set_name}_{scaling_key}_batch_stats.png'))
                    plt.close()
                    
                # Save raw data
                df.to_csv(os.path.join(output_dir, f'{set_name}_{scaling_key}_damping_data.csv'), index=False)
    
    # Create summary report
    with open(os.path.join(output_dir, 'damping_analysis_summary.md'), 'w') as f:
        f.write("# Damping Factor Analysis Summary\n\n")
        
        for set_name, set_results in results.items():
            f.write(f"## {set_name}\n\n")
            
            for scaling_key, data in set_results.items():
                scaling_factor = scaling_key.split('_')[1]
                f.write(f"### Scaling Factor: {scaling_factor}\n\n")
                
                damping_data = data['damping_data']
                df = pd.DataFrame(damping_data)
                
                if not df.empty:
                    # Individual statistics
                    individual_df = df[df.get('is_batch', False) == False].copy()
                    if not individual_df.empty and 'damping_factor' in individual_df.columns:
                        f.write("#### Individual Embedding Statistics\n\n")
                        f.write(f"- Count: {len(individual_df)}\n")
                        f.write(f"- Min Damping: {individual_df['damping_factor'].min():.4f}\n")
                        f.write(f"- Max Damping: {individual_df['damping_factor'].max():.4f}\n")
                        f.write(f"- Mean Damping: {individual_df['damping_factor'].mean():.4f}\n")
                        f.write(f"- Median Damping: {individual_df['damping_factor'].median():.4f}\n\n")
                    
                    # Batch statistics
                    batch_df = df[df.get('is_batch', False) == True].copy()
                    if not batch_df.empty and 'mean_damping' in batch_df.columns:
                        f.write("#### Batch Statistics\n\n")
                        f.write(f"- Number of Batches: {len(batch_df)}\n")
                        f.write(f"- Average Mean Damping: {batch_df['mean_damping'].mean():.4f}\n")
                        
                        if 'applied_count' in batch_df.columns and 'total_count' in batch_df.columns:
                            application_ratio = batch_df['applied_count'].sum() / batch_df['total_count'].sum()
                            f.write(f"- Overall Application Ratio: {application_ratio:.2%}\n\n")
                
                f.write(f"![Damping vs Surprise](plots/{set_name}_{scaling_key}_damping_vs_surprise.png)\n\n")
                f.write(f"![Batch Statistics](plots/{set_name}_{scaling_key}_batch_stats.png)\n\n")
                
                f.write("---\n\n")


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Damping Factor Analysis for HPC-QR Flow Manager")
    parser.add_argument(
        '--embedding-count', 
        type=int, 
        default=500,
        help='Number of embeddings per set'
    )
    parser.add_argument(
        '--num-sets', 
        type=int, 
        default=4,
        help='Number of different embedding sets to test'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--scaling-factors', 
        type=float, 
        nargs='+', 
        default=[0.1, 0.5, 1.0],
        help='List of dynamic scaling factors to test'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='test_results',
        help='Output directory for results'
    )
    
    return parser.parse_args()


async def main():
    """
    Main entry point.
    """
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate embedding sets
    embedding_sets = generate_embedding_sets(
        count_per_set=args.embedding_count,
        num_sets=args.num_sets
    )
    
    # Run analysis
    results = await run_damping_analysis(
        embedding_sets=embedding_sets,
        batch_size=args.batch_size,
        dynamic_scaling_factors=args.scaling_factors
    )
    
    # Visualize results
    visualize_results(results, args.output_dir)
    
    # Save full results
    with open(os.path.join(args.output_dir, 'damping_analysis_results.json'), 'w') as f:
        # Convert non-serializable objects to strings
        serializable_results = {}
        for set_name, set_results in results.items():
            serializable_results[set_name] = {}
            for scaling_key, data in set_results.items():
                serializable_results[set_name][scaling_key] = {
                    'damping_data': data['damping_data'],
                    'config': {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v 
                               for k, v in data['config'].items()},
                    'stats': {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v 
                              for k, v in data['stats'].items()}
                }
        
        json.dump({
            'timestamp': time.time(),
            'results': serializable_results,
            'args': vars(args),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }, f, indent=2)
    
    logger.info(f"Analysis completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
