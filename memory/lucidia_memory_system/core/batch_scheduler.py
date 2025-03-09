"""
Adaptive batch scheduling for Lucidia's HPC and tensor server interactions.

Provides efficient batching mechanisms that adapt to server performance and load,
optimizing throughput while preventing overload.
"""

import time
import asyncio
import logging
import collections
from typing import Dict, Any, Optional, List, Tuple


class AdaptiveHPCBatchScheduler:
    """Dynamically adjusts batch sizes based on HPC performance."""
    
    def __init__(self, min_batch=5, max_batch=50, target_latency_ms=250, 
                 warmup_batches=5, adjustment_rate=0.2):
        """Initialize the adaptive batch scheduler.
        
        Args:
            min_batch: Minimum batch size
            max_batch: Maximum batch size
            target_latency_ms: Target processing latency in milliseconds
            warmup_batches: Number of batches to process before adjusting size
            adjustment_rate: How quickly to adjust batch size (0.0-1.0)
        """
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.target_latency = target_latency_ms / 1000  # Convert to seconds
        self.current_batch_size = min_batch
        self.recent_latencies = collections.deque(maxlen=20)
        self.recent_throughputs = collections.deque(maxlen=10)
        self.processed_batches = 0
        self.adjustment_rate = adjustment_rate
        self.warmup_batches = warmup_batches
        self.logger = logging.getLogger(__name__)
        
    def record_performance(self, batch_size: int, processing_time: float, queue_size: int = None):
        """Record performance metrics for a processed batch.
        
        Args:
            batch_size: Number of items in the batch
            processing_time: Time taken to process the batch in seconds
            queue_size: Current size of the queue (optional)
        """
        if batch_size == 0:
            return
            
        self.processed_batches += 1
        latency_per_item = processing_time / batch_size
        self.recent_latencies.append(latency_per_item)
        
        throughput = batch_size / processing_time
        self.recent_throughputs.append(throughput)
        
        # Only adjust after warmup period
        if self.processed_batches <= self.warmup_batches:
            self.logger.debug(f"Warming up batch scheduler: {self.processed_batches}/{self.warmup_batches} batches")
            return
            
        self._adjust_batch_size(queue_size)
        
    def _adjust_batch_size(self, queue_size: int = None):
        """Dynamically adjust batch size based on performance.
        
        Args:
            queue_size: Current size of the queue (optional)
        """
        if not self.recent_latencies:
            return
            
        avg_latency = sum(self.recent_latencies) / len(self.recent_latencies)
        
        # Calculate adjustment factor based on latency
        latency_ratio = self.target_latency / avg_latency
        adjustment = (latency_ratio - 1) * self.adjustment_rate
        
        # Additional adjustments based on queue size
        if queue_size is not None:
            if queue_size > self.current_batch_size * 3:
                # Queue building up, increase batch size more aggressively
                adjustment = max(adjustment, 0.1)  # Ensure positive adjustment
                self.logger.debug(f"Queue building up ({queue_size} items), increasing batch size")
            elif queue_size < self.current_batch_size / 2 and queue_size > 0:
                # Queue draining fast, be more conservative
                adjustment = min(adjustment, 0)  # Cap at zero (don't increase)
                self.logger.debug(f"Queue draining ({queue_size} items), maintaining batch size")
            
        # Apply adjustment with smoothing
        new_size = self.current_batch_size * (1 + adjustment)
        new_size = min(self.max_batch, max(self.min_batch, round(new_size)))
        
        if new_size != self.current_batch_size:
            self.logger.info(f"Adjusting batch size: {self.current_batch_size} -> {new_size} "
                           f"(avg_latency={avg_latency*1000:.1f}ms, target={self.target_latency*1000:.1f}ms)")
            
        self.current_batch_size = new_size
        
    def get_current_batch_size(self) -> int:
        """Get the current recommended batch size.
        
        Returns:
            Current optimal batch size
        """
        return self.current_batch_size
        
    def get_optimal_batch_size(self, queue_size: int) -> int:
        """Get the optimal batch size based on queue size and current performance.
        
        Args:
            queue_size: Size of the queue to be processed
            
        Returns:
            Optimal batch size for the given queue
        """
        # Use current batch size as baseline
        optimal_size = self.current_batch_size
        
        # For very small queues, process all at once if under min_batch
        if queue_size <= self.min_batch:
            return max(1, queue_size)
            
        # For larger queues, use current batch size with some adjustments
        if queue_size > self.current_batch_size * 3:
            # If queue is building up quickly, use a larger batch size
            optimal_size = min(self.max_batch, int(self.current_batch_size * 1.5))
        
        # Ensure we don't exceed queue size
        return min(queue_size, optimal_size)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            "current_batch_size": self.current_batch_size,
            "processed_batches": self.processed_batches,
            "min_batch": self.min_batch,
            "max_batch": self.max_batch,
            "target_latency_ms": self.target_latency * 1000,
        }
        
        if self.recent_latencies:
            avg_latency = sum(self.recent_latencies) / len(self.recent_latencies)
            metrics["avg_latency_ms"] = avg_latency * 1000
        
        if self.recent_throughputs:
            avg_throughput = sum(self.recent_throughputs) / len(self.recent_throughputs)
            metrics["avg_throughput"] = avg_throughput
            
        return metrics
        
    def update_metrics(self, processed_items: int, batch_size: int):
        """Update metrics after processing a batch.
        
        This is a simplified version of record_performance that doesn't
        require timing information.
        
        Args:
            processed_items: Number of items processed
            batch_size: Size of the batch used
        """
        self.processed_batches += 1
        # Since we don't have timing info, just record that we processed this batch
        # without adjusting the batch size
        
    def calculate_optimal_timeout(self, queue_size: int) -> float:
        """Calculate optimal timeout for batch collection based on queue size.
        
        Args:
            queue_size: Current queue size
            
        Returns:
            Timeout in seconds
        """
        target_size = self.current_batch_size
        
        if queue_size == 0:
            return 0.05  # Short timeout for empty queue
            
        # Ratio of current queue to target batch size
        ratio = queue_size / target_size
        
        if ratio >= 2.0:
            # Queue has plenty of items, minimal timeout
            return 0.01
        elif ratio >= 1.0:
            # Queue has enough for a batch, short timeout
            return 0.05
        elif ratio >= 0.5:
            # Queue has half a batch, medium timeout
            return 0.1
        else:
            # Queue is relatively empty, longer timeout
            return 0.2
            
    async def collect_batch(self, queue: asyncio.Queue, max_wait: float = 0.5) -> List[Any]:
        """Collect an optimally-sized batch of items from the queue.
        
        Args:
            queue: AsyncIO queue to collect items from
            max_wait: Maximum time to wait for batch completion
            
        Returns:
            List of queue items collected into a batch
        """
        target_size = self.current_batch_size
        batch = []
        start_time = time.time()
        
        # Initial queue size check
        queue_size = queue.qsize()
        timeout = min(max_wait, self.calculate_optimal_timeout(queue_size))
        
        # Collect batch with adaptive timeout
        while len(batch) < target_size and (time.time() - start_time) < max_wait:
            try:
                # Update timeout for each item based on current progress
                remaining_items = target_size - len(batch)
                elapsed = time.time() - start_time
                remaining_time = max(0.01, max_wait - elapsed)  # At least 10ms
                
                # Shorter timeout as we approach target size
                completion_ratio = len(batch) / target_size if target_size > 0 else 0
                adjusted_timeout = min(remaining_time, timeout * (1 - completion_ratio * 0.5))
                
                item = await asyncio.wait_for(queue.get(), timeout=adjusted_timeout)
                batch.append(item)
                queue.task_done()
                
                # Recalculate queue size for next timeout
                if len(batch) % 5 == 0:  # Check every 5 items
                    queue_size = queue.qsize()
                    if queue_size == 0 and len(batch) >= self.min_batch:
                        # Queue empty and we have minimum batch size
                        break
                    
            except asyncio.TimeoutError:
                # Timeout occurred, check if we have minimum batch size
                if len(batch) >= self.min_batch:
                    break
                elif time.time() - start_time >= max_wait:
                    # Max wait time exceeded
                    break
                    
        # Log batch collection performance
        collection_time = time.time() - start_time
        if batch:
            self.logger.debug(f"Collected batch of {len(batch)} items in {collection_time*1000:.1f}ms")
            
        return batch
