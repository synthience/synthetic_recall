import time
import json
import asyncio
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    A class for tracking and recording performance metrics for the memory middleware components.
    Provides insights into memory operations, latency, and resource utilization.
    """
    
    def __init__(self, output_dir: str = "./metrics", sampling_interval: int = 100):
        """
        Initialize the performance metrics tracker.
        
        Args:
            output_dir: Directory to store metrics data
            sampling_interval: How frequently to sample and record metrics (every N operations)
        """
        self.output_dir = output_dir
        self.sampling_interval = sampling_interval
        self.metrics = {
            "store_operation": [],
            "retrieve_operation": [],
            "session_load": [],
            "session_save": [],
            "memory_size": []
        }
        self.operation_count = 0
        self._metrics_lock = asyncio.Lock()
        
        # Ensure metrics directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    async def record_operation(self, operation_type: str, duration_ms: float, metadata: Dict[str, Any]):
        """
        Record a memory operation with its duration and metadata.
        
        Args:
            operation_type: Type of operation (store, retrieve, load, save)
            duration_ms: Duration of the operation in milliseconds
            metadata: Additional information about the operation
        """
        async with self._metrics_lock:
            if operation_type in self.metrics:
                timestamp = datetime.now().isoformat()
                self.metrics[operation_type].append({
                    "timestamp": timestamp,
                    "duration_ms": duration_ms,
                    **metadata
                })
                
                # Trim to keep only the last 1000 operations per type
                if len(self.metrics[operation_type]) > 1000:
                    self.metrics[operation_type] = self.metrics[operation_type][-1000:]
                
                self.operation_count += 1
                if self.operation_count % self.sampling_interval == 0:
                    await self.save_metrics()
    
    async def record_memory_size(self, session_id: str, history_size: int, byte_size: int):
        """
        Record the current memory size for a session.
        
        Args:
            session_id: ID of the session being measured
            history_size: Number of interactions in the history
            byte_size: Approximate size in bytes of the session data
        """
        async with self._metrics_lock:
            timestamp = datetime.now().isoformat()
            self.metrics["memory_size"].append({
                "timestamp": timestamp,
                "session_id": session_id,
                "history_size": history_size,
                "byte_size": byte_size
            })
            
            # Trim to keep only the last 1000 size records
            if len(self.metrics["memory_size"]) > 1000:
                self.metrics["memory_size"] = self.metrics["memory_size"][-1000:]
    
    async def save_metrics(self):
        """
        Save the current metrics to disk.
        """
        try:
            filename = os.path.join(self.output_dir, f"memory_metrics_{datetime.now().strftime('%Y%m%d')}.json")
            async with self._metrics_lock:
                # Generate summary statistics
                summary = self._generate_summary()
                output_data = {
                    "summary": summary,
                    "detailed_metrics": self.metrics
                }
                
                with open(filename, 'w') as f:
                    json.dump(output_data, f, indent=2)
                    
                logger.debug(f"Saved memory metrics to {filename}")
                return True
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            return False
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics from the collected metrics.
        """
        summary = {}
        
        for op_type, data in self.metrics.items():
            if not data:
                summary[op_type] = {"count": 0}
                continue
                
            durations = [entry.get("duration_ms", 0) for entry in data if "duration_ms" in entry]
            
            if durations:
                summary[op_type] = {
                    "count": len(data),
                    "avg_duration_ms": sum(durations) / len(durations),
                    "min_duration_ms": min(durations),
                    "max_duration_ms": max(durations),
                    "latest_timestamp": data[-1].get("timestamp", "")
                }
            else:
                summary[op_type] = {"count": len(data)}
                
            # Add special metrics for memory size
            if op_type == "memory_size" and data:
                history_sizes = [entry.get("history_size", 0) for entry in data if "history_size" in entry]
                byte_sizes = [entry.get("byte_size", 0) for entry in data if "byte_size" in entry]
                
                if history_sizes:
                    summary[op_type]["avg_history_size"] = sum(history_sizes) / len(history_sizes)
                    summary[op_type]["max_history_size"] = max(history_sizes)
                
                if byte_sizes:
                    summary[op_type]["avg_byte_size"] = sum(byte_sizes) / len(byte_sizes)
                    summary[op_type]["max_byte_size"] = max(byte_sizes)
                    summary[op_type]["total_memory_mb"] = sum(byte_sizes) / (1024 * 1024)
        
        return summary

class MetricsDecorator:
    """
    Decorator utility for easily adding performance metrics to memory operations.
    """
    
    def __init__(self, metrics: PerformanceMetrics):
        """
        Initialize the metrics decorator.
        
        Args:
            metrics: PerformanceMetrics instance to record the metrics
        """
        self.metrics = metrics
    
    def track_operation(self, operation_type: str, **metadata_kwargs):
        """
        Decorator for tracking operation performance.
        
        Args:
            operation_type: Type of operation being tracked
            **metadata_kwargs: Additional metadata to record
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Extract dynamic metadata if provided as callback
                    metadata = {}
                    for key, value_or_callback in metadata_kwargs.items():
                        if callable(value_or_callback):
                            try:
                                metadata[key] = value_or_callback(result, *args, **kwargs)
                            except Exception as e:
                                logger.error(f"Error extracting metric metadata {key}: {e}")
                                metadata[key] = None
                        else:
                            metadata[key] = value_or_callback
                    
                    # Add result status
                    metadata["success"] = True
                    
                    await self.metrics.record_operation(operation_type, duration_ms, metadata)
                    return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Record failure metrics
                    metadata = {**metadata_kwargs, "success": False, "error": str(e)}
                    await self.metrics.record_operation(operation_type, duration_ms, metadata)
                    raise
            return wrapper
        return decorator

# Convenience function to get approximate byte size of an object
def get_approximate_size(obj: Any) -> int:
    """
    Get the approximate size of an object in bytes.
    
    Args:
        obj: Object to measure
    
    Returns:
        Approximate size in bytes
    """
    try:
        return len(json.dumps(obj).encode('utf-8'))
    except (TypeError, OverflowError):
        return 0
