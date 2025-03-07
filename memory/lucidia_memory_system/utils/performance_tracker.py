"""
LUCID RECALL PROJECT
Performance Tracker

Monitors system performance, tracks metrics, and provides analytics
for identifying bottlenecks.
"""

import time
import asyncio
import logging
import statistics
from typing import Dict, Any, List, Optional, Callable, Coroutine, TypeVar, Union
from collections import defaultdict, deque
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Type variable for function return values

class PerformanceTracker:
    """
    Performance tracking and monitoring utility.
    
    Features:
    - Operation timing
    - Rate limiting
    - Bottleneck detection
    - Performance analytics
    - Memory operation profiling
    """
    
    def __init__(self, 
                 history_size: int = 100, 
                 alert_threshold: float = 2.0, 
                 debug: bool = False):
        """
        Initialize the performance tracker.
        
        Args:
            history_size: Number of recent operations to track
            alert_threshold: Multiplier for average time to trigger alerts
            debug: Whether to log detailed debug information
        """
        self.history_size = history_size
        self.alert_threshold = alert_threshold
        self.debug = debug
        
        # Track operation times by category
        self.operations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.history_size))
        
        # Track ongoing operations
        self.ongoing_operations: Dict[str, Dict[str, float]] = {}
        
        # Global stats
        self.stats = {
            'total_operations': 0,
            'slow_operations': 0,
            'failed_operations': 0,
            'start_time': time.time()
        }
        
        # Performance report data
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.operation_times: Dict[str, float] = defaultdict(float)
        self.operation_failures: Dict[str, int] = defaultdict(int)
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info(f"Performance tracker initialized with history_size={history_size}")
    
    async def record_operation(self, 
                             operation: str, 
                             duration: float, 
                             success: bool = True, 
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an operation's performance.
        
        Args:
            operation: Name/category of operation
            duration: Time taken in seconds
            success: Whether operation succeeded
            metadata: Optional additional data
        """
        async with self._lock:
            # Update global stats
            self.stats['total_operations'] += 1
            if not success:
                self.stats['failed_operations'] += 1
            
            # Update operation-specific stats
            self.operation_counts[operation] += 1
            self.operation_times[operation] += duration
            if not success:
                self.operation_failures[operation] += 1
            
            # Create operation record
            record = {
                'duration': duration,
                'timestamp': time.time(),
                'success': success,
                'metadata': metadata or {}
            }
            
            # Add to history
            self.operations[operation].append(record)
            
            # Check if slow
            avg_time = self._get_average_time(operation)
            if avg_time > 0 and duration > avg_time * self.alert_threshold:
                self.stats['slow_operations'] += 1
                if self.debug:
                    logger.warning(f"Slow operation detected: {operation} took {duration:.3f}s "
                                 f"(avg: {avg_time:.3f}s)")
    
    async def start_operation(self, operation: str, 
                            op_id: Optional[str] = None) -> str:
        """
        Start tracking an operation's time.
        
        Args:
            operation: Name/category of operation
            op_id: Optional operation ID for correlation
            
        Returns:
            Operation ID for stopping
        """
        op_id = op_id or f"{operation}_{int(time.time() * 1000)}"
        
        async with self._lock:
            self.ongoing_operations[op_id] = {
                'operation': operation,
                'start_time': time.time()
            }
            
            if self.debug:
                logger.debug(f"Started tracking operation: {operation} (ID: {op_id})")
                
        return op_id
    
    async def stop_operation(self, op_id: str, success: bool = True,
                           metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        Stop tracking an operation and record its performance.
        
        Args:
            op_id: Operation ID from start_operation
            success: Whether operation succeeded
            metadata: Optional additional data
            
        Returns:
            Duration in seconds or -1 if operation wasn't tracked
        """
        if op_id not in self.ongoing_operations:
            logger.warning(f"Operation {op_id} not found in ongoing operations")
            return -1
            
        async with self._lock:
            # Get operation data
            op_data = self.ongoing_operations.pop(op_id)
            operation = op_data['operation']
            start_time = op_data['start_time']
            
            # Calculate duration
            end_time = time.time()
            duration = end_time - start_time
            
            # Record the operation
            await self.record_operation(operation, duration, success, metadata)
            
            if self.debug:
                logger.debug(f"Completed operation: {operation} in {duration:.3f}s (success: {success})")
                
            return duration
    
    @asynccontextmanager
    async def track_operation(self, operation: str) -> None:
        """
        Context manager for tracking operation time.
        
        Usage:
        ```
        async with performance_tracker.track_operation("db_query"):
            result = await db.query(...)
        ```
        
        Args:
            operation: Name/category of operation
        """
        # Start operation timing
        op_id = await self.start_operation(operation)
        success = True
        
        try:
            # Yield control back to the context block
            yield
        except Exception as e:
            # Mark as failed on exception
            success = False
            raise
        finally:
            # Record operation time
            await self.stop_operation(op_id, success)
    
    async def timed_execution(self, operation: str, func: Callable[..., Coroutine[Any, Any, T]], 
                            *args, **kwargs) -> T:
        """
        Execute a coroutine function with timing.
        
        Usage:
        ```
        result = await performance_tracker.timed_execution(
            "db_query", db.query, "SELECT * FROM table"
        )
        ```
        
        Args:
            operation: Name/category of operation
            func: Coroutine function to execute
            *args: Arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of the function
        """
        # Start operation timing
        op_id = await self.start_operation(operation)
        success = True
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            # Mark as failed on exception
            success = False
            raise
        finally:
            # Record operation time
            await self.stop_operation(op_id, success)
    
    def _get_average_time(self, operation: str) -> float:
        """
        Get average execution time for an operation.
        
        Args:
            operation: Name/category of operation
            
        Returns:
            Average execution time in seconds
        """
        if operation not in self.operations or not self.operations[operation]:
            return 0.0
            
        # Calculate average duration
        durations = [record['duration'] for record in self.operations[operation]]
        return sum(durations) / len(durations)
    
    def _get_percentile_time(self, operation: str, percentile: float = 95) -> float:
        """
        Get percentile execution time for an operation.
        
        Args:
            operation: Name/category of operation
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile execution time in seconds
        """
        if operation not in self.operations or not self.operations[operation]:
            return 0.0
            
        # Calculate percentile duration
        durations = [record['duration'] for record in self.operations[operation]]
        
        try:
            return statistics.quantiles(durations, n=100)[int(percentile) - 1]
        except (ValueError, IndexError):
            # Fall back to simple calculation for small samples
            durations.sort()
            idx = int((percentile / 100) * len(durations))
            return durations[idx - 1] if idx > 0 else durations[0]
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Returns:
            Dict with performance metrics
        """
        async with self._lock:
            # Calculate stats for each operation
            operation_stats = {}
            
            for operation in self.operations:
                # Skip operations with no records
                if not self.operations[operation]:
                    continue
                    
                # Get durations
                durations = [record['duration'] for record in self.operations[operation]]
                
                # Calculate statistics
                try:
                    if len(durations) >= 2:
                        percentiles = statistics.quantiles(durations, n=4)
                        p25, p50, p75 = percentiles
                        p95 = self._get_percentile_time(operation, 95)
                        p99 = self._get_percentile_time(operation, 99)
                    else:
                        p25 = p50 = p75 = p95 = p99 = durations[0] if durations else 0
                        
                    operation_stats[operation] = {
                        'count': len(self.operations[operation]),
                        'avg_time': sum(durations) / len(durations),
                        'min_time': min(durations),
                        'max_time': max(durations),
                        'p25': p25,
                        'p50': p50,
                        'p75': p75,
                        'p95': p95,
                        'p99': p99,
                        'success_rate': sum(1 for r in self.operations[operation] if r['success']) / len(self.operations[operation]),
                        'total_time': sum(durations)
                    }
                except (ValueError, IndexError, statistics.StatisticsError):
                    # Fall back to simple stats for small samples
                    operation_stats[operation] = {
                        'count': len(self.operations[operation]),
                        'avg_time': sum(durations) / max(1, len(durations)),
                        'min_time': min(durations) if durations else 0,
                        'max_time': max(durations) if durations else 0,
                        'success_rate': sum(1 for r in self.operations[operation] if r['success']) / max(1, len(self.operations[operation])),
                        'total_time': sum(durations)
                    }
            
            # Calculate global stats
            uptime = time.time() - self.stats['start_time']
            total_operations = self.stats['total_operations']
            ops_per_second = total_operations / uptime if uptime > 0 else 0
            
            # Identify potential bottlenecks
            bottlenecks = []
            if operation_stats:
                # Sort operations by total time spent
                sorted_by_time = sorted(
                    operation_stats.items(), 
                    key=lambda x: x[1]['total_time'], 
                    reverse=True
                )
                
                # Top 3 operations by time
                top_by_time = sorted_by_time[:3]
                
                # Add to bottlenecks if they take more than 10% of total time
                total_time = sum(op['total_time'] for _, op in operation_stats.items())
                if total_time > 0:
                    for operation, stats in top_by_time:
                        time_percentage = (stats['total_time'] / total_time) * 100
                        if time_percentage > 10:
                            bottlenecks.append({
                                'operation': operation,
                                'time_percentage': time_percentage,
                                'avg_time': stats['avg_time'],
                                'count': stats['count']
                            })
            
            # Compile full report
            report = {
                'global_stats': {
                    'uptime': uptime,
                    'total_operations': total_operations,
                    'operations_per_second': ops_per_second,
                    'slow_operations': self.stats['slow_operations'],
                    'failed_operations': self.stats['failed_operations'],
                    'failure_rate': self.stats['failed_operations'] / max(1, total_operations)
                },
                'operation_stats': operation_stats,
                'bottlenecks': bottlenecks,
                'ongoing_operations': len(self.ongoing_operations)
            }
            
            return report
    
    async def reset_stats(self) -> None:
        """Reset all statistics."""
        async with self._lock:
            # Clear operation histories
            self.operations.clear()
            self.operations = defaultdict(lambda: deque(maxlen=self.history_size))
            
            # Reset global stats
            self.stats = {
                'total_operations': 0,
                'slow_operations': 0,
                'failed_operations': 0,
                'start_time': time.time()
            }
            
            # Reset operation tracking
            self.operation_counts.clear()
            self.operation_times.clear()
            self.operation_failures.clear()
            
            logger.info("Performance tracker stats reset")
    
    async def get_operation_history(self, operation: str) -> List[Dict[str, Any]]:
        """
        Get history for a specific operation.
        
        Args:
            operation: Name/category of operation
            
        Returns:
            List of operation records
        """
        async with self._lock:
            if operation not in self.operations:
                return []
                
            return list(self.operations[operation])