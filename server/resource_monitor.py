"""Resource Monitor for Lucidia

This module provides enhanced system resource tracking and optimization
for dynamically adjusting system behavior based on available resources.
"""

import os
import time
import logging
import asyncio
import psutil
import json
import platform
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path

# Import model selector for system state updates
from .model_selector import ModelSelector, SystemState

logger = logging.getLogger("ResourceMonitor")

class ResourceMetrics:
    """Container for system resource metrics and statistics"""
    
    def __init__(self):
        """Initialize resource metrics"""
        self.cpu_usage = []  # List of recent CPU usage percentages
        self.memory_usage = []  # List of recent memory usage percentages
        self.disk_usage = []  # List of recent disk usage percentages
        self.network_io = []  # List of recent network IO stats
        self.process_stats = {}  # Stats for key processes
        self.gpu_usage = []  # List of recent GPU usage if available
        
        # Metrics history (hourly, daily averages)
        self.hourly_metrics = {}
        self.daily_metrics = {}
        
        # Performance metrics
        self.response_times = {}  # Model -> list of response times
        self.token_rates = {}  # Model -> tokens per second
        
        # Resource allocation
        self.allocated_memory = {}  # Component -> allocated memory
        self.allocated_cpu = {}  # Component -> allocated CPU percentage
        
        # Maximum history length to prevent unbounded growth
        self.max_history = 60  # Keep up to 60 data points (1 hour at 1 min intervals)

    def add_cpu_usage(self, usage: float) -> None:
        """Add CPU usage data point"""
        self.cpu_usage.append(usage)
        if len(self.cpu_usage) > self.max_history:
            self.cpu_usage.pop(0)
    
    def add_memory_usage(self, usage: float) -> None:
        """Add memory usage data point"""
        self.memory_usage.append(usage)
        if len(self.memory_usage) > self.max_history:
            self.memory_usage.pop(0)
    
    def add_disk_usage(self, usage: float) -> None:
        """Add disk usage data point"""
        self.disk_usage.append(usage)
        if len(self.disk_usage) > self.max_history:
            self.disk_usage.pop(0)
    
    def add_network_io(self, sent: int, received: int) -> None:
        """Add network IO data point"""
        self.network_io.append({"sent": sent, "received": received, "time": time.time()})
        if len(self.network_io) > self.max_history:
            self.network_io.pop(0)
    
    def add_gpu_usage(self, usage: float) -> None:
        """Add GPU usage data point if GPU is available"""
        self.gpu_usage.append(usage)
        if len(self.gpu_usage) > self.max_history:
            self.gpu_usage.pop(0)
    
    def update_process_stats(self, process_name: str, cpu: float, memory: float) -> None:
        """Update statistics for a specific process"""
        self.process_stats[process_name] = {
            "cpu": cpu,
            "memory": memory,
            "updated_at": time.time()
        }
    
    def add_response_time(self, model: str, response_time: float) -> None:
        """Add response time for a specific model"""
        if model not in self.response_times:
            self.response_times[model] = []
        
        self.response_times[model].append(response_time)
        # Keep only last 20 response times per model
        if len(self.response_times[model]) > 20:
            self.response_times[model].pop(0)
    
    def update_token_rate(self, model: str, tokens: int, seconds: float) -> None:
        """Update tokens per second rate for a model"""
        if seconds > 0:
            tokens_per_second = tokens / seconds
            if model not in self.token_rates:
                self.token_rates[model] = []
            
            self.token_rates[model].append(tokens_per_second)
            # Keep only last 20 token rates per model
            if len(self.token_rates[model]) > 20:
                self.token_rates[model].pop(0)
    
    def get_avg_cpu_usage(self, window: int = None) -> float:
        """Get average CPU usage over the specified window"""
        if not self.cpu_usage:
            return 0.0
        window = min(window or len(self.cpu_usage), len(self.cpu_usage))
        return sum(self.cpu_usage[-window:]) / window
    
    def get_avg_memory_usage(self, window: int = None) -> float:
        """Get average memory usage over the specified window"""
        if not self.memory_usage:
            return 0.0
        window = min(window or len(self.memory_usage), len(self.memory_usage))
        return sum(self.memory_usage[-window:]) / window
    
    def get_avg_response_time(self, model: str) -> float:
        """Get average response time for a model"""
        if model not in self.response_times or not self.response_times[model]:
            return 0.0
        return sum(self.response_times[model]) / len(self.response_times[model])
    
    def get_avg_token_rate(self, model: str) -> float:
        """Get average tokens per second for a model"""
        if model not in self.token_rates or not self.token_rates[model]:
            return 0.0
        return sum(self.token_rates[model]) / len(self.token_rates[model])
    
    def summarize_hourly_metrics(self) -> None:
        """Calculate and store hourly metrics"""
        hour_key = datetime.now().strftime("%Y-%m-%d-%H")
        self.hourly_metrics[hour_key] = {
            "cpu_avg": self.get_avg_cpu_usage(),
            "memory_avg": self.get_avg_memory_usage(),
            "disk_avg": sum(self.disk_usage) / len(self.disk_usage) if self.disk_usage else 0,
            "response_times": {model: self.get_avg_response_time(model) for model in self.response_times},
            "token_rates": {model: self.get_avg_token_rate(model) for model in self.token_rates},
            "timestamp": time.time()
        }
        
        # Keep only last 24 hours of hourly metrics
        if len(self.hourly_metrics) > 24:
            oldest_key = min(self.hourly_metrics.keys())
            del self.hourly_metrics[oldest_key]
    
    def summarize_daily_metrics(self) -> None:
        """Calculate and store daily metrics from hourly metrics"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_hours = [k for k in self.hourly_metrics if k.startswith(today)]
        
        if today_hours:
            cpu_avgs = [self.hourly_metrics[h]["cpu_avg"] for h in today_hours]
            memory_avgs = [self.hourly_metrics[h]["memory_avg"] for h in today_hours]
            disk_avgs = [self.hourly_metrics[h]["disk_avg"] for h in today_hours]
            
            self.daily_metrics[today] = {
                "cpu_avg": sum(cpu_avgs) / len(cpu_avgs),
                "memory_avg": sum(memory_avgs) / len(memory_avgs),
                "disk_avg": sum(disk_avgs) / len(disk_avgs),
                "hours_recorded": len(today_hours),
                "timestamp": time.time()
            }
            
            # Keep only last 30 days of daily metrics
            if len(self.daily_metrics) > 30:
                oldest_key = min(self.daily_metrics.keys())
                del self.daily_metrics[oldest_key]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization"""
        return {
            "current": {
                "cpu": self.cpu_usage[-1] if self.cpu_usage else 0,
                "memory": self.memory_usage[-1] if self.memory_usage else 0,
                "disk": self.disk_usage[-1] if self.disk_usage else 0,
                "gpu": self.gpu_usage[-1] if self.gpu_usage else 0,
                "network": self.network_io[-1] if self.network_io else {"sent": 0, "received": 0}
            },
            "averages": {
                "cpu_avg_5min": self.get_avg_cpu_usage(5),
                "cpu_avg_15min": self.get_avg_cpu_usage(15),
                "cpu_avg_60min": self.get_avg_cpu_usage(),
                "memory_avg_5min": self.get_avg_memory_usage(5),
                "memory_avg_15min": self.get_avg_memory_usage(15),
                "memory_avg_60min": self.get_avg_memory_usage()
            },
            "process_stats": self.process_stats,
            "model_performance": {
                "response_times": {model: self.get_avg_response_time(model) for model in self.response_times},
                "token_rates": {model: self.get_avg_token_rate(model) for model in self.token_rates}
            },
            "hourly_metrics": self.hourly_metrics,
            "daily_metrics": self.daily_metrics,
            "resource_allocation": {
                "memory": self.allocated_memory,
                "cpu": self.allocated_cpu
            }
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load metrics from dictionary"""
        if "hourly_metrics" in data:
            self.hourly_metrics = data["hourly_metrics"]
        if "daily_metrics" in data:
            self.daily_metrics = data["daily_metrics"]
        if "process_stats" in data:
            self.process_stats = data["process_stats"]
        if "resource_allocation" in data:
            if "memory" in data["resource_allocation"]:
                self.allocated_memory = data["resource_allocation"]["memory"]
            if "cpu" in data["resource_allocation"]:
                self.allocated_cpu = data["resource_allocation"]["cpu"]


class ResourceOptimizer:
    """Optimizes resource allocation based on usage patterns and priorities"""
    
    def __init__(self, metrics: ResourceMetrics):
        """Initialize optimizer with resource metrics"""
        self.metrics = metrics
        self.component_priorities = {
            "memory_system": 10,
            "llm_service": 9,
            "dream_processor": 8,
            "knowledge_graph": 7,
            "reflection_engine": 6,
            "embedding_service": 5,
            "api_server": 4
        }
        self.target_cpu_usage = 80.0  # Target CPU usage percentage
        self.target_memory_usage = 80.0  # Target memory usage percentage
        self.minimum_allocations = {
            "memory_system": {"cpu": 5.0, "memory": 256 * 1024 * 1024},  # 256 MB
            "llm_service": {"cpu": 10.0, "memory": 512 * 1024 * 1024},   # 512 MB
            "dream_processor": {"cpu": 5.0, "memory": 128 * 1024 * 1024},  # 128 MB
            "knowledge_graph": {"cpu": 5.0, "memory": 256 * 1024 * 1024},  # 256 MB
            "reflection_engine": {"cpu": 5.0, "memory": 128 * 1024 * 1024},  # 128 MB
            "embedding_service": {"cpu": 5.0, "memory": 128 * 1024 * 1024},  # 128 MB
            "api_server": {"cpu": 5.0, "memory": 64 * 1024 * 1024}  # 64 MB
        }
        
    def allocate_resources(self, total_memory: int, cpu_count: int) -> Dict[str, Dict[str, Union[float, int]]]:
        """Allocate resources based on priorities and system capacity
        
        Args:
            total_memory: Total system memory in bytes
            cpu_count: Number of CPU cores
            
        Returns:
            Resource allocation by component
        """
        # Calculate available resources (target percentage of total)
        available_memory = total_memory * (self.target_memory_usage / 100.0)
        available_cpu = cpu_count * 100.0 * (self.target_cpu_usage / 100.0)  # CPU percentage across all cores
        
        # Sort components by priority (highest first)
        sorted_components = sorted(
            self.component_priorities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # First pass: allocate minimum resources
        allocated_memory = 0
        allocated_cpu = 0
        allocations = {}
        
        for component, _ in sorted_components:
            min_allocation = self.minimum_allocations.get(component, {"cpu": 5.0, "memory": 64 * 1024 * 1024})
            allocations[component] = {
                "memory": min_allocation["memory"],
                "cpu": min_allocation["cpu"]
            }
            allocated_memory += min_allocation["memory"]
            allocated_cpu += min_allocation["cpu"]
        
        # Second pass: allocate remaining resources proportionally to priorities
        remaining_memory = max(0, available_memory - allocated_memory)
        remaining_cpu = max(0, available_cpu - allocated_cpu)
        
        total_priority = sum(self.component_priorities.values())
        
        for component, priority in sorted_components:
            # Allocate based on priority weight
            priority_ratio = priority / total_priority
            additional_memory = remaining_memory * priority_ratio
            additional_cpu = remaining_cpu * priority_ratio
            
            allocations[component]["memory"] += additional_memory
            allocations[component]["cpu"] += additional_cpu
        
        # Update metrics with allocations
        self.metrics.allocated_memory = {k: v["memory"] for k, v in allocations.items()}
        self.metrics.allocated_cpu = {k: v["cpu"] for k, v in allocations.items()}
        
        return allocations
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for optimizing resource usage"""
        recommendations = {
            "reduce_memory_usage": False,
            "reduce_cpu_usage": False,
            "components_to_optimize": [],
            "suggested_actions": []
        }
        
        # Check if resources are constrained
        avg_cpu_15min = self.metrics.get_avg_cpu_usage(15)
        avg_memory_15min = self.metrics.get_avg_memory_usage(15)
        
        if avg_cpu_15min > 90.0:
            recommendations["reduce_cpu_usage"] = True
            recommendations["suggested_actions"].append("Reduce model complexity and batch size")
        
        if avg_memory_15min > 90.0:
            recommendations["reduce_memory_usage"] = True
            recommendations["suggested_actions"].append("Switch to smaller models or reduce parallel operations")
        
        # Find inefficient components
        for component, stats in self.metrics.process_stats.items():
            if stats["cpu"] > self.metrics.allocated_cpu.get(component, 100.0):
                recommendations["components_to_optimize"].append({
                    "name": component,
                    "issue": "cpu_overuse",
                    "current": stats["cpu"],
                    "allocated": self.metrics.allocated_cpu.get(component, 0)
                })
            
            component_memory_mb = stats["memory"] / (1024 * 1024)
            allocated_memory_mb = self.metrics.allocated_memory.get(component, 0) / (1024 * 1024)
            
            if component_memory_mb > allocated_memory_mb:
                recommendations["components_to_optimize"].append({
                    "name": component,
                    "issue": "memory_overuse",
                    "current_mb": component_memory_mb,
                    "allocated_mb": allocated_memory_mb
                })
        
        return recommendations


class GPUMonitor:
    """Monitors GPU usage if available"""
    
    def __init__(self):
        """Initialize GPU monitor"""
        self.gpu_available = False
        self.gpu_info = {}
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.gpu_count = torch.cuda.device_count()
                for i in range(self.gpu_count):
                    self.gpu_info[i] = {
                        "name": torch.cuda.get_device_name(i),
                        "memory_total": torch.cuda.get_device_properties(i).total_memory
                    }
            logger.info(f"GPU monitoring initialized, available: {self.gpu_available}")
        except (ImportError, Exception) as e:
            logger.warning(f"GPU monitoring not available: {e}")
    
    def get_gpu_usage(self) -> Dict[int, Dict[str, Any]]:
        """Get current GPU usage statistics"""
        if not self.gpu_available:
            return {}
        
        result = {}
        try:
            import torch
            for i in range(self.gpu_count):
                torch.cuda.set_device(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = self.gpu_info[i]["memory_total"]
                
                result[i] = {
                    "name": self.gpu_info[i]["name"],
                    "memory_allocated": memory_allocated,
                    "memory_reserved": memory_reserved,
                    "memory_total": memory_total,
                    "utilization_percent": (memory_allocated / memory_total) * 100.0
                }
        except Exception as e:
            logger.error(f"Error getting GPU usage: {e}")
        
        return result


class ResourceMonitor:
    """Enhanced system resource monitor with optimization capabilities"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'ResourceMonitor':
        """Get or create the singleton instance of ResourceMonitor"""
        if cls._instance is None:
            cls._instance = ResourceMonitor()
        return cls._instance
    
    def __init__(self):
        """Initialize the resource monitor"""
        self.metrics = ResourceMetrics()
        self.optimizer = ResourceOptimizer(self.metrics)
        self.gpu_monitor = GPUMonitor()
        
        self.selector = ModelSelector.get_instance()
        self.monitor_task = None
        self.running = False
        self.check_interval = 60  # Seconds between resource checks
        self.detail_interval = 300  # Seconds between detailed checks
        self.metrics_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "resource_metrics.json"
        )
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        
        # System info
        self.system_info = self._get_system_info()
        
        # Resource thresholds
        self.cpu_high_threshold = 80  # Percentage
        self.cpu_low_threshold = 40   # Percentage
        self.memory_high_threshold = 80  # Percentage
        self.memory_low_threshold = 40   # Percentage
        
        # Performance metrics for models
        self.model_metrics = {}
        
        # Initialize system state
        self.current_state = SystemState.IDLE
        
        # Auto-optimization settings
        self.auto_optimize = True
        self.optimization_interval = 3600  # 1 hour between auto-optimizations
        self.last_optimization_time = 0
        
        # Key process names to monitor
        self.processes_to_monitor = [
            "python", "llm_server", "embedding_server", "tensor_server"
        ]
        
        # Load previous metrics if available
        self._load_metrics()
        
        logger.info(f"Enhanced ResourceMonitor initialized on {self.system_info['platform']}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_physical_count": psutil.cpu_count(logical=False),
            "memory_total": psutil.virtual_memory().total,
            "hostname": platform.node(),
            "gpu_available": self.gpu_monitor.gpu_available
        }
        
        if self.gpu_monitor.gpu_available:
            info["gpu_info"] = self.gpu_monitor.gpu_info
        
        return info
    
    def start(self):
        """Start resource monitoring"""
        if not self.running:
            self.running = True
            self.monitor_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started enhanced resource monitoring")
    
    def stop(self):
        """Stop resource monitoring"""
        if self.running:
            self.running = False
            if self.monitor_task and not self.monitor_task.done():
                self.monitor_task.cancel()
            # Save metrics before stopping
            self._save_metrics()
            logger.info("Stopped resource monitoring")
    
    async def _monitoring_loop(self):
        """Background loop that monitors resources"""
        last_detail_check = 0
        last_hourly_summary = 0
        last_daily_summary = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Regular resource check
                await self._check_basic_resources()
                
                # Detailed check at longer intervals
                if current_time - last_detail_check >= self.detail_interval:
                    await self._check_detailed_resources()
                    last_detail_check = current_time
                
                # Hourly summary
                if current_time - last_hourly_summary >= 3600:  # 1 hour
                    self.metrics.summarize_hourly_metrics()
                    last_hourly_summary = current_time
                
                # Daily summary
                if current_time - last_daily_summary >= 86400:  # 24 hours
                    self.metrics.summarize_daily_metrics()
                    last_daily_summary = current_time
                    # Save metrics daily
                    self._save_metrics()
                
                # Check if auto-optimization should run
                if self.auto_optimize and current_time - self.last_optimization_time >= self.optimization_interval:
                    await self._auto_optimize()
                    self.last_optimization_time = current_time
                
                # Sleep until next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                logger.info("Resource monitoring loop cancelled")
                break
                
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                # Sleep briefly to avoid high CPU usage in case of repeated errors
                await asyncio.sleep(5)
    
    async def _check_basic_resources(self):
        """Check basic system resources"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.add_cpu_usage(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.metrics.add_memory_usage(memory_percent)
            
            # Disk usage for the system drive
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            self.metrics.add_disk_usage(disk_percent)
            
            # Network IO
            net_io = psutil.net_io_counters()
            self.metrics.add_network_io(net_io.bytes_sent, net_io.bytes_recv)
            
            # GPU usage if available
            if self.gpu_monitor.gpu_available:
                gpu_usage = self.gpu_monitor.get_gpu_usage()
                # Average utilization across all GPUs
                if gpu_usage:
                    avg_gpu_util = sum(gpu["utilization_percent"] for gpu in gpu_usage.values()) / len(gpu_usage)
                    self.metrics.add_gpu_usage(avg_gpu_util)
            
            # Determine system state based on resources
            new_state = self._determine_system_state(cpu_percent, memory_percent)
            
            # Update system state if changed
            if new_state != self.current_state:
                self.current_state = new_state
                self.selector.update_system_state(new_state)
                logger.info(f"Updated system state to: {new_state.value}")
            
        except Exception as e:
            logger.error(f"Error checking basic resources: {e}")
    
    async def _check_detailed_resources(self):
        """Check detailed system resources including process stats"""
        try:
            # Check process-specific stats
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                try:
                    proc_info = proc.info
                    proc_name = proc_info['name'].lower()
                    
                    # Monitor specific processes of interest
                    if any(target in proc_name for target in self.processes_to_monitor):
                        cpu_percent = proc_info['cpu_percent']
                        memory_bytes = proc_info['memory_info'].rss if proc_info['memory_info'] else 0
                        memory_percent = (memory_bytes / self.system_info['memory_total']) * 100.0
                        
                        self.metrics.update_process_stats(proc_name, cpu_percent, memory_bytes)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            # Optimize resource allocation
            if self.auto_optimize:
                self.optimizer.allocate_resources(
                    self.system_info['memory_total'],
                    self.system_info['cpu_count']
                )
            
            logger.debug("Completed detailed resource check")
            
        except Exception as e:
            logger.error(f"Error checking detailed resources: {e}")
    
    async def _auto_optimize(self):
        """Perform automatic optimization based on resource usage patterns"""
        try:
            # Get optimization recommendations
            recommendations = self.optimizer.get_optimization_recommendations()
            
            if recommendations["reduce_memory_usage"] or recommendations["reduce_cpu_usage"]:
                logger.info("Auto-optimization triggered due to resource constraints")
                
                # Update system state to LOW_RESOURCES to trigger model switching
                if self.current_state != SystemState.LOW_RESOURCES:
                    self.set_system_state(SystemState.LOW_RESOURCES)
                
                # Log recommendations
                for action in recommendations["suggested_actions"]:
                    logger.info(f"Optimization recommendation: {action}")
                
                # Log components that need optimization
                for component in recommendations["components_to_optimize"]:
                    logger.info(f"Component requiring optimization: {component['name']} "
                               f"({component['issue']}, current: {component.get('current', 'N/A')}, "
                               f"allocated: {component.get('allocated', 'N/A')})")
            
            logger.info("Auto-optimization check completed")
            
        except Exception as e:
            logger.error(f"Error during auto-optimization: {e}")
    
    def _determine_system_state(self, cpu_percent: float, memory_percent: float) -> SystemState:
        """Determine system state based on resource usage
        
        Args:
            cpu_percent: CPU usage percentage
            memory_percent: Memory usage percentage
            
        Returns:
            Appropriate SystemState
        """
        # Check if resources are constrained
        if cpu_percent > self.cpu_high_threshold or memory_percent > self.memory_high_threshold:
            return SystemState.LOW_RESOURCES
            
        # Check if resources are abundant
        if cpu_percent < self.cpu_low_threshold and memory_percent < self.memory_low_threshold:
            # If we're in IDLE or DREAMING state, maintain it
            if self.current_state in (SystemState.IDLE, SystemState.DREAMING):
                return self.current_state
            return SystemState.HIGH_RESOURCES
            
        # Default to current state or ACTIVE if resources are moderate
        return self.current_state if self.current_state != SystemState.LOW_RESOURCES else SystemState.ACTIVE
    
    def set_system_state(self, state: SystemState) -> None:
        """Manually set the system state
        
        Args:
            state: New system state
        """
        if state != self.current_state:
            self.current_state = state
            self.selector.update_system_state(state)
            logger.info(f"Manually set system state to: {state.value}")
    
    def _save_metrics(self) -> bool:
        """Save metrics to disk"""
        try:
            data = self.metrics.to_dict()
            with open(self.metrics_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved resource metrics to {self.metrics_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            return False
    
    def _load_metrics(self) -> bool:
        """Load metrics from disk"""
        if not os.path.exists(self.metrics_path):
            logger.info(f"No existing metrics file found at {self.metrics_path}")
            return False
            
        try:
            with open(self.metrics_path, 'r') as f:
                data = json.load(f)
            self.metrics.from_dict(data)
            logger.info(f"Loaded resource metrics from {self.metrics_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            return False
    
    def track_model_performance(self, model_name: str, response_time: float, tokens: int) -> None:
        """Track performance metrics for a model
        
        Args:
            model_name: Name of the model
            response_time: Time taken to generate response in seconds
            tokens: Number of tokens in the response
        """
        # Update metrics
        self.metrics.add_response_time(model_name, response_time)
        self.metrics.update_token_rate(model_name, tokens, response_time)
        
        # Update model stats in the model manager
        stats = {
            "response_time": self.metrics.get_avg_response_time(model_name),
            "resource_usage": {
                "cpu": min(1.0, response_time / 10),  # Scale to 0-1 range
                "memory": min(1.0, response_time / 15)  # Scale to 0-1 range
            }
        }
        
        self.selector.update_model_stats(model_name, stats)
        logger.debug(f"Updated performance stats for model {model_name}: avg_time={self.metrics.get_avg_response_time(model_name):.2f}s")
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource statistics with detailed metrics
        
        Returns:
            Dictionary of comprehensive resource statistics
        """
        return {
            "current": {
                "cpu_percent": self.metrics.cpu_usage[-1] if self.metrics.cpu_usage else 0,
                "memory_percent": self.metrics.memory_usage[-1] if self.metrics.memory_usage else 0,
                "disk_percent": self.metrics.disk_usage[-1] if self.metrics.disk_usage else 0,
                "system_state": self.current_state.value,
            },
            "averages": {
                "cpu_avg_5min": self.metrics.get_avg_cpu_usage(5),
                "cpu_avg_15min": self.metrics.get_avg_cpu_usage(15),
                "memory_avg_5min": self.metrics.get_avg_memory_usage(5),
                "memory_avg_15min": self.metrics.get_avg_memory_usage(15)
            },
            "model_performance": {
                "response_times": {model: self.metrics.get_avg_response_time(model) for model in self.metrics.response_times},
                "token_rates": {model: self.metrics.get_avg_token_rate(model) for model in self.metrics.token_rates}
            },
            "resource_allocation": {
                "memory": self.metrics.allocated_memory,
                "cpu": self.metrics.allocated_cpu
            },
            "system_info": self.system_info,
            "timestamp": time.time()
        }