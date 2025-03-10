import time
import random
import json
import threading
import os
import sys
import asyncio
from typing import Dict, Any

# Add the parent directory to the path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import resource monitoring components
from server.model_selector import ModelSelector, SystemState
from server.resource_monitor import ResourceMonitor

def simulate_workload(duration=5, intensity=0.5):
    """
    Simulate CPU workload for testing.
    
    Args:
        duration: Time in seconds to run the workload
        intensity: Intensity of the workload (0.0 to 1.0)
    """
    print(f"Simulating workload: intensity={intensity}, duration={duration}s")
    start_time = time.time()
    while time.time() - start_time < duration:
        # Adjust the sleep time based on intensity (higher intensity = less sleep)
        time.sleep(0.01 + (1 - intensity) * 0.1)
        # Generate some CPU load
        _ = [i**2 for i in range(int(10000 * intensity))]

def print_resource_stats(stats: Dict[str, Any]):
    """
    Pretty print resource statistics.
    """
    print("\n==== RESOURCE STATISTICS ====")
    print(f"Timestamp: {stats.get('timestamp')}")
    
    # Current state
    current = stats.get('current', {})
    print("\nCURRENT STATE:")
    print(f"  CPU: {current.get('cpu_percent', 0):.2f}%")
    print(f"  Memory: {current.get('memory_percent', 0):.2f}%")
    print(f"  Disk: {current.get('disk_percent', 0):.2f}%")
    print(f"  System State: {current.get('system_state', 'Unknown')}")
    
    # Averages
    averages = stats.get('averages', {})
    print("\nAVERAGES:")
    print(f"  CPU (5min): {averages.get('cpu_avg_5min', 0):.2f}%")
    print(f"  CPU (15min): {averages.get('cpu_avg_15min', 0):.2f}%")
    print(f"  Memory (5min): {averages.get('memory_avg_5min', 0):.2f}%")
    print(f"  Memory (15min): {averages.get('memory_avg_15min', 0):.2f}%")
    
    # Model performance
    model_perf = stats.get('model_performance', {})
    print("\nMODEL PERFORMANCE:")
    response_times = model_perf.get('response_times', {})
    token_rates = model_perf.get('token_rates', {})
    
    if response_times:
        print("  Response Times (ms):")
        for model, time_ms in response_times.items():
            print(f"    {model}: {time_ms:.2f}ms")
    
    if token_rates:
        print("  Token Rates (tokens/sec):")
        for model, rate in token_rates.items():
            print(f"    {model}: {rate:.2f} tokens/sec")
    
    # Resource allocation
    allocation = stats.get('resource_allocation', {})
    print("\nRESOURCE ALLOCATION:")
    # Check if CPU is a dict or a simple value and handle accordingly
    cpu_alloc = allocation.get('cpu', 0)
    if isinstance(cpu_alloc, dict):
        print(f"  CPU: {cpu_alloc}")
    else:
        print(f"  CPU: {cpu_alloc:.2f}%")
    
    # Check if memory is a dict or a simple value and handle accordingly
    mem_alloc = allocation.get('memory', 0)
    if isinstance(mem_alloc, dict):
        print(f"  Memory: {mem_alloc}")
    else:
        print(f"  Memory: {mem_alloc:.2f}MB")
    
    # System info
    sys_info = stats.get('system_info', {})
    print("\nSYSTEM INFO:")
    for key, value in sys_info.items():
        print(f"  {key}: {value}")

def print_optimization_recommendations(recommendations):
    """
    Print optimization recommendations.
    """
    print("\n==== OPTIMIZATION RECOMMENDATIONS ====")
    
    # Check the type of recommendations
    if recommendations is None:
        print("  No recommendations available.")
        return
    
    if isinstance(recommendations, bool):
        print(f"  Optimization status: {'Success' if recommendations else 'Failed'}")
        return
    
    if not isinstance(recommendations, dict):
        print(f"  Unexpected recommendation format: {type(recommendations).__name__}")
        print(f"  Value: {recommendations}")
        return
    
    # Process dictionary of recommendations
    for category, items in recommendations.items():
        print(f"\n{category.upper()}:")
        
        # Check if items is iterable
        if isinstance(items, (list, tuple)):
            for item in items:
                print(f"  - {item}")
        else:
            print(f"  {items}")

async def async_main():
    # Create output directory for metrics if it doesn't exist
    os.makedirs("metrics", exist_ok=True)
    
    print("Initializing ModelSelector...")
    selector = ModelSelector.get_instance()
    
    print("Initializing ResourceMonitor...")
    monitor = ResourceMonitor.get_instance()
    
    # Start the monitoring
    print("Starting resource monitoring...")
    monitor.start()
    
    try:
        # Instead of registering mock models, list available models from LM Studio
        print("\nListing available LM Studio models...")
        # Let's assume models are already registered in the model manager
        # Get the models from the model manager
        available_models = selector.manager.get_all_models()
        if available_models:
            print(f"Found {len(available_models)} models:")
            for model_name, model_data in available_models.items():
                print(f"  - {model_name}: {model_data.get('description', 'No description')}")
        else:
            print("No models found. Using sample local models for testing.")
            # For testing purposes, we'll track performance for some local models that might be used
            local_models = ["llama-v1.5-7b", "qwen2.5-7b", "deepseek-coder"]
            print(f"Using sample models for testing: {', '.join(local_models)}")
        
        # Initial stats
        print("\nInitial system state:")
        stats = monitor.get_resource_stats()
        print_resource_stats(stats)
        
        # Run a series of tests simulating different workloads
        print("\nRunning tests with different workloads...")
        
        # Test 1: Low intensity workload
        simulate_workload(duration=3, intensity=0.2)
        stats = monitor.get_resource_stats()
        print_resource_stats(stats)
        
        # Simulate model usage
        print("\nSimulating model usage...")
        # Extract the list of available model names or use sample local models
        model_names = list(available_models.keys()) if available_models else ["llama-v1.5-7b", "qwen2.5-7b", "deepseek-coder"]
        
        # Simulate local model usage with typical performance metrics
        monitor.track_model_performance(model_names[0] if model_names else "llama-v1.5-7b", response_time=1.8, tokens=150)
        if len(model_names) > 1:
            monitor.track_model_performance(model_names[1], response_time=1.2, tokens=200)
        if len(model_names) > 2:
            monitor.track_model_performance(model_names[2], response_time=0.95, tokens=250)
        
        # Test 2: Medium intensity workload
        simulate_workload(duration=5, intensity=0.5)
        stats = monitor.get_resource_stats()
        print_resource_stats(stats)
        
        # Get optimization recommendations
        recommendations = monitor.optimizer.get_optimization_recommendations()
        print_optimization_recommendations(recommendations)
        
        # Test 3: High intensity workload
        simulate_workload(duration=3, intensity=0.8)
        stats = monitor.get_resource_stats()
        print_resource_stats(stats)
        
        # Get updated optimization recommendations
        recommendations = monitor.optimizer.get_optimization_recommendations()
        print_optimization_recommendations(recommendations)
        
        # Check if metrics were saved
        if os.path.exists(monitor.metrics_path):
            print(f"\nMetrics saved to: {monitor.metrics_path}")
            with open(monitor.metrics_path, 'r') as f:
                metrics_data = json.load(f)
                print(f"Metrics file size: {len(json.dumps(metrics_data))} bytes")
                print(f"Number of data points: {len(metrics_data.get('cpu_usage', []))}")
        
        # Load metrics from file
        print("\nTesting load metrics from file...")
        await asyncio.sleep(1)  # Give time for any pending saves
        success = monitor._save_metrics()
        print(f"Metrics saved: {success}")
        new_monitor = ResourceMonitor.get_instance()
        loaded = new_monitor._load_metrics()
        print(f"Metrics loaded successfully: {loaded}")
        
        # Test switching system state
        print("\nTesting system state transitions...")
        current_state = monitor.current_state
        print(f"Current state: {current_state}")
        
        # Force a system state change
        print("Forcing system state to HIGH_RESOURCES...")
        monitor.set_system_state(SystemState.HIGH_RESOURCES)
        print(f"New state: {monitor.current_state}")
        
        # Force a system state change back
        print("Forcing system state to ACTIVE...")
        monitor.set_system_state(SystemState.ACTIVE)
        print(f"New state: {monitor.current_state}")
        
        # Give time for any background tasks to complete
        await asyncio.sleep(2)
        
    finally:
        # Stop monitoring
        print("\nStopping resource monitoring...")
        monitor.stop()
        print("Resource monitoring stopped.")

def main():
    # Run the async main function in the asyncio event loop
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("Test interrupted!")
        # Ensure resource monitor is stopped
        ResourceMonitor.get_instance().stop()

if __name__ == "__main__":
    main()
