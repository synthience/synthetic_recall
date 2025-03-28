#!/usr/bin/env python3
"""
Lucidia Think Trace - Cognitive Flow Diagnostic Utility

This utility enables end-to-end tracing of Lucidia's cognitive process:
1. Submits a query to Lucidia's ContextCascadeEngine
2. Captures the IntentGraph and cognitive trace
3. Retrieves and formats diagnostic metrics
4. Provides a visual representation of the cognitive flow

Usage:
    python lucidia_think_trace.py --query "What were the key design principles behind the Titans paper?"

Author: Lucidia Team
Created: 2025-03-28
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Adjust Python path to find the proper modules
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:  # Avoid adding duplicates
    sys.path.insert(0, root_dir)
    print(f"Added {root_dir} to sys.path")
else:
    print(f"{root_dir} already in sys.path")

import aiohttp
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import print as rprint

# --- Use ABSOLUTE IMPORTS ---
try:
    # Import directly from the package root
    from synthians_memory_core.geometry_manager import GeometryManager
    from synthians_memory_core.orchestrator.context_cascade_engine import ContextCascadeEngine
    from synthians_memory_core.synthians_trainer_server.metrics_store import get_metrics_store
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current Python path: {sys.path}")
    print(f"Attempted to import from root: {root_dir}")
    print("Ensure synthians_memory_core and its submodules are correctly structured and importable.")
    sys.exit(1)

# Initialize rich console for pretty printing
console = Console()


async def run_diagnostic_test(query: str, emotion: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        memory_core_url: str = "http://localhost:5010",
                        neural_memory_url: str = "http://localhost:8001",
                        diagnostics_url: str = "http://localhost:8001/diagnose_emoloop",
                        window: str = "last_100") -> Dict[str, Any]:
    """
    Run a complete diagnostic test of Lucidia's cognitive process
    
    Args:
        query: The query to process
        emotion: Optional user emotion
        metadata: Optional metadata
        memory_core_url: URL of the Memory Core service
        neural_memory_url: URL of the Neural Memory Server
        diagnostics_url: URL of the diagnostics endpoint
        window: Window for diagnostics (last_100, last_hour, etc.)
        
    Returns:
        Dictionary with test results
    """
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    if emotion and "emotion" not in metadata:
        metadata["emotion"] = emotion
        
    metadata["session"] = metadata.get("session", f"diagnostic_{int(time.time())}")
    metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Initialize ContextCascadeEngine with geometry manager for proper embedding handling
    console.print("[bold blue]Initializing ContextCascadeEngine[/bold blue]")
    
    try:
        # Initialize GeometryManager with specific configuration for handling dimension mismatches
        # This ensures both 384 and 768 dimension embeddings can be handled safely
        geometry_manager = GeometryManager(config={
            'alignment_strategy': 'truncate',  # or 'pad' - truncate larger vectors to match smaller ones
            'normalization_enabled': True,      # ensure vectors are normalized before comparison
            'embedding_dim': 768               # default dimension, will be adjusted if needed
        })
        
        engine = ContextCascadeEngine(
            memory_core_url=memory_core_url,
            neural_memory_url=neural_memory_url,
            geometry_manager=geometry_manager,
            metrics_enabled=True
        )
    except Exception as e:
        console.print(f"[bold red]Error initializing ContextCascadeEngine:[/bold red] {e}")
        return {"error": str(e), "status": "initialization_failed"}
    
    # Process input
    console.print(f"[bold green]Processing query:[/bold green] {query}")
    start_time = time.time()
    try:
        # Safe processing that handles dimension mismatches and malformed embeddings
        response = await engine.process_new_input(
            content=query,
            embedding=None,  # Let the system generate the embedding
            metadata=metadata
        )
        process_time = time.time() - start_time
    except Exception as e:
        console.print(f"[bold red]Error processing input:[/bold red] {e}")
        return {"error": str(e), "status": "processing_failed", "process_time_ms": (time.time() - start_time) * 1000}
    
    # Get intent graph
    intent_id = response.get("intent_id")
    intent_graph = None
    intent_graph_path = None
    
    if intent_id:
        # Try to find the intent graph file
        intent_graphs_dir = Path("logs/intent_graphs")
        if intent_graphs_dir.exists():
            for file in intent_graphs_dir.glob(f"*{intent_id}*.json"):
                intent_graph_path = file
                try:
                    with open(file, 'r') as f:
                        intent_graph = json.load(f)
                except json.JSONDecodeError:
                    console.print(f"[bold yellow]Warning: Could not parse intent graph file:[/bold yellow] {file}")
                break
    
    # Get diagnostics
    diagnostics = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{diagnostics_url}?window={window}") as resp:
                if resp.status == 200:
                    diagnostics = await resp.json()
    except Exception as e:
        console.print(f"[bold red]Error getting diagnostics:[/bold red] {e}")
    
    # Format diagnostics as table if metrics_store is available
    formatted_diagnostics = None
    try:
        metrics_store = get_metrics_store()
        if metrics_store and diagnostics:
            formatted_diagnostics = metrics_store.format_diagnostics_as_table(diagnostics)
    except Exception as e:
        console.print(f"[bold yellow]Warning: Could not format diagnostics:[/bold yellow] {e}")
    
    return {
        "response": response,
        "intent_id": intent_id,
        "intent_graph": intent_graph,
        "intent_graph_path": str(intent_graph_path) if intent_graph_path else None,
        "diagnostics": diagnostics,
        "formatted_diagnostics": formatted_diagnostics,
        "process_time_ms": process_time * 1000
    }


def display_cognitive_trace(results: Dict[str, Any]) -> None:
    """
    Display a visual representation of the cognitive trace
    
    Args:
        results: Results from run_diagnostic_test
    """
    response = results.get("response", {})
    intent_graph = results.get("intent_graph")
    
    # Display response summary
    console.print("\n[bold cyan]RESPONSE SUMMARY[/bold cyan]")
    summary_table = Table(show_header=True)
    summary_table.add_column("Key", style="dim")
    summary_table.add_column("Value")
    
    summary_table.add_row("Memory ID", response.get("memory_id", "N/A"))
    summary_table.add_row("Intent ID", response.get("intent_id", "N/A"))
    summary_table.add_row("Status", response.get("status", "N/A"))
    summary_table.add_row("Time", f"{results.get('process_time_ms', 0):.2f} ms")
    
    # Add surprise metrics if available
    surprise = response.get("surprise_metrics", {})
    if surprise:
        loss = surprise.get("loss")
        grad_norm = surprise.get("grad_norm")
        boost = surprise.get("boost_calculated")
        
        if loss is not None:
            summary_table.add_row("Loss", f"{loss:.6f}")
        if grad_norm is not None:
            summary_table.add_row("Gradient Norm", f"{grad_norm:.6f}")
        if boost is not None:
            summary_table.add_row("QuickRecal Boost", f"{boost:.6f}")
    
    console.print(summary_table)
    
    # Display intent graph tree if available
    if intent_graph:
        console.print("\n[bold magenta]INTENT GRAPH TRACE[/bold magenta]")
        
        # Create tree structure
        tree = Tree(f"[bold]🧠 Cognitive Trace[/bold] ({response.get('intent_id', 'unknown')})")
        
        # Memory trace
        memory_trace = intent_graph.get("memory_trace", {})
        if memory_trace:
            mem_node = tree.add("[bold yellow]Memory Operations[/bold yellow]")
            
            # Memory storage
            storage = memory_trace.get("storage", [])
            if storage:
                storage_node = mem_node.add(f"[yellow]Storage ({len(storage)} operations)[/yellow]")
                for i, item in enumerate(storage[:3]):  # Show first 3
                    memory_id = item.get("memory_id", "unknown")
                    score = item.get("quickrecal_score", 0)
                    storage_node.add(f"Memory {i+1}: ID={memory_id}, QR={score:.4f}")
                if len(storage) > 3:
                    storage_node.add(f"... {len(storage)-3} more")
            
            # Memory retrievals
            retrieved = memory_trace.get("retrieved", [])
            if retrieved:
                retrieval_node = mem_node.add(f"[yellow]Retrievals ({len(retrieved)} operations)[/yellow]")
                for i, item in enumerate(retrieved[:3]):  # Show first 3
                    memory_id = item.get("memory_id", "unknown")
                    emotion = item.get("dominant_emotion", "neutral")
                    retrieval_node.add(f"Memory {i+1}: ID={memory_id}, Emotion={emotion}")
                if len(retrieved) > 3:
                    retrieval_node.add(f"... {len(retrieved)-3} more")
        
        # Neural memory trace
        neural_trace = intent_graph.get("neural_memory_trace", {})
        if neural_trace:
            neural_node = tree.add("[bold blue]Neural Memory Operations[/bold blue]")
            
            # Updates
            updates = neural_trace.get("updates", [])
            if updates:
                update_node = neural_node.add(f"[blue]Updates ({len(updates)} operations)[/blue]")
                for i, item in enumerate(updates[:3]):  # Show first 3
                    loss = item.get("loss", 0)
                    grad = item.get("grad_norm", 0)
                    update_node.add(f"Update {i+1}: Loss={loss:.6f}, GradNorm={grad:.6f}")
                if len(updates) > 3:
                    update_node.add(f"... {len(updates)-3} more")
        
        # Reasoning steps
        steps = intent_graph.get("reasoning_steps", [])
        if steps:
            reasoning_node = tree.add("[bold green]Reasoning Steps[/bold green]")
            for i, step in enumerate(steps):
                step_desc = step.get("description", "Unknown step")
                # Truncate if too long
                if len(step_desc) > 70:
                    step_desc = step_desc[:67] + "..."
                reasoning_node.add(f"Step {i+1}: {step_desc}")
        
        # Final output
        output = intent_graph.get("final_output", "No output recorded")
        result_node = tree.add("[bold cyan]Final Output[/bold cyan]")
        if isinstance(output, str) and len(output) > 100:
            result_node.add(f"{output[:97]}...")
        else:
            result_node.add(str(output))
        
        # Print the tree
        console.print(tree)
        
        # Print path to intent graph file
        if results.get("intent_graph_path"):
            console.print(f"\nFull intent graph saved to: [italic]{results['intent_graph_path']}[/italic]")
    
    # Display diagnostics if available
    if results.get("formatted_diagnostics"):
        console.print("\n[bold cyan]COGNITIVE DIAGNOSTICS[/bold cyan]")
        console.print(results["formatted_diagnostics"])
    elif results.get("diagnostics"):
        console.print("\n[bold cyan]COGNITIVE DIAGNOSTICS (raw)[/bold cyan]")
        console.print(json.dumps(results["diagnostics"], indent=2))


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Lucidia Think Trace - Cognitive Flow Diagnostic Utility")
    parser.add_argument("--query", type=str, required=True, help="Query to process")
    parser.add_argument("--emotion", type=str, default=None, help="User emotion (e.g., curiosity, confusion)")
    parser.add_argument("--memcore-url", type=str, default="http://localhost:5010", help="Memory Core URL")
    parser.add_argument("--neural-url", type=str, default="http://localhost:8001", help="Neural Memory Server URL")
    parser.add_argument("--window", type=str, default="last_100", help="Diagnostics window")
    parser.add_argument("--topic", type=str, default=None, help="Topic tag for metadata")
    parser.add_argument("--session", type=str, default=None, help="Session ID for metadata")
    parser.add_argument("--output-json", type=str, default=None, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Prepare metadata
    metadata = {
        "source": "lucidia_think_trace"
    }
    
    if args.emotion:
        metadata["emotion"] = args.emotion
    
    if args.topic:
        metadata["topic"] = args.topic
        
    if args.session:
        metadata["session"] = args.session
    
    # Run diagnostic test
    console.print(Panel.fit(
        f"[bold]LUCIDIA THINK TRACE[/bold]\n\nQuery: {args.query}",
        title="🧠 Cognitive Flow Diagnostics",
        border_style="cyan"
    ))
    
    results = await run_diagnostic_test(
        query=args.query,
        emotion=args.emotion,
        metadata=metadata,
        memory_core_url=args.memcore_url,
        neural_memory_url=args.neural_url,
        window=args.window
    )
    
    # Display results
    display_cognitive_trace(results)
    
    # Save results to file if requested
    if args.output_json:
        # Remove formatted_diagnostics as it's not JSON serializable
        results_copy = {k: v for k, v in results.items() if k != "formatted_diagnostics"}
        with open(args.output_json, 'w') as f:
            json.dump(results_copy, f, indent=2)
        console.print(f"\nResults saved to: [italic]{args.output_json}[/italic]")
    
    return results


if __name__ == "__main__":
    try:
        results = asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)
