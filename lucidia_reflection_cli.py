#!/usr/bin/env python3
"""
LUCIDIA REFLECTION CLI

A command-line interface for monitoring Lucidia's self-awareness, memory evolution,
and reflection processes. This tool connects to the Dream API Server, Tensor Server,
and HPC Server to provide insights into Lucidia's cognitive development.

Usage:
    python lucidia_reflection_cli.py [command] [options]
"""

import os
import sys
import json
import time
import asyncio
import aiohttp
import argparse
import logging
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import websockets
from websockets.exceptions import ConnectionClosed
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.columns import Columns
from rich.text import Text
from rich import box
from rich.prompt import Confirm
import matplotlib.pyplot as plt
import numpy as np
from pyvis.network import Network
import networkx as nx
import random
import urllib.parse
import re

# Import the LM Studio client
from llm_client import LMStudioClient

# Console for rich output with enhanced style
console = Console(highlight=True)

async def async_input(prompt: str) -> str:
    """Get input from user asynchronously.
    
    Args:
        prompt: The prompt to display
        
    Returns:
        The user input
    """
    return await asyncio.get_event_loop().run_in_executor(None, lambda: console.input(prompt))

# Default configuration
DEFAULT_CONFIG = {
    "dream_api_url": "http://localhost:8081",
    "tensor_server_url": "ws://localhost:5001",
    "hpc_server_url": "ws://localhost:5005",
    "lm_studio_url": "http://127.0.0.1:1234",
    "use_local_llm": False,
    "ping_interval": 30.0,
    "auto_reconnect": True,
    "metrics_history_size": 100,
    "log_level": "INFO",
    "log_file": "lucidia_reflection_cli.log",
    "session_dir": "session_data",
    "default_dream_duration": 600,  # 10 minutes
    "visualization_update_interval": 5,  # seconds
    "continuous_monitoring": False,
    "export_metrics": True,
    "interactive_mode": True,
    "theme": {
        "primary": "cyan",
        "secondary": "blue",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "info": "dim cyan",
        "accent": "magenta"
    }
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DEFAULT_CONFIG["log_file"])
    ]
)
logger = logging.getLogger("LucidiaReflectionCLI")

class LucidiaReflectionClient:
    """Client for connecting to Lucidia's memory system and monitoring reflection processes."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the reflection client with configuration options."""
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.dream_api_url = self.config["dream_api_url"]
        self.tensor_server_url = self.config["tensor_server_url"]
        self.hpc_server_url = self.config["hpc_server_url"]
        
        # Theme settings
        self.theme = self.config.get("theme", DEFAULT_CONFIG["theme"])
        
        # Connection objects
        self.session = None
        self.tensor_ws = None
        self.hpc_ws = None
        
        # Local LLM client
        self.llm_client = LMStudioClient(self.config) if self.config["use_local_llm"] else None
        
        # Metrics tracking
        self.metrics = {
            "self_awareness_depth": [],
            "integration_effectiveness": [],
            "surprise_significance": [],
            "reflective_coherence": [],
            "kg_connectivity": [],
            "spiral_phase_maturity": [],
            "dreaming_sessions": [],
            "memory_count": {"stm": [], "ltm": [], "mpl": []},
            "current_spiral_phase": None,
            "phase_history": [],
            "knowledge_graph": None,
            "significant_memories": [],
            "dream_insights": []
        }
        
        # Create session directory if it doesn't exist
        os.makedirs(self.config["session_dir"], exist_ok=True)
        
        # Generate unique session ID
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = os.path.join(self.config["session_dir"], self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Active dream session
        self.active_dream_session = None
        
        logger.info(f"Initialized Lucidia Reflection CLI with session ID: {self.session_id}")
    
    async def connect(self):
        """Connect to all required services.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create aiohttp session if it doesn't exist
            if self.session is None:
                self.session = aiohttp.ClientSession()
            
            # Test connection to dream API
            logger.info("Connecting to Dream API...")
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[{self.theme['primary']}]Connecting to Dream API...[/{self.theme['primary']}]"),
                console=console
            ) as progress:
                task = progress.add_task("Connecting", total=None)
                response = await self._request_with_retry(
                    self.session.get, 
                    f"{self.dream_api_url}/health"
                )
            
            if response and response.status == 200:
                data = await response.json()
                logger.info(f"Connected to Dream API: {data.get('status')}")
                console.print(f"[{self.theme['success']}]✓ Connected to Dream API: {data.get('status')}[/{self.theme['success']}]")
                return True
            else:
                logger.error("Failed to connect to Dream API")
                console.print(f"[{self.theme['error']}]✗ Failed to connect to Dream API[/{self.theme['error']}]")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to services: {e}")
            console.print(f"[{self.theme['error']}]✗ Error connecting to services: {e}[/{self.theme['error']}]")
            return False
    
    async def disconnect(self):
        """Disconnect from all services."""
        # Close aiohttp session
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Disconnected from Dream API")
        
        # Close WebSocket connections
        if self.tensor_ws:
            await self.tensor_ws.close()
            self.tensor_ws = None
            logger.info("Disconnected from Tensor Server")
            
        if self.hpc_ws:
            await self.hpc_ws.close()
            self.hpc_ws = None
            logger.info("Disconnected from HPC Server")
        
        # Disconnect from local LLM if connected
        if self.llm_client:
            await self.llm_client.disconnect()
    
    async def fetch_system_state(self):
        """Fetch the current state of Lucidia's memory system."""
        try:
            # Get system health (at root path)
            response = await self._request_with_retry(self.session.get, f"{self.dream_api_url}/health")
            if response:
                health = await response.json()
                logger.info(f"System health: {health['status']}")
            
            # Get memory stats
            response = await self._request_with_retry(self.session.get, f"{self.dream_api_url}/api/stats")
            if response:
                stats = await response.json()
                self._update_memory_metrics(stats)
                logger.info(f"Retrieved memory stats: {len(stats.get('memories', {}))} memories")
            
            # Get spiral phase info
            response = await self._request_with_retry(self.session.get, f"{self.dream_api_url}/api/spiral-phase")
            if response:
                phase_data = await response.json()
                self._update_spiral_phase(phase_data)
                logger.info(f"Current spiral phase: {phase_data.get('current_phase')}")
            
            # Get knowledge graph
            response = await self._request_with_retry(self.session.get, f"{self.dream_api_url}/api/knowledge-graph")
            if response:
                kg_data = await response.json()
                self.metrics["knowledge_graph"] = kg_data
                self._calculate_kg_connectivity(kg_data)
                logger.info(f"Retrieved knowledge graph: {len(kg_data.get('nodes', []))} nodes, {len(kg_data.get('edges', []))} edges")
                
            return True
        except Exception as e:
            logger.error(f"Error fetching system state: {e}")
            return False

    async def start_dream_session(self, duration: int = None, depth: float = None, 
                             creativity: float = None, with_memories: List[str] = None):
        """Start a new dream session.
        
        Args:
            duration: Dream session duration in seconds
            depth: Reflection depth (0.0-1.0)
            creativity: Creative exploration level (0.0-1.0)
            with_memories: List of memory IDs to include in the dream
        
        Returns:
            str: Session ID if successful, None otherwise
        """
        try:
            # Build parameters
            params = {
                "duration": duration or self.config["default_dream_duration"],
                "depth": depth or 0.7,
                "creativity": creativity or 0.5,
                "memories": with_memories or []
            }
            
            logger.info(f"Starting dream session with params: {params}")
            
            # Check if we should use local LLM
            if self.config["use_local_llm"] and self.llm_client:
                return await self._start_local_dream_session(params)
            
            # Otherwise use the Dream API
            console.print(Panel(
                f"Starting dream session with:\n"
                f"  Duration: {params['duration']} seconds\n"
                f"  Depth: {params['depth']}\n"
                f"  Creativity: {params['creativity']}\n"
                f"  Memories: {len(params['memories'])} selected",
                title="Dream Session",
                border_style=self.theme["primary"]
            ))
            
            response = await self._request_with_retry(
                self.session.post, 
                f"{self.dream_api_url}/api/sessions/start", 
                json=params
            )
            
            if not response:
                logger.error("Failed to start dream session")
                console.print(f"[{self.theme['error']}]Failed to start dream session[/{self.theme['error']}]")
                return None
                
            result = await response.json()
            session_id = result.get("session_id")
            
            if not session_id:
                logger.error(f"Invalid response: {result}")
                console.print(f"[{self.theme['error']}]Invalid response from server[/{self.theme['error']}]")
                return None
                
            logger.info(f"Started dream session: {session_id}")
            console.print(f"[{self.theme['success']}]Started dream session: {session_id}[/{self.theme['success']}]")
            
            # Monitor session
            await self._monitor_dream_session(session_id)
            
            # Process results
            await self._process_dream_results(session_id)
            
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting dream session: {e}")
            console.print(f"[{self.theme['error']}]Error starting dream session: {e}[/{self.theme['error']}]")
            return None

    async def _start_local_dream_session(self, params: Dict[str, Any]) -> str:
        """Start a dream session using the local LLM.
        
        Args:
            params: Dream session parameters
            
        Returns:
            str: Session ID
        """
        # Generate a session ID
        session_id = f"local_{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting local dream session: {session_id}")
        
        console.print(Panel(
            f"Starting local dream session with:\n"
            f"  Session ID: {session_id}\n"
            f"  Depth: {params['depth']}\n"
            f"  Creativity: {params['creativity']}",
            title="Local Dream Session",
            border_style=self.theme["secondary"]
        ))
        
        # Get some significant memories to reflect on
        try:
            memories = await self.get_significant_memories(limit=5)
            
            # Validate memories
            if not isinstance(memories, list):
                logger.warning(f"Invalid memories returned: Expected list, got {type(memories).__name__}")
                memories = []
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            memories = []
            
        if not memories:
            logger.warning("No memories available for reflection, creating dummy memories")
            console.print(f"[{self.theme['warning']}]No memories available for reflection, creating test memories[/{self.theme['warning']}]")
            # Create some dummy memories for testing
            memories = [
                {
                    "id": f"memory_{i}",
                    "content": f"Test memory {i} for reflection",
                    "created_at": datetime.now().isoformat(),
                    "significance": random.random()
                }
                for i in range(3)
            ]
        
        # Display progress
        reflection = None
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[{self.theme['primary']}]Generating reflection...[/{self.theme['primary']}]"),
                console=console
            ) as progress:
                progress.add_task("Reflecting", total=None)
                
                # Generate reflection
                reflection = await self.llm_client.generate_reflection(
                    memories, 
                    depth=params["depth"], 
                    creativity=params["creativity"]
                )
        except Exception as e:
            error_msg = f"Error during reflection generation: {e}"
            logger.error(error_msg)
            reflection = {"status": "error", "message": error_msg}
        
        # Save reflection to session directory if it's a valid dictionary
        if reflection and isinstance(reflection, dict):
            try:
                reflection_path = os.path.join(self.session_dir, f"{session_id}_reflection.json")
                with open(reflection_path, "w") as f:
                    json.dump(reflection, f, indent=2)
                logger.info(f"Saved reflection to {reflection_path}")
            except Exception as e:
                logger.error(f"Error saving reflection to file: {e}")
        else:
            logger.warning(f"Could not save reflection: Invalid format: {type(reflection)}")
        
        # Display the reflection
        await self._display_local_reflection(reflection, session_id)
        
        return session_id
    
    async def _display_local_reflection(self, reflection: Any, session_id: str):
        """Display a local reflection result.
        
        Args:
            reflection: The reflection result
            session_id: The session ID
        """
        # Handle the case when reflection is None
        if reflection is None:
            console.print(f"[{self.theme['error']}]Reflection failed: No result returned[/{self.theme['error']}]")
            return
            
        # Handle the case when reflection is a string - try to parse it as JSON first
        if isinstance(reflection, str):
            try:
                reflection = json.loads(reflection)
            except json.JSONDecodeError:
                console.print(f"[{self.theme['error']}]Reflection failed: {reflection}[/{self.theme['error']}]")
                return
            
        # Handle non-dictionary types after potential JSON parsing
        if not isinstance(reflection, dict):
            console.print(f"[{self.theme['error']}]Reflection failed: Unexpected result type {type(reflection)}[/{self.theme['error']}]")
            return
            
        # If it's a dictionary but not a success
        if reflection.get("status") == "error":
            console.print(f"[{self.theme['error']}]Reflection failed: {reflection.get('message', 'Unknown error')}[/{self.theme['error']}]")
            return
        
        # Extract components safely with proper type checking
        # For backward compatibility, extract either 'fragments' or individual fragment types
        fragments = reflection.get("fragments", [])
        if not isinstance(fragments, list):
            fragments = []
        
        # Handle either new format (all fragments in one list) or old format (separate lists by type)
        insights = reflection.get("insights", [])
        if not isinstance(insights, list):
            insights = []
            
        questions = reflection.get("questions", [])
        if not isinstance(questions, list):
            questions = []
            
        hypotheses = reflection.get("hypotheses", [])
        if not isinstance(hypotheses, list):
            hypotheses = []
            
        counterfactuals = reflection.get("counterfactuals", [])
        if not isinstance(counterfactuals, list):
            counterfactuals = []
            
        # If we have fragments but not the individual lists, distribute them by type
        if fragments and not (insights or questions or hypotheses or counterfactuals):
            for fragment in fragments:
                if not isinstance(fragment, dict):
                    continue
                    
                fragment_type = fragment.get("type", "").lower()
                if fragment_type == "insight":
                    insights.append(fragment)
                elif fragment_type == "question":
                    questions.append(fragment)
                elif fragment_type == "hypothesis":
                    hypotheses.append(fragment)
                elif fragment_type == "counterfactual":
                    counterfactuals.append(fragment)
        
        # Display the successful reflection
        console.print(f"\n[{self.theme['success']} bold]Reflection Session: {session_id}[/{self.theme['success']} bold]")

        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Create header
        metadata = reflection.get("metadata", {})
        header_text = Text(f"Title: {reflection['title']}\n", style=f"bold {self.theme['primary']}")
        header_text.append(f"Depth: {metadata.get('depth', 'N/A')} | ")
        header_text.append(f"Creativity: {metadata.get('creativity', 'N/A')} | ")
        header_text.append(f"Memory Count: {metadata.get('memory_count', 'N/A')}")
        layout["header"].update(Panel(header_text, title="Reflection", border_style=self.theme["primary"]))
        
        # Group fragments by type
        insights_text = "\n\n".join([f"[{i+1}] {insight.get('content', '')} ({insight.get('confidence', 0):.2f})" 
                               for i, insight in enumerate(insights)])
        insights_panel = Panel(insights_text or "No insights generated", 
                                title=f"Insights ({len(insights)})",
                                border_style=self.theme["secondary"])
        
        questions_text = "\n\n".join([f"[{i+1}] {question.get('content', '')} ({question.get('confidence', 0):.2f})" 
                                 for i, question in enumerate(questions)])
        questions_panel = Panel(questions_text or "No questions generated", 
                                title=f"Questions ({len(questions)})",
                                border_style=self.theme["accent"])
        
        hypotheses_text = "\n\n".join([f"[{i+1}] {hypothesis.get('content', '')} ({hypothesis.get('confidence', 0):.2f})" 
                                  for i, hypothesis in enumerate(hypotheses)])
        hypotheses_panel = Panel(hypotheses_text or "No hypotheses generated", 
                                title=f"Hypotheses ({len(hypotheses)})",
                                border_style=self.theme["secondary"])
        
        counterfactuals_text = "\n\n".join([f"[{i+1}] {cf.get('content', '')} ({cf.get('confidence', 0):.2f})" 
                                       for i, cf in enumerate(counterfactuals)])
        counterfactuals_panel = Panel(counterfactuals_text or "No counterfactuals generated", 
                                    title=f"Counterfactuals ({len(counterfactuals)})",
                                    border_style=self.theme["accent"])
        
        # Create body with split layout
        body_layout = Layout()
        body_layout.split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        # Left side: Insights and Questions
        left_layout = Layout()
        left_layout.split_column(
            Layout(name="insights"),
            Layout(name="questions")
        )
        
        left_layout["insights"].update(insights_panel)
        left_layout["questions"].update(questions_panel)
        
        body_layout["left"].update(left_layout)
        
        # Right side: Hypotheses and Counterfactuals
        right_layout = Layout()
        right_layout.split_column(
            Layout(name="hypotheses"),
            Layout(name="counterfactuals")
        )
        
        right_layout["hypotheses"].update(hypotheses_panel)
        right_layout["counterfactuals"].update(counterfactuals_panel)
        
        body_layout["right"].update(right_layout)
        layout["body"].update(body_layout)
        
        # Create footer
        footer_text = Text(f"Session ID: {session_id}\n")
        footer_text.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        layout["footer"].update(Panel(footer_text, title="Session Info", border_style=self.theme["info"]))
        
        # Render layout with safe width
        console_width = console.width if console.width else 100
        safe_width = min(console_width - 5, 120)  # Ensure we don't exceed console width
        
        try:
            # Print to console
            console.print(layout, width=safe_width)
        except Exception as e:
            logger.error(f"Error rendering reflection layout: {e}")
            # Fallback to simple text display
            console.print(f"[{self.theme['success']} bold]Reflection Title: {reflection.get('title', 'Untitled')}[/{self.theme['success']} bold]")
            
            for fragment_type, fragments_list in {
                "Insights": insights,
                "Questions": questions,
                "Hypotheses": hypotheses,
                "Counterfactuals": counterfactuals
            }.items():
                if fragments_list:
                    console.print(f"\n[bold]{fragment_type} ({len(fragments_list)}):[/bold]")
                    for i, f in enumerate(fragments_list):
                        console.print(f"  [{i+1}] {f.get('content', '')}")
    
    async def _monitor_dream_session(self, session_id: str):
        """Monitor a dream session for insights and updates."""
        if not session_id:
            logger.error("No session ID provided for monitoring")
            return
            
        logger.info(f"Monitoring dream session: {session_id}")
        
        try:
            # Wait for the dream to finish with periodic status checks
            start_time = time.time()
            done = False
            
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[{self.theme['primary']}]Monitoring dream session: {session_id}[/{self.theme['primary']}]"),
                console=console
            ) as progress:
                monitoring_task = progress.add_task("Monitoring", total=100)
                
                while not done:
                    try:
                        # Check session status
                        response = await self._request_with_retry(
                            self.session.get,
                            f"{self.dream_api_url}/api/sessions/status/{session_id}"
                        )
                        
                        if not response:
                            logger.error(f"Failed to get status for session {session_id}")
                            progress.update(monitoring_task, description=f"[{self.theme['warning']}]Connection issue, retrying...[/{self.theme['warning']}]")
                            await asyncio.sleep(2)
                            continue
                        
                        status = await response.json()
                        
                        # If the dream has finished, process the results
                        if status.get("status") == "complete":
                            logger.info(f"Dream session {session_id} completed")
                            progress.update(monitoring_task, completed=100, description=f"[{self.theme['success']}]Dream completed![/{self.theme['success']}]")
                            done = True
                            await self._process_dream_results(session_id)
                        elif status.get("status") == "error":
                            logger.error(f"Dream session {session_id} ended with an error: {status.get('message', 'Unknown error')}")
                            progress.update(monitoring_task, completed=100, description=f"[{self.theme['error']}]Dream error: {status.get('message', 'Unknown error')}[/{self.theme['error']}]")
                            done = True
                        elif status.get("status") == "in_progress":
                            # Update progress if available
                            progress_value = status.get("progress", 0)
                            elapsed = time.time() - start_time
                            progress.update(monitoring_task, completed=progress_value, description=f"Dream in progress: {elapsed:.1f}s elapsed")
                        
                        # Brief pause before checking again
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"Error checking dream status: {e}")
                        progress.update(monitoring_task, description=f"[{self.theme['error']}]Error: {e}[/{self.theme['error']}]")
                        await asyncio.sleep(2)
        finally:
            # Clear active session once done
            if self.active_dream_session == session_id:
                self.active_dream_session = None
    
    async def _process_dream_results(self, session_id: str):
        """Process and display the results of a dream session."""
        try:
            console.print(f"[{self.theme['primary']}]Processing results from dream session {session_id}...[/{self.theme['primary']}]")
            
            # Get dream results
            response = await self._request_with_retry(
                self.session.get,
                f"{self.dream_api_url}/api/sessions/results/{session_id}"
            )
            
            if not response:
                logger.error(f"Failed to get results for dream session {session_id}")
                console.print(f"[{self.theme['error']}]Failed to get results for dream session {session_id}[/{self.theme['error']}]")
                return
                
            results = await response.json()
            
            # Process insights
            insights = results.get("insights", [])
            if insights:
                logger.info(f"Dream generated {len(insights)} insights")
                console.print(f"[{self.theme['success']}]Dream generated {len(insights)} insights[/{self.theme['success']}]")
                
                # Display insights in a pretty table
                table = Table(title=f"Insights from Dream Session {session_id}", box=box.ROUNDED)
                table.add_column("Type", style=self.theme["secondary"])
                table.add_column("Content")
                table.add_column("Significance", style=self.theme["accent"])
                
                for insight in insights:
                    table.add_row(
                        insight.get("type", "insight"),
                        insight.get("content", "")[:80] + ("..." if len(insight.get("content", "")) > 80 else ""),
                        f"{insight.get('significance', 0):.2f}"
                    )
                
                console.print(table)
                
                # Add to our metrics
                self.metrics["dream_insights"].extend(insights)
                
                # Update other metrics based on dream results
                if "metrics" in results:
                    self._update_metrics_from_dream(results["metrics"])
            else:
                logger.warning("Dream session completed but no insights were generated")
                console.print(f"[{self.theme['warning']}]Dream session completed but no insights were generated[/{self.theme['warning']}]")
                
            # Update system state to reflect changes from the dream
            await self.fetch_system_state()
            
            return results
        except Exception as e:
            logger.error(f"Error processing dream results: {e}")
            console.print(f"[{self.theme['error']}]Error processing dream results: {e}[/{self.theme['error']}]")
            return None
    
    def _update_memory_metrics(self, stats: Dict[str, Any]):
        """Update memory metrics from stats API response."""
        if not stats:
            return
        
        # Memory counts - check for both flat and hierarchical memory structures
        if "hierarchical_memory" in stats and isinstance(stats["hierarchical_memory"], dict):
            # New API structure with hierarchical memory
            self.metrics["memory_count"]["stm"].append(stats["hierarchical_memory"].get("stm", 0))
            self.metrics["memory_count"]["ltm"].append(stats["hierarchical_memory"].get("ltm", 0))
            self.metrics["memory_count"]["mpl"].append(stats["hierarchical_memory"].get("mpl", 0))
        else:
            # Legacy flat structure
            self.metrics["memory_count"]["stm"].append(stats.get("stm_count", 0))
            self.metrics["memory_count"]["ltm"].append(stats.get("ltm_count", 0))
            self.metrics["memory_count"]["mpl"].append(stats.get("mpl_count", 0))
        
        # Integration effectiveness (ratio of memories successfully integrated from STM to LTM)
        if "integration_rate" in stats:
            self.metrics["integration_effectiveness"].append(stats["integration_rate"])
        
        # Reflective coherence (based on coherence score if available)
        if "coherence_score" in stats:
            self.metrics["reflective_coherence"].append(stats["coherence_score"])
        
        # Self-awareness depth (if available)
        if "self_awareness_depth" in stats:
            self.metrics["self_awareness_depth"].append(stats["self_awareness_depth"])
    
    def _update_spiral_phase(self, phase_data: Dict[str, Any]):
        """Update spiral phase metrics."""
        if not phase_data:
            return
        
        current_phase = phase_data.get("current_phase")
        if current_phase:
            self.metrics["current_spiral_phase"] = current_phase
            
            # Add to history if it's a new phase
            if not self.metrics["phase_history"] or self.metrics["phase_history"][-1].get("phase") != current_phase:
                self.metrics["phase_history"].append({
                    "phase": current_phase,
                    "timestamp": datetime.now().isoformat(),
                    "reason": phase_data.get("transition_reason", "unknown")
                })
        
        # Spiral phase maturity (how ready the system is to transition)
        if "maturity" in phase_data:
            self.metrics["spiral_phase_maturity"].append(phase_data["maturity"])
    
    def _calculate_kg_connectivity(self, kg_data: Dict[str, Any]):
        """Calculate knowledge graph connectivity metrics."""
        if not kg_data or "nodes" not in kg_data or "edges" not in kg_data:
            return
        
        nodes = kg_data.get("nodes", [])
        edges = kg_data.get("edges", [])
        
        if not nodes:
            return
        
        # Calculate basic connectivity metrics
        node_count = len(nodes)
        edge_count = len(edges)
        
        if node_count <= 1:
            connectivity = 0
        else:
            # Calculate average degree (connections per node)
            connectivity = (2 * edge_count) / node_count
            
            # Normalize to 0-1 range (assuming ideal connectivity is around 5-10 connections per node)
            connectivity = min(1.0, connectivity / 10.0)
        
        self.metrics["kg_connectivity"].append(connectivity)
    
    def _update_metrics_from_dream(self, dream_metrics: Dict[str, Any]):
        """Update metrics based on dream session results."""
        if not dream_metrics:
            return
        
        # Surprise significance factor
        if "avg_surprise" in dream_metrics:
            self.metrics["surprise_significance"].append(dream_metrics["avg_surprise"])
        
        # Self-awareness depth
        if "self_awareness_depth" in dream_metrics:
            self.metrics["self_awareness_depth"].append(dream_metrics["self_awareness_depth"])
        
        # Integration effectiveness
        if "integration_effectiveness" in dream_metrics:
            self.metrics["integration_effectiveness"].append(dream_metrics["integration_effectiveness"])
        
        # Reflective coherence
        if "coherence" in dream_metrics:
            self.metrics["reflective_coherence"].append(dream_metrics["coherence"])
    
    async def get_memory_by_id(self, memory_id: str):
        """Retrieve a specific memory by ID."""
        try:
            response = await self._request_with_retry(
                self.session.get, 
                f"{self.dream_api_url}/api/memories/{memory_id}"
            )
            
            if response:
                memory = await response.json()
                return memory
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None

    async def get_significant_memories(self, limit: int = 10):
        """Get the most significant memories."""
        try:
            logger.info(f"Retrieving {limit} significant memories from Dream API")
            response = await self._request_with_retry(
                self.session.get, 
                f"{self.dream_api_url}/api/memories/significant?limit={limit}"
            )
            
            if not response:
                logger.warning("No response from Dream API while retrieving memories")
                return []
                
            if response.status == 404:
                logger.warning(f"Request to {response.url} returned status {response.status}")
                return []
                
            if response.status != 200:
                logger.error(f"Failed to get significant memories: {response.status}")
                return []
            
            memories = await response.json()
            
            # Validate memories are in the expected format
            if not isinstance(memories, list):
                logger.warning(f"Unexpected memories format: Expected list, got {type(memories).__name__}")
                return []
                
            logger.info(f"Retrieved {len(memories)} significant memories")
            
            # Store in metrics
            self.metrics["significant_memories"] = memories
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving significant memories: {e}")
            return []

    async def get_parameters(self):
        """Get all system parameters."""
        try:
            response = await self._request_with_retry(
                self.session.get, 
                f"{self.dream_api_url}/api/parameters/config"
            )
            
            if not response:
                logger.error("Failed to get parameters")
                return None
                
            if response.status != 200:
                logger.warning(f"Request to {self.dream_api_url}/api/parameters/config returned status {response.status}")
                logger.error(f"Failed to get parameters: {response.status}")
                return None
                
            return await response.json()
            
        except Exception as e:
            logger.error(f"Error getting parameters: {e}")
            return None

    async def get_parameter(self, param_path: str):
        """Get a specific parameter by its path."""
        try:
            # URL encode the parameter path
            encoded_path = urllib.parse.quote(param_path)
            response = await self._request_with_retry(
                self.session.get, 
                f"{self.dream_api_url}/api/parameters/config/{encoded_path}"
            )
            
            if response:
                return await response.json()
            return None
            
        except Exception as e:
            logger.error(f"Error getting parameter {param_path}: {e}")
            return None

    async def update_parameter(self, param_path: str, value: Any):
        """Update a parameter value."""
        try:
            # URL encode the parameter path
            encoded_path = urllib.parse.quote(param_path)
            response = await self._request_with_retry(
                self.session.post, 
                f"{self.dream_api_url}/api/parameters/config/{encoded_path}", 
                json={"value": value}
            )
            
            if response:
                result = await response.json()
                logger.info(f"Updated parameter {param_path}: {result}")
                
                # If the update was successful on the server, also update the local config file
                if result.get("success") == True:
                    # Load current config
                    local_config = {}
                    if os.path.exists(self.config["config_path"]):
                        with open(self.config["config_path"], "r") as f:
                            local_config = json.load(f)
                    
                    # Update the parameter in the local config
                    path_parts = param_path.split('.')
                    current = local_config
                    
                    # Navigate to the correct nested location
                    for i, part in enumerate(path_parts[:-1]):
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    # Set the value at the final location
                    current[path_parts[-1]] = value
                    
                    # Save the updated config back to disk
                    with open(self.config["config_path"], "w") as f:
                        json.dump(local_config, f, indent=2)
                        
                    logger.info(f"Updated local configuration file: {self.config['config_path']}")
                
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error updating parameter {param_path}: {e}")
            return False
    
    async def get_local_llm_models(self):
        """Get available models from the LM Studio server."""
        if not self.config["use_local_llm"] or not self.llm_client:
            logger.error("Local LLM is not enabled")
            return []
        
        return await self.llm_client.get_available_models()
        
    async def evaluate_memory_significance(self, memory_content: str) -> float:
        """Evaluate the significance of a memory using the local LLM."""
        if not self.config["use_local_llm"] or not self.llm_client:
            logger.error("Local LLM is not enabled")
            return 0.5
        
        return await self.llm_client.evaluate_significance(memory_content)

    async def visualize_knowledge_graph(self, output_file: Optional[str] = None):
        """Visualize the knowledge graph using pyvis."""
        if not self.metrics["knowledge_graph"]:
            logger.warning("No knowledge graph data available")
            console.print(f"[{self.theme['warning']}]No knowledge graph data available[/{self.theme['warning']}]")
            return False
        
        try:
            console.print(f"[{self.theme['primary']}]Visualizing knowledge graph...[/{self.theme['primary']}]")
            
            kg_data = self.metrics["knowledge_graph"]
            nodes = kg_data.get("nodes", [])
            edges = kg_data.get("edges", [])
            
            if not nodes:
                logger.warning("Knowledge graph has no nodes")
                console.print(f"[{self.theme['warning']}]Knowledge graph has no nodes[/{self.theme['warning']}]")
                return False
            
            # Create network graph
            net = Network(height="750px", width="100%", notebook=False, directed=True)
            
            # Add nodes
            for node in nodes:
                node_id = node.get("id")
                label = node.get("label", node_id)
                node_type = node.get("type", "concept")
                
                # Determine node color based on type
                color = {
                    "concept": "#6929c4",  # Purple for concepts
                    "entity": "#1192e8",  # Blue for entities
                    "memory": "#005d5d",  # Teal for memories
                    "insight": "#fa4d56",  # Red for insights
                    "question": "#570408"  # Dark red for questions
                }.get(node_type, "#8a3ffc")
                
                importance = node.get("importance", 0.5)
                size = 10 + (importance * 20)  # Size based on importance
                
                net.add_node(
                    node_id, 
                    label=label, 
                    title=node.get("description", label),
                    color=color,
                    size=size
                )
            
            # Add edges
            for edge in edges:
                source = edge.get("source")
                target = edge.get("target")
                
                if source and target:
                    weight = edge.get("weight", 0.5)
                    edge_type = edge.get("type", "related")
                    
                    # Edge width based on weight
                    width = 1 + (weight * 4)
                    
                    # Edge color based on type
                    color = {
                        "derived_from": "#8a3ffc",  # Purple
                        "causes": "#fa4d56",  # Red
                        "part_of": "#1192e8",  # Blue
                        "similar_to": "#005d5d",  # Teal
                        "opposite_of": "#570408"  # Dark red
                    }.get(edge_type, "#6929c4")
                    
                    net.add_edge(
                        source, 
                        target, 
                        value=width,
                        title=edge.get("label", edge_type),
                        color=color
                    )
            
            # Set physics layout
            net.barnes_hut(
                gravity=-80000,
                central_gravity=0.3,
                spring_length=200,
                spring_strength=0.05,
                damping=0.09
            )
            
            # Save to file
            output_file = output_file or os.path.join(self.session_dir, "knowledge_graph.html")
            net.save_graph(output_file)
            
            logger.info(f"Knowledge graph visualization saved to {output_file}")
            console.print(f"[{self.theme['success']}]Knowledge graph visualization saved to {output_file}[/{self.theme['success']}]")
            return output_file
        
        except Exception as e:
            logger.error(f"Error visualizing knowledge graph: {e}")
            console.print(f"[{self.theme['error']}]Error visualizing knowledge graph: {e}[/{self.theme['error']}]")
            return False
    
    async def visualize_metrics(self, output_dir: Optional[str] = None):
        """Create visualizations of the collected metrics."""
        try:
            output_dir = output_dir or self.session_dir
            os.makedirs(output_dir, exist_ok=True)
            
            console.print(f"[{self.theme['primary']}]Creating metrics visualizations...[/{self.theme['primary']}]")
            
            metrics_to_plot = [
                "self_awareness_depth",
                "integration_effectiveness",
                "surprise_significance",
                "reflective_coherence",
                "kg_connectivity",
                "spiral_phase_maturity"
            ]
            
            plt.figure(figsize=(15, 10))
            
            for i, metric_name in enumerate(metrics_to_plot, 1):
                plt.subplot(2, 3, i)
                
                values = self.metrics.get(metric_name, [])
                if not values:
                    continue
                
                plt.plot(values, marker='o', linestyle='-', linewidth=2, markersize=5)
                plt.title(metric_name.replace('_', ' ').title())
                plt.xlabel('Measurement')
                plt.ylabel('Value')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Add trend line
                if len(values) > 1:
                    z = np.polyfit(range(len(values)), values, 1)
                    p = np.poly1d(z)
                    plt.plot(range(len(values)), p(range(len(values))), "r--", alpha=0.7)
            
            plt.tight_layout()
            output_file = os.path.join(output_dir, "evolution_metrics.png")
            plt.savefig(output_file)
            
            # Also save memory count plot
            plt.figure(figsize=(10, 6))
            for memory_type, counts in self.metrics["memory_count"].items():
                if counts:
                    plt.plot(counts, label=memory_type.upper())
            
            plt.title("Memory Count Evolution")
            plt.xlabel("Measurement")
            plt.ylabel("Count")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            memory_file = os.path.join(output_dir, "memory_evolution.png")
            plt.savefig(memory_file)
            
            logger.info(f"Metrics visualizations saved to {output_dir}")
            console.print(f"[{self.theme['success']}]Metrics visualizations saved to {output_dir}[/{self.theme['success']}]")
            return True
        
        except Exception as e:
            logger.error(f"Error visualizing metrics: {e}")
            console.print(f"[{self.theme['error']}]Error visualizing metrics: {e}[/{self.theme['error']}]")
            return False
    
    async def export_metrics(self, output_file: Optional[str] = None):
        """Export collected metrics to a JSON file."""
        try:
            output_file = output_file or os.path.join(self.session_dir, "metrics.json")
            
            console.print(f"[{self.theme['primary']}]Exporting metrics to {output_file}...[/{self.theme['primary']}]")
            
            # Prepare export data
            export_data = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    k: v for k, v in self.metrics.items() 
                    if k != "knowledge_graph"  # Exclude large KG data
                }
            }
            
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics exported to {output_file}")
            console.print(f"[{self.theme['success']}]Metrics exported to {output_file}[/{self.theme['success']}]")
            return output_file
        
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            console.print(f"[{self.theme['error']}]Error exporting metrics: {e}[/{self.theme['error']}]")
            return False
    
    async def continuous_monitoring(self, interval: int = 60, duration: int = 3600):
        """
        Continuously monitor Lucidia's evolution over a period of time.
        
        Args:
            interval: Polling interval in seconds
            duration: Total monitoring duration in seconds
        """
        start_time = time.time()
        end_time = start_time + duration
        
        console.print(Panel(
            f"Starting continuous monitoring for {duration//60} minutes\n"
            f"Polling every {interval} seconds for updates",
            style=f"{self.theme['primary']} bold",
            title="Continuous Monitoring"
        ))
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                monitor_task = progress.add_task("Monitoring Lucidia's evolution...", total=duration)
                
                while time.time() < end_time:
                    elapsed = time.time() - start_time
                    progress.update(monitor_task, completed=elapsed, description=f"Monitoring Lucidia's evolution... {elapsed//60:.0f}m {elapsed%60:.0f}s elapsed")
                    
                    # Fetch current state
                    await self.fetch_system_state()
                    
                    # Check if a dream is active
                    if not self.active_dream_session:
                        # Check system idle status
                        response = await self._request_with_retry(
                            self.session.get, 
                            f"{self.dream_api_url}/api/system/idle"
                        )
                        if response:
                            idle_data = await response.json()
                            
                            if idle_data.get("idle", False) and idle_data.get("can_dream", False):
                                console.print(f"[{self.theme['warning']}]System idle detected, initiating dream session...[/{self.theme['warning']}]")
                                await self.start_dream_session()
                    
                    await asyncio.sleep(interval)
                
                progress.update(monitor_task, completed=duration, description=f"Completed {duration//60} minutes of monitoring")
            
            # Create final visualizations
            await self.visualize_metrics()
            await self.visualize_knowledge_graph()
            
            if self.config["export_metrics"]:
                await self.export_metrics()
            
            console.print(Panel(
                f"Continuous monitoring completed.\nSession data saved to {self.session_dir}",
                style=f"{self.theme['success']} bold",
                title="Monitoring Complete"
            ))
        
        except KeyboardInterrupt:
            console.print(f"[{self.theme['warning']}]Monitoring interrupted by user[/{self.theme['warning']}]")
        except Exception as e:
            logger.error(f"Error during continuous monitoring: {e}")
            console.print(f"[{self.theme['error']}]Error during continuous monitoring: {e}[/{self.theme['error']}]")
    
    async def display_evolution_dashboard(self):
        """Display an interactive dashboard of Lucidia's evolution metrics."""
        # Create a layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="metrics", ratio=2),
            Layout(name="details", ratio=3)
        )
        
        layout["details"].split_column(
            Layout(name="spiral_phase"),
            Layout(name="memory_stats"),
            Layout(name="insights")
        )
        
        # Header content
        def make_header():
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            session = f"Session: {self.session_id}"
            phase = f"Current Phase: {self.metrics['current_spiral_phase'] or 'Unknown'}"
            return Panel(
                f"{session}   |   {phase}   |   Last Updated: {now}",
                title=f"[bold]Lucidia Evolution Dashboard[/bold]",
                style=self.theme["primary"]
            )
        
        # Metrics table
        def make_metrics_table():
            table = Table(title="Evolution Metrics", box=box.ROUNDED)
            table.add_column("Metric", style=self.theme["primary"])
            table.add_column("Current", style=self.theme["success"])
            table.add_column("Change", style=self.theme["accent"])
            
            # Add rows for each metric
            for metric_name in [
                "self_awareness_depth",
                "integration_effectiveness",
                "surprise_significance",
                "reflective_coherence",
                "kg_connectivity",
                "spiral_phase_maturity"
            ]:
                values = self.metrics.get(metric_name, [])
                if not values:
                    continue
                
                current = values[-1]
                
                # Calculate change
                if len(values) > 1:
                    change = current - values[-2]
                    change_str = f"{change:+.4f}"
                    change_style = self.theme["success"] if change > 0 else self.theme["error"]
                else:
                    change_str = "N/A"
                    change_style = self.theme["warning"]
                
                table.add_row(
                    metric_name.replace('_', ' ').title(),
                    f"{current:.4f}",
                    Text(change_str, style=change_style)
                )
            
            return Panel(table, title="Metrics", border_style=self.theme["secondary"])
        
        # Spiral phase panel
        def make_spiral_panel():
            if not self.metrics["phase_history"]:
                return Panel("No phase history available", title="Spiral Phase Evolution", border_style=self.theme["info"])
            
            table = Table(box=None)
            table.add_column("Time", style=self.theme["info"])
            table.add_column("Phase", style=self.theme["secondary"])
            table.add_column("Reason", style=self.theme["accent"])
            
            for phase in reversed(self.metrics["phase_history"][-5:]):
                # Extract time from timestamp
                try:
                    time_str = datetime.fromisoformat(phase["timestamp"]).strftime("%H:%M:%S")
                except:
                    time_str = phase["timestamp"]
                
                table.add_row(
                    time_str,
                    phase["phase"],
                    phase.get("reason", "Unknown")
                )
            
            return Panel(table, title="Spiral Phase Evolution", border_style=self.theme["secondary"])
        
        # Memory stats panel
        def make_memory_stats_panel():
            memory_counts = self.metrics["memory_count"]
            
            table = Table(box=None)
            table.add_column("Memory Type", style=self.theme["primary"])
            table.add_column("Count", style=self.theme["secondary"])
            table.add_column("Change")
            
            for memory_type, counts in memory_counts.items():
                if not counts:
                    continue
                
                current = counts[-1]
                
                # Calculate change
                if len(counts) > 1:
                    change = current - counts[-2]
                    change_str = f"{change:+d}"
                    change_style = self.theme["success"] if change > 0 else self.theme["error"] if change < 0 else "white"
                else:
                    change_str = "N/A"
                    change_style = self.theme["warning"]
                
                table.add_row(
                    memory_type.upper(),
                    str(current),
                    Text(change_str, style=change_style)
                )
            
            return Panel(table, title="Memory Statistics", border_style=self.theme["secondary"])
        
        # Insights panel
        def make_insights_panel():
            insights = self.metrics["dream_insights"][-5:] if self.metrics["dream_insights"] else []
            
            if not insights:
                return Panel("No insights available", title="Recent Insights", border_style=self.theme["info"])
            
            table = Table(box=None)
            table.add_column("Type", style=self.theme["accent"])
            table.add_column("Content")
            table.add_column("Sig.", style=self.theme["secondary"])
            
            for insight in reversed(insights):
                table.add_row(
                    insight.get("type", "unknown"),
                    insight.get("content", "")[:50] + ("..." if len(insight.get("content", "")) > 50 else ""),
                    f"{insight.get('significance', 0):.2f}"
                )
            
            return Panel(table, title="Recent Insights", border_style=self.theme["secondary"])
        
        # Footer content
        def make_footer():
            return Panel(
                "Press Ctrl+C to exit | Updated every 5 seconds",
                style=self.theme["info"],
                border_style=self.theme["info"]
            )
        
        # Update and render the layout
        try:
            interval = self.config["visualization_update_interval"]
            
            console.print(f"[{self.theme['primary']} bold]Starting Lucidia Evolution Dashboard[/{self.theme['primary']} bold]")
            console.print(f"[{self.theme['info']}]Press Ctrl+C to exit[/{self.theme['info']}]")
            
            with Live(layout, refresh_per_second=1/interval) as live:
                while True:
                    # Fetch latest state
                    await self.fetch_system_state()
                    
                    # Update layout
                    layout["header"].update(make_header())
                    layout["metrics"].update(make_metrics_table())
                    layout["spiral_phase"].update(make_spiral_panel())
                    layout["memory_stats"].update(make_memory_stats_panel())
                    layout["insights"].update(make_insights_panel())
                    layout["footer"].update(make_footer())
                    
                    await asyncio.sleep(interval)
        except KeyboardInterrupt:
            console.print(f"[{self.theme['warning']}]Dashboard closed by user[/{self.theme['warning']}]")
    
    async def display_parameters(self, filter_path: Optional[str] = None):
        """Display the system parameters in a hierarchical table."""
        params = await self.get_parameters()
        
        if not params:
            console.print(f"[{self.theme['error']}]Unable to retrieve parameters[/{self.theme['error']}]")
            return
        
        if filter_path and filter_path in params:
            # Display only a specific parameter branch
            params = {filter_path: params[filter_path]}
        
        def print_param_tree(param_dict, path="", indent=0):
            # Create and print a table for this level
            table = Table(box=box.ROUNDED)
            table.add_column("Parameter", style=self.theme["primary"])
            table.add_column("Value", style=self.theme["secondary"])
            table.add_column("Type", style=self.theme["info"])
            table.add_column("Metadata")
            
            for key, data in param_dict.items():
                full_path = f"{path}.{key}" if path else key
                
                if isinstance(data, dict) and 'value' not in data:
                    # This is a branch node, not a leaf parameter
                    table.add_row(
                        f"[bold {self.theme['secondary']}]{key}[/bold {self.theme['secondary']}]",
                        "[dim]<branch>[/dim]",
                        "[dim]N/A[/dim]",
                        f"[dim]{len(data)} child params[/dim]"
                    )
                else:
                    # This is a leaf parameter
                    # Handle both formats: direct value or dict with 'value' key
                    if isinstance(data, dict) and 'value' in data:
                        # Parameter is already in the expected format
                        value = data.get('value', 'undefined')
                        param_type = data.get('type', 'unknown')
                        
                        # Format metadata
                        metadata = []
                        if 'min' in data:
                            metadata.append(f"min={data['min']}")
                        if 'max' in data:
                            metadata.append(f"max={data['max']}")
                        if 'locked' in data and data['locked']:
                            metadata.append(f"[{self.theme['error']}]locked[/{self.theme['error']}]")
                        if 'description' in data:
                            desc = data['description']
                            if len(desc) > 30:
                                desc = desc[:27] + "..."
                            metadata.append(f"\"{desc}\"")
                    else:
                        # Parameter is a direct value (from parameter manager)
                        value = data
                        param_type = type(data).__name__
                        metadata = []
                    
                    metadata_str = ", ".join(metadata) if metadata else ""                    
                    
                    table.add_row(
                        key,
                        str(value),
                        param_type,
                        metadata_str
                    )
            
            console.print(table)
            
            # Recursively print child branches
            for key, data in param_dict.items():
                if isinstance(data, dict) and 'value' not in data:
                    full_path = f"{path}.{key}" if path else key
                    console.print(f"\n[bold {self.theme['primary']}]Branch: {full_path}[/bold {self.theme['primary']}]")
                    print_param_tree(data, full_path, indent + 1)
        
        console.print(Panel(
            f"System Parameters{' (filtered: ' + filter_path + ')' if filter_path else ''}",
            style=f"{self.theme['success']} bold"
        ))
        
        print_param_tree(params)
    
    async def interactive_parameter_edit(self):
        """Interactive parameter editing mode."""
        params = await self.get_parameters()
        
        if not params:
            console.print(f"[{self.theme['error']}]Unable to retrieve parameters[/{self.theme['error']}]")
            return
        
        console.print(Panel("Interactive Parameter Editing Mode", style=f"{self.theme['primary']} bold"))
        console.print("Type 'exit' to return to main menu\n")
        
        # First display all top-level parameters
        table = Table(title="Top-Level Parameter Categories", box=box.ROUNDED)
        table.add_column("Category", style=self.theme["secondary"])
        table.add_column("Description")
        
        for category, data in params.items():
            description = ""            
            if isinstance(data, dict):
                if 'description' in data:
                    description = data['description']
                else:
                    child_count = len(data)
                    description = f"Contains {child_count} parameters/categories"
            
            table.add_row(f"[{self.theme['secondary']}]{category}[/{self.theme['secondary']}]", description)
        
        console.print(table)
        console.print("\nTo browse a category, type its name. To edit a parameter, use 'edit <param_path>'\n")
        
        while True:
            command = console.input(f"[{self.theme['success']} bold]parameter>[/{self.theme['success']} bold] ")
            
            if command.lower() == 'exit':
                break
            elif command.lower().startswith('help'):
                console.print(Panel("\n".join([
                    "Available commands:",
                    "  <category>       - Browse a parameter category",
                    "  edit <path>      - Edit a specific parameter value",
                    "  show <path>      - Show details of a parameter",
                    "  search <term>    - Search for parameters containing term",
                    "  reset            - Return to top-level view",
                    "  exit             - Exit parameter editing mode"
                ]), title="Help", border_style=self.theme["primary"]))
            elif command.lower() == 'reset':
                await self.display_parameters()
            elif command.lower().startswith('edit '):
                param_path = command[5:].strip()
                await self._edit_parameter(param_path)
            elif command.lower().startswith('show '):
                param_path = command[5:].strip()
                param_data = await self.get_parameter(param_path)
                
                if not param_data:
                    console.print(f"[{self.theme['error']}]Parameter {param_path} not found[/{self.theme['error']}]")
                    continue
                
                # Display detailed parameter info
                console.print(Panel(f"Parameter: {param_path}", style=self.theme["secondary"]))
                for key, value in param_data.items():
                    console.print(f"[{self.theme['primary']}]{key}:[/{self.theme['primary']}] {value}")
            elif command.lower().startswith('search '):
                search_term = command[7:].strip().lower()
                await self._search_parameters(params, search_term)
            else:
                # Assume it's a category to browse
                category = command.strip()
                if category in params:
                    await self.display_parameters(category)
                else:
                    console.print(f"[{self.theme['error']}]Category or command '{category}' not found. Type 'help' for available commands.[/{self.theme['error']}]")
    
    async def _search_parameters(self, params: Dict[str, Any], search_term: str, current_path: str = ""):
        """Search for parameters containing the search term."""
        results = []
        
        def search_recursive(params_dict, path=""):
            for key, data in params_dict.items():
                full_path = f"{path}.{key}" if path else key
                
                # Check if this parameter matches
                if search_term in full_path.lower():
                    results.append((full_path, data))
                
                # If this is a branch, search its children
                if isinstance(data, dict) and 'value' not in data:
                    search_recursive(data, full_path)
                elif isinstance(data, dict) and 'description' in data:
                    # Also search in description
                    if search_term in str(data.get('description', '')).lower():
                        results.append((full_path, data))
        
        search_recursive(params)
        
        if not results:
            console.print(f"[{self.theme['warning']}]No parameters found containing '{search_term}'[/{self.theme['warning']}]")
            return
        
        # Display results
        table = Table(title=f"Search Results for '{search_term}'", box=box.ROUNDED)
        table.add_column("Parameter Path", style=self.theme["primary"])
        table.add_column("Value", style=self.theme["secondary"])
        table.add_column("Description")
        
        for path, data in results:
            if isinstance(data, dict) and 'value' in data:
                value = str(data.get('value', 'N/A'))
                description = data.get('description', '')
                table.add_row(path, value, description)
            else:
                table.add_row(f"[{self.theme['secondary']}]{path}[/{self.theme['secondary']}]", "[dim]<branch>[/dim]", "")
        
        console.print(table)
    
    async def _edit_parameter(self, param_path: str):
        """Edit a specific parameter."""
        param_data = await self.get_parameter(param_path)
        
        if not param_data:
            console.print(f"[{self.theme['error']}]Parameter {param_path} not found[/{self.theme['error']}]")
            return
        
        # Check if parameter is locked
        if param_data.get('locked', False):
            console.print(f"[{self.theme['error']}]Parameter {param_path} is locked and cannot be modified[/{self.theme['error']}]")
            return
        
        # Display current value and metadata
        console.print(f"\nEditing parameter: [bold {self.theme['primary']}]{param_path}[/bold {self.theme['primary']}]")
        console.print(f"Current value: {param_data.get('value')}")
        console.print(f"Type: {param_data.get('type', 'unknown')}")
        
        if 'description' in param_data:
            console.print(f"Description: {param_data['description']}")
        
        if 'min' in param_data and 'max' in param_data:
            console.print(f"Valid range: {param_data['min']} to {param_data['max']}")
        
        # Get new value
        new_value_str = console.input(f"\nEnter new value (or 'cancel' to abort): ")
        
        if new_value_str.lower() == 'cancel':
            console.print(f"[{self.theme['warning']}]Edit canceled[/{self.theme['warning']}]")
            return
        
        # Convert to appropriate type
        param_type = param_data.get('type', 'str')
        try:
            if param_type == 'int':
                new_value = int(new_value_str)
            elif param_type == 'float':
                new_value = float(new_value_str)
            elif param_type == 'bool':
                new_value = new_value_str.lower() in ('true', 'yes', '1', 'y', 't')
            else:
                new_value = new_value_str
                
            # Validate range if applicable
            if 'min' in param_data and 'max' in param_data:
                min_val = param_data['min']
                max_val = param_data['max']
                
                if new_value < min_val or new_value > max_val:
                    console.print(f"[{self.theme['error']}]Value must be between {min_val} and {max_val}[/{self.theme['error']}]")
                    return
        except ValueError:
            console.print(f"[{self.theme['error']}]Invalid value for type {param_type}[/{self.theme['error']}]")
            return
        
        # Confirm change
        if Confirm.ask(f"Change parameter {param_path} from {param_data.get('value')} to {new_value}?"):
            success = await self.update_parameter(param_path, new_value)
            
            if success:
                console.print(f"[{self.theme['success']}]Parameter {param_path} updated successfully[/{self.theme['success']}]")
            else:
                console.print(f"[{self.theme['error']}]Failed to update parameter {param_path}[/{self.theme['error']}]")
        else:
            console.print(f"[{self.theme['warning']}]Update canceled[/{self.theme['warning']}]")
    
    async def _request_with_retry(self, method, url, max_retries=3, initial_delay=1.0, max_delay=10.0, **kwargs):
        """Execute a request with exponential backoff retry logic.
        
        Args:
            method: HTTP method function (e.g., self.session.get, self.session.post)
            url: URL to request
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            **kwargs: Additional arguments to pass to the request method
            
        Returns:
            Response object if successful, None if all retries failed
        """
        retry_count = 0
        delay = initial_delay
        
        while True:
            try:
                response = await method(url, **kwargs)
                if response.status == 200:
                    return response
                
                # For non-200 responses, we may still want to return the response
                # but log the issue
                if response.status < 500:  # Client errors generally shouldn't be retried
                    logger.warning(f"Request to {url} returned status {response.status}")
                    logger.error(f"Failed to get parameters: {response.status}")
                    return response
                    
                error_data = await response.text()
                logger.error(f"Request to {url} failed with status {response.status}: {error_data}")
                # Close the response to avoid resource leaks
                response.close()
            except aiohttp.ClientError as e:
                logger.error(f"Request to {url} failed: {e}")
            
            retry_count += 1
            if retry_count >= max_retries:
                # Ask user if they want to retry for critical operations
                if self.config.get("interactive_mode", True):
                    retry_prompt = Confirm.ask(f"Connection to {url} failed after {max_retries} attempts. Retry?", default=True)
                    if not retry_prompt:
                        return None
                    # Reset retry count if user wants to continue retrying
                    retry_count = 0
                    delay = initial_delay
                    continue
                else:
                    logger.error(f"Max retries ({max_retries}) exceeded for {url}")
                    return None
            
            # Calculate delay with jitter for exponential backoff
            jitter = random.uniform(0, 0.1 * delay)
            adjusted_delay = min(delay + jitter, max_delay)
            logger.info(f"Retrying request to {url} in {adjusted_delay:.2f}s (attempt {retry_count}/{max_retries})")
            await asyncio.sleep(adjusted_delay)
            
            # Exponential backoff
            delay = min(delay * 2, max_delay)

    async def start_chat_session(self, depth: float = 0.7, creativity: float = 0.5, local: bool = False):
        """Start an interactive chat session with Lucidia.
        
        This starts a multi-turn conversation with memory creation and retrieval capabilities.
        Users can use slash commands to perform various actions.
        
        Args:
            depth: Reflection depth (0.0-1.0)
            creativity: Creativity level (0.0-1.0)
            local: Use local LLM instead of Dream API
        """
        if not self.session:
            await self.connect()
            
        # If local flag is provided, always use local LLM
        use_local = local or self.config["use_local_llm"]
        
        if use_local and not self.llm_client:
            self.llm_client = LMStudioClient(self.config)
            if not await self.llm_client.connect():
                console.print(f"[{self.theme['error']}]Failed to connect to LM Studio server[/{self.theme['error']}]")
                return
                
        # Create a chat context to maintain conversation state
        chat_context = {
            "messages": [],
            "memories": [],
            "session_id": f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "depth": depth,
            "creativity": creativity,
            "memory_index": {}  # Index to quickly look up memories by ID
        }
        
        # Create the chat session directory
        chat_dir = os.path.join(self.session_dir, chat_context["session_id"])
        os.makedirs(chat_dir, exist_ok=True)
        
        # Display welcome message and instructions
        console.print(Panel(
            f"[{self.theme['primary']}]Welcome to Lucidia Chat![/{self.theme['primary']}]\n\n"
            f"[{self.theme['secondary']}]Available slash commands:[/{self.theme['secondary']}]\n\n"
            "/help - Show this help message\n"
            "/memory <text> - Create a new memory\n"
            "/recall <query> - Search for related memories\n"
            "/forget <memory_id> - Delete a memory\n"
            "/dream - Generate reflections based on conversation\n"
            "/list - List all memories in the current session\n"
            "/save - Save the current chat session\n"
            "/exit - Exit the chat session\n\n"
            f"[{self.theme['success']}]Type your message or use a slash command to begin.[/{self.theme['success']}]",
            title="Lucidia Chat", border_style=self.theme["primary"]
        ))
        
        # Main chat loop
        running = True
        while running:
            # Get user input
            try:
                user_input = await async_input(f"[bold {self.theme['secondary']}]You:[/bold {self.theme['secondary']}] ")
            except (KeyboardInterrupt, EOFError):
                running = False
                break
                
            # Skip empty messages
            if not user_input.strip():
                continue
                
            # Add user message to context
            chat_context["messages"].append({"role": "user", "content": user_input})
            
            # Handle slash commands
            if user_input.startswith("/"):
                command_parts = user_input.split(" ", 1)
                command = command_parts[0][1:].lower()
                args = command_parts[1] if len(command_parts) > 1 else ""
                
                if command == "help":
                    # Display help message
                    await self._display_chat_help()
                    
                elif command == "memory":
                    # Create a new memory
                    if not args:
                        console.print(f"[{self.theme['warning']}]Please provide text for the memory[/{self.theme['warning']}]")
                        continue
                        
                    # Create memory - fix for duplicate output
                    memory = await self._create_memory(args, chat_context, print_output=True)
                    if memory:
                        # Don't print this message as it's already handled in _create_memory with print_output=True
                        chat_context["memories"].append(memory)
                        chat_context["memory_index"][memory["id"]] = memory
                    
                elif command == "recall":
                    # Search for related memories
                    if not args:
                        console.print(f"[{self.theme['warning']}]Please provide a search query[/{self.theme['warning']}]")
                        continue
                        
                    await self._recall_memories(args, chat_context)
                    
                elif command == "forget":
                    # Delete a memory
                    if not args:
                        console.print(f"[{self.theme['warning']}]Please provide a memory ID to forget[/{self.theme['warning']}]")
                        continue
                        
                    await self._forget_memory(args, chat_context)
                    
                elif command == "dream":
                    # Generate reflections
                    await self._generate_chat_reflection(chat_context, use_local)
                    
                elif command == "list":
                    # List all memories
                    await self._list_memories(chat_context)
                    
                elif command == "save":
                    # Save the chat session
                    saved_path = await self._save_chat_session(chat_context)
                    console.print(f"[{self.theme['success']}]Chat session saved to {saved_path}[/{self.theme['success']}]")
                    
                elif command == "exit":
                    # Exit the chat session
                    running = False
                    console.print(f"[{self.theme['warning']}]Exiting chat session...[/{self.theme['warning']}]")
                    break
                    
                else:
                    # Unknown command
                    console.print(f"[{self.theme['warning']}]Unknown command: {command}. Type /help for available commands.[/{self.theme['warning']}]")
                    
            else:
                # Normal message - process with LLM or Dream API
                response = await self._process_chat_message(user_input, chat_context, use_local)
                console.print(f"[bold {self.theme['primary']}]Lucidia:[/bold {self.theme['primary']}] {response}")
                
        # Save before exiting
        if chat_context["messages"]:
            saved_path = await self._save_chat_session(chat_context)
            console.print(f"[{self.theme['success']}]Chat session saved to {saved_path}[/{self.theme['success']}]")
            
        console.print(f"[{self.theme['primary']}]Chat session ended[/{self.theme['primary']}]")
        
    async def _display_chat_help(self):
        """Display help information for the chat session."""
        console.print(Panel(
            f"[{self.theme['secondary']}]Available slash commands:[/{self.theme['secondary']}]\n\n"
            "/help - Show this help message\n"
            "/memory <text> - Create a new memory\n"
            "/recall <query> - Search for related memories\n"
            "/forget <memory_id> - Delete a memory\n"
            "/dream - Generate reflections based on conversation\n"
            "/list - List all memories in the current session\n"
            "/save - Save the current chat session\n"
            "/exit - Exit the chat session",
            title="Lucidia Chat Help", border_style=self.theme["primary"]
        ))
        
    async def _create_memory(self, content: str, chat_context: Dict[str, Any], significance_override: float = None, print_output: bool = True) -> Dict[str, Any]:
        """Create a new memory from chat content.
        
        Args:
            content: The memory content
            chat_context: The current chat context
            significance_override: Override the default significance value
            print_output: Whether to print creation messages to console
            
        Returns:
            The created memory object or None if failed
        """
        try:
            # Check Docker server connectivity first
            docker_available = False
            if self.session and not self.config["use_local_llm"]:
                try:
                    # Try to ping the Dream API to verify connectivity
                    health_response = await self._request_with_retry(
                        self.session.get,
                        f"{self.dream_api_url}/health",
                        max_retries=1,  # Quick check
                        initial_delay=0.5
                    )
                    docker_available = health_response and health_response.status == 200
                    logger.info(f"Dream API connectivity: {'Available' if docker_available else 'Unavailable'}")
                except Exception as e:
                    logger.warning(f"Docker container connectivity check failed: {e}")
                    docker_available = False
            
            # Generate significance score based on available methods
            significance = 0.5  # Better default minimum significance
            
            if docker_available:
                # Use Docker container for significance calculation
                try:
                    response = await self._request_with_retry(
                        self.session.post,
                        f"{self.dream_api_url}/api/evaluate_significance",
                        json={"content": content},
                        max_retries=2
                    )
                    if response and response.status == 200:
                        result = await response.json()
                        if "significance" in result:
                            significance = result["significance"]
                            logger.info(f"Using Docker-calculated significance: {significance}")
                except Exception as e:
                    logger.error(f"Error calculating significance via Docker: {e}")
            elif self.llm_client and significance_override is None:
                # Fallback to local LLM
                try:
                    significance = await self.llm_client.evaluate_significance(content)
                    logger.info(f"Using local LLM significance: {significance}")
                except Exception as e:
                    logger.error(f"Error calculating significance via local LLM: {e}")
            elif significance_override is not None:
                # Use override if provided
                significance = significance_override
                logger.info(f"Using override significance: {significance}")
            else:
                logger.info(f"Using default significance: {significance}")
            
            # Create memory object
            memory_id = f"m_{len(chat_context['memories']) + 1}_{int(time.time())}"
            memory = {
                "id": memory_id,
                "content": content,
                "significance": significance,
                "created_at": datetime.now().isoformat(),
                "metadata": {
                    "source": "chat",
                    "session_id": chat_context["session_id"],
                    "creator": "user"
                }
            }
            
            # Save memory to file
            memory_path = os.path.join(self.session_dir, chat_context["session_id"], f"{memory_id}.json")
            os.makedirs(os.path.dirname(memory_path), exist_ok=True)
            with open(memory_path, "w") as f:
                json.dump(memory, f, indent=2)
                
            # Try to add memory to Dream API if connected
            if docker_available:
                try:
                    # First create the memory in the standard API
                    response = await self._request_with_retry(
                        self.session.post,
                        f"{self.dream_api_url}/api/memories",
                        json={
                            "content": content,
                            "significance": significance,
                            "metadata": memory["metadata"]
                        }
                    )
                    
                    if response and response.status == 200:
                        result = await response.json()
                        if "memory_id" in result:
                            # Update with server-assigned ID
                            server_memory_id = result["memory_id"]
                            memory["server_id"] = server_memory_id
                            logger.info(f"Memory added to Dream API with ID: {server_memory_id}")
                            memory["persisted"] = True
                            
                            # Output consolidated in one place
                            if print_output:
                                console.print(f"[{self.theme['success']}]Memory [bold]{server_memory_id}[/bold] persisted across sessions (significance: {significance:.2f})[/{self.theme['success']}]")
                        else:
                            logger.warning(f"Failed to get memory ID from Dream API")
                            if print_output:
                                console.print(f"[{self.theme['success']}]Memory created with ID: [bold]{memory_id}[/bold] (significance: {significance:.2f})[/{self.theme['success']}]")
                    else:
                        logger.warning(f"Failed to add memory to Dream API: {response.status if response else 'No response'}")
                        if print_output:
                            console.print(f"[{self.theme['success']}]Memory created with ID: [bold]{memory_id}[/bold] (significance: {significance:.2f})[/{self.theme['success']}]")
                except Exception as e:
                    logger.error(f"Error adding memory to Dream API: {e}")
                    if print_output:
                        console.print(f"[{self.theme['warning']}]Warning: Memory only saved locally - Docker not available[/{self.theme['warning']}]")
                        console.print(f"[{self.theme['success']}]Memory created with ID: [bold]{memory_id}[/bold] (significance: {significance:.2f})[/{self.theme['success']}]")
            else:
                logger.warning("Skipping Dream API memory creation - Docker not available")
                if print_output:
                    console.print(f"[{self.theme['success']}]Memory created with ID: [bold]{memory_id}[/bold] (significance: {significance:.2f})[/{self.theme['success']}]")
                
            return memory
            
        except Exception as e:
            logger.error(f"Error creating memory: {e}")
            if print_output:
                console.print(f"[{self.theme['error']}]Error creating memory: {e}[/{self.theme['error']}]")
            return None
            
    async def _recall_memories(self, query: str, chat_context: Dict[str, Any]):
        """Search for memories related to the query.
        
        Args:
            query: The search query
            chat_context: The current chat context
        """
        try:
            console.print(f"[{self.theme['primary']}]Searching for memories related to: {query}[/{self.theme['primary']}]")
            
            # Expand query with common synonyms and related terms
            expanded_query_terms = [query.lower()]
            
            # Common synonym mappings for frequent recall terms
            synonyms = {
                "name": ["identity", "title", "label", "handle", "alias", "moniker"],
                "my": ["user", "person", "speaker"],
                "i": ["user", "person", "speaker"],
                "me": ["user", "person", "speaker"],
                "remember": ["recall", "memory", "stored", "saved"],
                "said": ["mentioned", "stated", "noted", "told", "expressed"],
                "told": ["mentioned", "stated", "noted", "said", "expressed"],
            }
            
            # Add expanded query terms
            query_words = query.lower().split()
            for word in query_words:
                if word in synonyms:
                    expanded_query_terms.extend(synonyms[word])
            
            logger.info(f"Expanded query '{query}' to include: {expanded_query_terms}")
            
            # First, check local memories with expanded terms
            local_results = []
            
            # Score function to rank results
            def score_memory(memory_content, query_terms):
                content_lower = memory_content.lower()
                # Base score starts at 0
                score = 0
                
                # Add points for exact matches
                for term in query_terms:
                    if term in content_lower:
                        # Exact match gets higher score
                        score += 1
                        
                        # Bonus for term appearing at beginning of content
                        if content_lower.startswith(term):
                            score += 0.5
                            
                # Add points for containing all original query words
                if all(word in content_lower for word in query_words):
                    score += 2
                    
                return score
            
            # Score all memories
            scored_memories = []
            for memory in chat_context["memories"]:
                memory_score = score_memory(memory["content"], expanded_query_terms)
                if memory_score > 0:
                    # Clone the memory and add score
                    scored_memory = memory.copy()
                    scored_memory["score"] = memory_score
                    scored_memories.append(scored_memory)
            
            # Sort by score (highest first)
            scored_memories.sort(key=lambda m: m["score"], reverse=True)
            
            # Select top results
            local_results = scored_memories
            
            # Then try to search from Dream API if available
            api_results = []
            if self.session and not self.config["use_local_llm"]:
                try:
                    # Include expanded terms in the API search
                    expanded_query_str = " OR ".join(expanded_query_terms)
                    response = await self._request_with_retry(
                        self.session.get,
                        f"{self.dream_api_url}/api/memories/search?query={urllib.parse.quote(expanded_query_str)}"
                    )
                    
                    if response and response.status == 200:
                        results = await response.json()
                        api_results = results.get("memories", [])
                except Exception as e:
                    logger.error(f"Error searching memories via API: {e}")
            
            # Combine and display results
            all_results = local_results + [m for m in api_results if m.get("id") not in [mem.get("id") for mem in local_results]]
            
            if all_results:
                # Display table of results
                table = Table(title=f"Memory Search Results for '{query}'", box=box.ROUNDED)
                table.add_column("ID", style=self.theme["secondary"])
                table.add_column("Content")
                table.add_column("Significance", style=self.theme["primary"])
                table.add_column("Relevance", style=self.theme["accent"])
                table.add_column("Created", style=self.theme["info"])
                
                for memory in all_results:
                   # Format score if available, otherwise blank
                    score_display = f"{memory.get('score', 0):.2f}" if "score" in memory else ""
                    
                    table.add_row(
                        memory.get("id", "Unknown"),
                        memory.get("content", "")[:50] + ("..." if len(memory.get("content", "")) > 50 else ""),
                        str(round(memory.get("significance", 0) * 100) / 100),
                        score_display,
                        memory.get("created_at", "Unknown")
                    )
                
                console.print(table)
            else:
                console.print(f"[{self.theme['warning']}]No memories found for query: {query}[/{self.theme['warning']}]")
                
        except Exception as e:
            logger.error(f"Error recalling memories: {e}")
            console.print(f"[{self.theme['error']}]Error recalling memories: {e}[/{self.theme['error']}]")
            
    async def _forget_memory(self, memory_id: str, chat_context: Dict[str, Any]):
        """Delete a memory by ID.
        
        Args:
            memory_id: The ID of the memory to delete
            chat_context: The current chat context
        """
        try:
            # Check if memory exists in local context
            if memory_id in chat_context["memory_index"]:
                memory = chat_context["memory_index"][memory_id]
                
                # Remove from local context
                chat_context["memories"] = [m for m in chat_context["memories"] if m.get("id") != memory_id]
                del chat_context["memory_index"][memory_id]
                
                # Delete memory file
                memory_path = os.path.join(self.session_dir, chat_context["session_id"], f"{memory_id}.json")
                if os.path.exists(memory_path):
                    os.remove(memory_path)
                
                # Try to delete from Dream API if it has a server ID
                if "server_id" in memory and self.session and not self.config["use_local_llm"]:
                    try:
                        response = await self._request_with_retry(
                            self.session.delete,
                            f"{self.dream_api_url}/api/memories/{memory['server_id']}"
                        )
                        
                        if response and response.status == 200:
                            logger.info(f"Deleted memory {memory_id} from server")
                    except Exception as e:
                        logger.error(f"Error deleting memory from Dream API: {e}")
                
                console.print(f"[{self.theme['success']}]Memory {memory_id} has been forgotten[/{self.theme['success']}]")
            else:
                console.print(f"[{self.theme['warning']}]Memory with ID {memory_id} not found[/{self.theme['warning']}]")
                
        except Exception as e:
            logger.error(f"Error forgetting memory: {e}")
            console.print(f"[{self.theme['error']}]Error forgetting memory: {e}[/{self.theme['error']}]")
            
    async def _list_memories(self, chat_context: Dict[str, Any]):
        """List all memories in the current session.
        
        Args:
            chat_context: The current chat context
        """
        try:
            if chat_context["memories"]:
                # Display table of memories
                table = Table(title="Memories in Current Session", box=box.ROUNDED)
                table.add_column("ID", style=self.theme["secondary"])
                table.add_column("Content")
                table.add_column("Significance", style=self.theme["primary"])
                table.add_column("Created", style=self.theme["info"])
                
                for memory in chat_context["memories"]:
                    table.add_row(
                        memory.get("id", "Unknown"),
                        memory.get("content", "")[:50] + ("..." if len(memory.get("content", "")) > 50 else ""),
                        str(round(memory.get("significance", 0) * 100) / 100),
                        memory.get("created_at", "Unknown")
                    )
                
                console.print(table)
            else:
                console.print(f"[{self.theme['warning']}]No memories in the current session[/{self.theme['warning']}]")
                
        except Exception as e:
            logger.error(f"Error listing memories: {e}")
            console.print(f"[{self.theme['error']}]Error listing memories: {e}[/{self.theme['error']}]")
            
    async def _generate_chat_reflection(self, chat_context: Dict[str, Any], use_local: bool):
        """Generate a reflection on the current conversation.
        
        Args:
            chat_context: The current chat context
            use_local: Whether to use local LLM
        """
        try:
            console.print(f"[{self.theme['primary']}]Generating reflection...[/{self.theme['primary']}]")
            
            # Extract recent conversation and memories as context
            recent_messages = chat_context["messages"][-10:] if len(chat_context["messages"]) > 10 else chat_context["messages"]
            conversation_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent_messages])
            
            memories_for_reflection = chat_context["memories"]
            
            if use_local and self.llm_client:
                # Format memories for reflection
                memory_dicts = [{
                    "content": memory["content"],
                    "significance": memory["significance"],
                    "created_at": memory["created_at"]
                } for memory in memories_for_reflection]
                
                # Add current conversation as a temporary memory
                memory_dicts.append({
                    "content": f"Recent conversation:\n{conversation_text}",
                    "significance": 0.9,
                    "created_at": datetime.now().isoformat()
                })
                
                # Show progress during generation
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"[{self.theme['primary']}]Generating reflection...[/{self.theme['primary']}]"),
                    console=console
                ) as progress:
                    task = progress.add_task("Generating", total=None)
                    
                    # Generate reflection using local LLM
                    reflection = await self.llm_client.generate_reflection(
                        memory_dicts,
                        depth=chat_context["depth"],
                        creativity=chat_context["creativity"]
                    )
                
                if reflection and reflection.get("status") != "error":
                    # Display reflection
                    await self._display_reflection(reflection)
                    
                    # Save reflection to file
                    reflection_path = os.path.join(
                        self.session_dir, 
                        chat_context["session_id"], 
                        f"reflection_{int(time.time())}.json"
                    )
                    with open(reflection_path, "w") as f:
                        json.dump(reflection, f, indent=2)
                    
                    console.print(f"[{self.theme['success']}]Reflection saved to {reflection_path}[/{self.theme['success']}]")
                else:
                    error_msg = reflection.get('message', 'Unknown error') if reflection else 'No response from LLM'
                    console.print(f"[{self.theme['error']}]Error generating reflection: {error_msg}[/{self.theme['error']}]")
                    
            else:
                # Try Dream API
                console.print(f"[{self.theme['warning']}]Dream API reflection not implemented yet[/{self.theme['warning']}]")
                console.print(f"[{self.theme['warning']}]Please use local LLM for reflection (--local flag)[/{self.theme['warning']}]")
                
        except Exception as e:
            logger.error(f"Error generating reflection: {e}")
            console.print(f"[{self.theme['error']}]Error generating reflection: {e}[/{self.theme['error']}]")
            
    async def _display_reflection(self, reflection: Dict[str, Any]):
        """Display a reflection in a formatted way.
        
        Args:
            reflection: The reflection data
        """
        session_id = reflection.get("session_id", "unknown")
        fragments = reflection.get("fragments", [])
        
        # Log the full reflection for Docker logs and debugging
        logger.info(f"\nREFLECTION: {reflection.get('title', 'Untitled')}\n" +
                    f"Session ID: {session_id}\n" +
                    f"Metadata: {json.dumps(reflection.get('metadata', {}), indent=2)}\n" +
                    f"Fragments: {len(fragments)}")
        
        for i, fragment in enumerate(fragments):
            fragment_type = fragment.get("type", "unknown").capitalize()
            content = fragment.get("content", "")
            confidence = fragment.get("confidence", 0)
            logger.info(f"Fragment {i+1} ({fragment_type}): {content} (confidence: {confidence:.2f})")
        
        # Handle either new format (all fragments in one list) or old format (separate lists by type)
        insights = reflection.get("insights", [])
        if not isinstance(insights, list):
            insights = []
            
        questions = reflection.get("questions", [])
        if not isinstance(questions, list):
            questions = []
            
        hypotheses = reflection.get("hypotheses", [])
        if not isinstance(hypotheses, list):
            hypotheses = []
            
        counterfactuals = reflection.get("counterfactuals", [])
        if not isinstance(counterfactuals, list):
            counterfactuals = []
            
        # If we have fragments but not the individual lists, distribute them by type
        if fragments and not (insights or questions or hypotheses or counterfactuals):
            for fragment in fragments:
                if not isinstance(fragment, dict):
                    continue
                    
                fragment_type = fragment.get("type", "").lower()
                if fragment_type == "insight":
                    insights.append(fragment)
                elif fragment_type == "question":
                    questions.append(fragment)
                elif fragment_type == "hypothesis":
                    hypotheses.append(fragment)
                elif fragment_type == "counterfactual":
                    counterfactuals.append(fragment)
        
        # Display the successful reflection
        console.print(f"\n[{self.theme['success']} bold]Reflection Session: {session_id}[/{self.theme['success']} bold]")

        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Create header
        metadata = reflection.get("metadata", {})
        header_text = Text(f"Title: {reflection.get('title', 'Untitled Reflection')}\n", style=f"bold {self.theme['primary']}")
        header_text.append(f"Depth: {metadata.get('depth', 'N/A')} | ")
        header_text.append(f"Creativity: {metadata.get('creativity', 'N/A')} | ")
        header_text.append(f"Memory Count: {metadata.get('memory_count', 'N/A')}")
        layout["header"].update(Panel(header_text, title="Reflection", border_style=self.theme["primary"]))
        
        # Group fragments by type
        insights_text = "\n\n".join([f"[{i+1}] {insight.get('content', '')} ({insight.get('confidence', 0):.2f})" 
                               for i, insight in enumerate(insights)])
        insights_panel = Panel(insights_text or "No insights generated", 
                                title=f"Insights ({len(insights)})",
                                border_style=self.theme["secondary"])
        
        questions_text = "\n\n".join([f"[{i+1}] {question.get('content', '')} ({question.get('confidence', 0):.2f})" 
                                 for i, question in enumerate(questions)])
        questions_panel = Panel(questions_text or "No questions generated", 
                                title=f"Questions ({len(questions)})",
                                border_style=self.theme["accent"])
        
        hypotheses_text = "\n\n".join([f"[{i+1}] {hypothesis.get('content', '')} ({hypothesis.get('confidence', 0):.2f})" 
                                  for i, hypothesis in enumerate(hypotheses)])
        hypotheses_panel = Panel(hypotheses_text or "No hypotheses generated", 
                                title=f"Hypotheses ({len(hypotheses)})",
                                border_style=self.theme["secondary"])
        
        counterfactuals_text = "\n\n".join([f"[{i+1}] {cf.get('content', '')} ({cf.get('confidence', 0):.2f})" 
                                       for i, cf in enumerate(counterfactuals)])
        counterfactuals_panel = Panel(counterfactuals_text or "No counterfactuals generated", 
                                    title=f"Counterfactuals ({len(counterfactuals)})",
                                    border_style=self.theme["accent"])
        
        # Create body with split layout
        body_layout = Layout()
        body_layout.split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        # Left side: Insights and Questions
        left_layout = Layout()
        left_layout.split_column(
            Layout(name="insights"),
            Layout(name="questions")
        )
        
        left_layout["insights"].update(insights_panel)
        left_layout["questions"].update(questions_panel)
        
        body_layout["left"].update(left_layout)
        
        # Right side: Hypotheses and Counterfactuals
        right_layout = Layout()
        right_layout.split_column(
            Layout(name="hypotheses"),
            Layout(name="counterfactuals")
        )
        
        right_layout["hypotheses"].update(hypotheses_panel)
        right_layout["counterfactuals"].update(counterfactuals_panel)
        
        body_layout["right"].update(right_layout)
        layout["body"].update(body_layout)
        
        # Create footer
        footer_text = Text(f"Session ID: {session_id}\n")
        footer_text.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        layout["footer"].update(Panel(footer_text, title="Session Info", border_style=self.theme["info"]))
        
        # Render layout with safe width
        console_width = console.width if console.width else 100
        safe_width = min(console_width - 5, 120)  # Ensure we don't exceed console width
        
        try:
            # Print to console
            console.print(layout, width=safe_width)
        except Exception as e:
            logger.error(f"Error rendering reflection layout: {e}")
            # Fallback to simple text display
            console.print(f"[{self.theme['success']} bold]Reflection Title: {reflection.get('title', 'Untitled')}[/{self.theme['success']} bold]")
            
            for fragment_type, fragments_list in {
                "Insights": insights,
                "Questions": questions,
                "Hypotheses": hypotheses,
                "Counterfactuals": counterfactuals
            }.items():
                if fragments_list:
                    console.print(f"\n[bold]{fragment_type} ({len(fragments_list)}):[/bold]")
                    for i, f in enumerate(fragments_list):
                        console.print(f"  [{i+1}] {f.get('content', '')}")
    
    async def _save_chat_session(self, chat_context: Dict[str, Any]) -> str:
        """Save the current chat session to a file.
        
        Args:
            chat_context: The current chat context
            
        Returns:
            The path to the saved file
        """
        try:
            # Save chat data to file
            chat_path = os.path.join(self.session_dir, f"{chat_context['session_id']}.json")
            
            # Prepare data for saving
            save_data = {
                "session_id": chat_context["session_id"],
                "messages": chat_context["messages"],
                "memories": chat_context["memories"],
                "metadata": {
                    "depth": chat_context["depth"],
                    "creativity": chat_context["creativity"],
                    "created_at": datetime.now().isoformat(),
                    "memory_count": len(chat_context["memories"])
                }
            }
            
            with open(chat_path, "w") as f:
                json.dump(save_data, f, indent=2)
            
            return chat_path
            
        except Exception as e:
            logger.error(f"Error saving chat session: {e}")
            console.print(f"[{self.theme['error']}]Error saving chat session: {e}[/{self.theme['error']}]")
            return ""
            
    async def _process_chat_message(self, message: str, chat_context: Dict[str, Any], use_local: bool):
        """
        Process a regular chat message.
        
        Args:
            message: The user message
            chat_context: The current chat context
            use_local: Whether to use local LLM
        """
        messages = chat_context.get("messages", [])
        
        try:
            if not use_local and self.dream_api_url:
                # Using Dream API with self/world model integration
                try:
                    console.print(f"[{self.theme['info']}](Using Dream API with self/world model integration)[/{self.theme['info']}]")
                    
                    # Prepare data for the Dream API
                    chat_data = {
                        "messages": messages,
                        "new_message": message,
                        "parameters": {
                            "use_self_model": True,
                            "depth": 0.7,
                            "creativity": 0.5,
                            "include_memories": True
                        },
                        "context": {}
                    }
                    
                    # Show loading indicator
                    with Progress(
                        SpinnerColumn(),
                        TextColumn(f"[{self.theme['primary']}]Processing response...[/{self.theme['primary']}]"),
                        console=console
                    ) as progress:
                        task = progress.add_task("Generating", total=None)
                        
                        # Call the Dream API
                        url = f"{self.dream_api_url}/api/chat"
                        async with aiohttp.ClientSession() as session:
                            async with session.post(url, json=chat_data) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    
                                    # Extract the response
                                    content = result.get("response", "")
                                    if not content:
                                        raise Exception("Empty response from Dream API")
                                    
                                    # Log insights from the integration
                                    insights = result.get("insights", [])
                                    if insights:
                                        console.print(f"[{self.theme['info']} italic]Insights from self-model integration:[/{self.theme['info']} italic]")
                                        for insight in insights[:3]:  # Show up to 3 insights
                                            console.print(f"[{self.theme['info']} italic]- {insight}[/{self.theme['info']} italic]")
                                    
                                    # Add current spiral phase if available
                                    spiral_phase = result.get("spiral_phase")
                                    if spiral_phase and spiral_phase != "unknown":
                                        console.print(f"[{self.theme['info']} italic]Current spiral phase: {spiral_phase}[/{self.theme['info']} italic]")
                                    
                                    # Update chat context with new message pair
                                    messages.append({"role": "user", "content": message})
                                    messages.append({"role": "assistant", "content": content})
                                    
                                    # Return the response
                                    return content
                                else:
                                    error_text = await response.text()
                                    console.print(f"[{self.theme['error']}]Error from Dream API: {response.status} - {error_text}[/{self.theme['error']}]")
                                    raise Exception(f"Error from Dream API: {response.status}")
                except Exception as e:
                    console.print(f"[{self.theme['warning']}]Warning: Failed to use Dream API: {e}. Falling back to local LLM.[/{self.theme['warning']}]")
                    use_local = True
            
            # If we're using local LLM or Dream API failed
            if use_local or not self.dream_api_url:
                console.print(f"[{self.theme['info']}](Using local LLM)[/{self.theme['info']}]")
                
                # Display a message if we're using local but dream API is available
                if self.dream_api_url and use_local:
                    console.print(f"[{self.theme['info']} italic]Note: Using local LLM without self-model integration. Some features may be limited.[/{self.theme['info']} italic]")
                
                # Prepare system prompt for local LLM
                system_prompt = """You are Lucidia, an AI assistant focused on helping with memory, reflection, and self-improvement.
                You are thoughtful, empathetic, and always try to provide helpful and balanced perspectives.
                Focus on providing value in your answers."""
                
                # Format for the chat model
                formatted_messages = [
                    {"role": "system", "content": system_prompt}
                ]
                
                # Add conversation history
                for msg in messages[-5:]:  # Only include the last 5 messages to save context
                    formatted_messages.append(msg)
                
                # Add current message
                formatted_messages.append({"role": "user", "content": message})
                
                # Show loading indicator
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"[{self.theme['primary']}]Generating response...[/{self.theme['primary']}]"),
                    console=console
                ) as progress:
                    task = progress.add_task("Generating", total=None)
                    
                    # Generate response with local LLM
                    response = await self._call_local_llm(
                        messages=formatted_messages,
                        temperature=0.7,
                        max_tokens=1000
                    )
                
                # Extract content
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content:
                    content = "I'm sorry, I couldn't generate a response. Please try again."
                
                # Update chat context with new message pair
                messages.append({"role": "user", "content": message})
                messages.append({"role": "assistant", "content": content})
                
                return content
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            console.print(f"[{self.theme['error']}]Error processing message: {e}[/{self.theme['error']}]")
            return "I'm sorry, there was an error processing your message. Please try again."
    
    async def _call_local_llm(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """Call the local LLM (LM Studio) for chat completion.
        
        Args:
            messages: List of message objects with role and content
            temperature: Creativity temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            The model's response as a dictionary
        """
        try:
            lm_studio_url = self.config.get("lm_studio_url", "http://localhost:1234")
            
            # Create a new session if needed
            if not hasattr(self, 'session') or self.session is None:
                self.session = aiohttp.ClientSession()
            
            # Call LM Studio API
            async with self.session.post(
                f"{lm_studio_url}/v1/chat/completions", 
                json={
                    "model": "local-model",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=60  # Local models might need more time
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Error from LM Studio: {error_text}")
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "I'm having trouble connecting to the local LLM. Please check if LM Studio is running."
                                }
                            }
                        ]
                    }
        except Exception as e:
            logger.error(f"Error calling local LLM: {e}")
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"Error calling local LLM: {e}"
                        }
                    }
                ]
            }
    
async def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Lucidia Reflection CLI")
    parser.add_argument("--config", "-c", type=str, help="Path to config file")
    
    # Commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Connect command
    connect_parser = subparsers.add_parser("connect", help="Connect to services")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor system state")
    monitor_parser.add_argument("--continuous", "-c", action="store_true", help="Continuously monitor")
    monitor_parser.add_argument("--interval", "-i", type=int, default=5, help="Update interval in seconds")
    
    # Dream command
    dream_parser = subparsers.add_parser("dream", help="Start a dream session")
    dream_parser.add_argument("--duration", "-d", type=int, help="Dream duration in seconds")
    dream_parser.add_argument("--depth", type=float, help="Reflection depth (0.0-1.0)")
    dream_parser.add_argument("--creativity", type=float, help="Creative exploration (0.0-1.0)")
    dream_parser.add_argument("--memories", "-m", type=str, nargs="+", help="Memory IDs to include")
    dream_parser.add_argument("--local", "-l", action="store_true", help="Use local LLM for reflection")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session with Lucidia")
    chat_parser.add_argument("--local", "-l", action="store_true", help="Use local LLM for chat")
    chat_parser.add_argument("--depth", type=float, default=0.7, help="Reflection depth (0.0-1.0)")
    chat_parser.add_argument("--creativity", type=float, default=0.5, help="Creative exploration (0.0-1.0)")
    
    # Parameters command
    params_parser = subparsers.add_parser("params", help="Get or set parameters")
    params_parser.add_argument("--path", "-p", type=str, help="Parameter path")
    params_parser.add_argument("--value", "-v", type=str, help="New parameter value")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize system state")
    viz_parser.add_argument("--type", "-t", type=str, choices=["kg", "metrics", "dashboard"], 
                           default="dashboard", help="Visualization type")
    viz_parser.add_argument("--output", "-o", type=str, help="Output file path")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export data")
    export_parser.add_argument("--type", "-t", type=str, choices=["metrics", "memories", "kg"], 
                            default="metrics", help="Export type")
    export_parser.add_argument("--output", "-o", type=str, help="Output file path")
    
    # Model info command for local LLM
    models_parser = subparsers.add_parser("models", help="List available local LLM models")
    
    # Theme command - NEW
    theme_parser = subparsers.add_parser("theme", help="Change visual theme")
    theme_parser.add_argument("--primary", type=str, help="Primary color")
    theme_parser.add_argument("--secondary", type=str, help="Secondary color")
    theme_parser.add_argument("--accent", type=str, help="Accent color")
    theme_parser.add_argument("--reset", action="store_true", help="Reset to default theme")
    
    args = parser.parse_args()
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    config_path = args.config or "lucidia_config.json"
    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
    
    # Store config_path in the config dictionary so it's accessible in other methods
    config["config_path"] = config_path
    
    # Handle local LLM flag
    if hasattr(args, "local") and args.local:
        config["use_local_llm"] = True
    
    # Display app header
    console.print(Panel(
        "[bold]LUCIDIA REFLECTION CLI[/bold]\n"
        "A tool for monitoring Lucidia's self-awareness and memory evolution",
        style="blue"
    ))
    
    # Create client
    client = LucidiaReflectionClient(config)
    
    try:
        # Handle commands
        if args.command == "connect":
            # Just connect and show status
            success = await client.connect()
            console.print(Panel("[green]Connected[/green]" if success else "[red]Failed to connect[/red]", 
                              title="Connection Status"))
        
        elif args.command == "monitor":
            # Connect first
            if not await client.connect():
                console.print(f"[{client.theme['error']}]Failed to connect to services[/{client.theme['error']}]")
                return 1
            
            # Monitor system state
            try:
                await client.continuous_monitoring(interval=args.interval, duration=3600 if args.continuous else 300)
            finally:
                await client.disconnect()
        
        elif args.command == "dream":
            # Connect first
            if not await client.connect():
                console.print(f"[{client.theme['error']}]Failed to connect to services[/{client.theme['error']}]")
                return 1
            
            try:
                # Start dream session
                session_id = await client.start_dream_session(
                    duration=args.duration,
                    depth=args.depth,
                    creativity=args.creativity,
                    with_memories=args.memories
                )
                
                if not session_id:
                    console.print(f"[{client.theme['error']}]Failed to start or complete dream session[/{client.theme['error']}]")
            finally:
                await client.disconnect()
        
        elif args.command == "chat":
            # Connect first
            if not await client.connect():
                console.print(f"[{client.theme['error']}]Failed to connect to services[/{client.theme['error']}]")
                return 1
            
            try:
                # Start chat session
                await client.start_chat_session(
                    depth=args.depth,
                    creativity=args.creativity,
                    local=args.local
                )
            finally:
                await client.disconnect()
        
        elif args.command == "params":
            # Connect first
            if not await client.connect():
                console.print(f"[{client.theme['error']}]Failed to connect to services[/{client.theme['error']}]")
                return 1
            
            try:
                if args.path and args.value:
                    # Update parameter
                    success = await client.update_parameter(args.path, json.loads(args.value))
                    console.print(f"[{client.theme['success']}]Parameter updated[/{client.theme['success']}]" if success else 
                                    f"[{client.theme['error']}]Failed to update parameter[/{client.theme['error']}]")
                
                elif args.path:
                    # Get specific parameter
                    param = await client.get_parameter(args.path)
                    if param is not None:
                        console.print(json.dumps(param, indent=2))
                    else:
                        console.print(f"[{client.theme['error']}]Parameter {args.path} not found[/{client.theme['error']}]")
                
                else:
                    # Display all parameters
                    await client.interactive_parameter_edit()
            finally:
                await client.disconnect()
        
        elif args.command == "visualize":
            # Connect first
            if not await client.connect():
                console.print(f"[{client.theme['error']}]Failed to connect to services[/{client.theme['error']}]")
                return 1
            
            try:
                if args.type == "kg":
                    await client.visualize_knowledge_graph(output_file=args.output)
                elif args.type == "metrics":
                    await client.visualize_metrics(output_file=args.output)
                else:
                    await client.display_evolution_dashboard()
            finally:
                await client.disconnect()
        
        elif args.command == "export":
            # Connect first
            if not await client.connect():
                console.print(f"[{client.theme['error']}]Failed to connect to services[/{client.theme['error']}]")
                return 1
            
            try:
                output_path = args.output or f"{client.session_id}_{args.type}.json"
                if args.type == "metrics":
                    await client.export_metrics(output_path)
                elif args.type == "memories":
                    await client.export_memories(output_path)
                elif args.type == "kg":
                    await client.export_knowledge_graph(output_path)
            finally:
                await client.disconnect()
                
        elif args.command == "models":
            # Connect first
            if not await client.connect():
                console.print(f"[{client.theme['error']}]Failed to connect to services[/{client.theme['error']}]")
                return 1
            
            try:
                models = await client.get_local_llm_models()
                if models:
                    # Create table
                    table = Table(title="Available LLM Models", box=box.ROUNDED)
                    table.add_column("ID", style=client.theme["primary"])
                    table.add_column("Object", style=client.theme["secondary"])
                    table.add_column("Created", style=client.theme["info"])
                    
                    for model in models:
                        table.add_row(
                            model.get("id", "Unknown"),
                            model.get("object", "Unknown"),
                            datetime.fromtimestamp(model.get("created", 0)).strftime("%Y-%m-%d")
                        )
                    
                    console.print(table)
                else:
                    console.print(f"[{client.theme['warning']}]No models available or Local LLM not enabled[/{client.theme['warning']}]")
            finally:
                await client.disconnect()
        
        elif args.command == "theme":
            # Update theme and save to config
            if args.reset:
                config["theme"] = DEFAULT_CONFIG["theme"].copy()
                console.print("[green]Theme reset to default[/green]")
            else:
                # Update specified colors
                if args.primary:
                    config["theme"]["primary"] = args.primary
                if args.secondary:
                    config["theme"]["secondary"] = args.secondary
                if args.accent:
                    config["theme"]["accent"] = args.accent
                
                console.print("[green]Theme updated[/green]")
            
            # Save updated config
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            # Show current theme
            table = Table(title="Current Theme", box=box.ROUNDED)
            table.add_column("Color Type")
            table.add_column("Value")
            table.add_column("Sample")
            
            for color_name, color_value in config["theme"].items():
                table.add_row(
                    color_name.capitalize(),
                    color_value,
                    f"[{color_value}]Sample Text[/{color_value}]"
                )
            
            console.print(table)
                
        else:
            # Show help if no command specified
            parser.print_help()
    
    except KeyboardInterrupt:
        console.print(f"[{client.theme['warning']}]Interrupted by user[/{client.theme['warning']}]")
        return 1
    except Exception as e:
        logger.exception("Error in main")
        console.print(f"[red]Error: {e}[/red]")
        return 1
    finally:
        # Always ensure client is disconnected
        await client.disconnect()
        console.print("Disconnected from services")
    
    return 0

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")