#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variant Diagnostics Dashboard - For monitoring Titans variant performance

This dashboard tool connects to the Context Cascade Orchestrator and
visualizes performance metrics for different Titans variants, facilitating
tuning and selection of optimal variants for different contexts.
"""

import os
import sys
import json
import argparse
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from aiohttp import ClientSession

# Rich library for better terminal display
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VariantDiagnostics')

# Console for rich output
console = Console()

# Set to True to enable debug mode (additional logging, etc.)
DEBUG = os.environ.get('VARIANT_DIAGNOSTICS_DEBUG', 'false').lower() == 'true'

class VariantDiagnosticsDashboard:
    """
    Dashboard for monitoring the performance of various Titans variants
    in the Synthians Cognitive Architecture.
    """
    
    def __init__(self, orchestrator_url: str = None, refresh_rate: int = 5):
        """
        Initialize the diagnostics dashboard.
        
        Args:
            orchestrator_url: URL of the Context Cascade Orchestrator API
            refresh_rate: How often to refresh metrics (in seconds)
        """
        self.orchestrator_url = orchestrator_url or os.environ.get('CCE_URL', 'http://localhost:8002')
        self.refresh_rate = refresh_rate
        self.metrics_history = []
        self.max_history = 100  # Keep up to 100 historical metrics snapshots
        self.is_running = False
        
        logger.info(f"Initializing Variant Diagnostics Dashboard")
        logger.info(f"Orchestrator URL: {self.orchestrator_url}")
        logger.info(f"Refresh Rate: {self.refresh_rate} seconds")
    
    def parse_cce_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a CCE response to extract relevant variant selection information,
        performance metrics, and adaptive parameters.
        
        Args:
            data: Raw CCE response data
            
        Returns:
            Parsed structured data for display
        """
        # Initialize parsed data structure with defaults
        parsed_data = {
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "status": data.get("status", "UNKNOWN"),
            "memory_id": data.get("memory_id", "N/A"),
            "active_variant": data.get("variant_output", {}).get("variant_type", "UNKNOWN"),
            "variant_metrics": {},  # Populated below
            "adaptive_params": {},  # Populated below
            "selector_info": data.get("variant_selection", {}),
            "llm_info": data.get("llm_advice_used", {}),
            "nm_update": data.get("neural_memory_update", {}),
            "qr_feedback": data.get("quickrecal_feedback", {})
        }
        
        # Extract variant-specific metrics and adaptive parameters
        vo = data.get("variant_output", {})
        vt_lower = parsed_data["active_variant"].lower()
        
        if vt_lower != "none" and vt_lower in vo:
            metrics_dict = vo[vt_lower]
            parsed_data["variant_metrics"] = metrics_dict
            
            # Extract adaptive parameters based on variant
            parsed_data["adaptive_params"] = {
                "focus_mode": metrics_dict.get("attention_focus", metrics_dict.get("attention_mode", "N/A")),
                "context_limit": metrics_dict.get("context_limit"),
                "temperature": metrics_dict.get("attention_temperature"),
                "blend_factor": metrics_dict.get("blend_factor"),  # MAL
                "gate_modifiers": metrics_dict.get("calculated_gates", metrics_dict.get("gate_modifiers")),  # MAG - check both possible keys
                "recency_bias": metrics_dict.get("recency_bias_applied"),  # MAC - check boolean flag
            }
            # Filter out None values
            parsed_data["adaptive_params"] = {k: v for k, v in parsed_data["adaptive_params"].items() if v is not None}
        
        # Performance metrics processing
        if "perf_metrics_used" in parsed_data["selector_info"]:
            perf = parsed_data["selector_info"]["perf_metrics_used"]
            # Format float values to 4 decimal places for readability
            for key in perf:
                if isinstance(perf[key], float):
                    perf[key] = round(perf[key], 4)
            
        # For debugging, log the full parsed data if in debug mode
        if DEBUG:
            logger.debug(f"Parsed CCE response: {json.dumps(parsed_data, indent=2, default=str)}")
            
        return parsed_data
    
    async def fetch_metrics(self, session: ClientSession, limit: int = 20) -> Dict[str, Any]:
        """
        Fetch recent metrics from the orchestrator.
        
        Args:
            session: aiohttp ClientSession for making requests
            limit: Maximum number of recent responses to retrieve
            
        Returns:
            Dictionary containing metrics data
        """
        try:
            endpoint = f"{self.orchestrator_url}/get_recent_metrics"
            async with session.post(endpoint, json={"limit": limit}) as response:
                if response.status == 200:
                    data = await response.json()
                    if DEBUG:
                        logger.debug(f"Received metrics: {json.dumps(data, indent=2)}")
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"Error fetching metrics: {response.status} - {error_text}")
                    return {"error": f"HTTP Error {response.status}: {error_text}"}
        except Exception as e:
            logger.error(f"Exception fetching metrics: {e}")
            return {"error": str(e)}
    
    def display_metrics(self, metrics: Dict[str, Any]):
        """
        Display metrics in a formatted way using rich library.
        
        Args:
            metrics: Dictionary containing metrics data
        """
        if "error" in metrics:
            console.print(f"[bold red]Error displaying metrics: {metrics.get('error', 'Unknown error')}[/bold red]")
            return

        console.clear()
        console.print(f"[bold cyan]📊 SYNTHIANS DIAGNOSTICS ({datetime.now().isoformat()}) 📊[/bold cyan]")
        console.print(f"{'-' * console.width}")

        # Get recent responses for detailed analysis
        recent_responses = metrics.get("recent_responses", [])
        if not recent_responses:
            console.print("[yellow]No recent responses available[/yellow]")
            return

        # Process the most recent response (typically what we want to display in detail)
        latest_response = recent_responses[0] if recent_responses else {}
        parsed_data = self.parse_cce_response(latest_response)

        # --- Main Info Panel ---
        status_style = "green" if parsed_data['status'] == 'completed' else "red"
        variant_style = "bold green" # Or style based on variant
        main_panel = Panel(
            f"Timestamp: {parsed_data['timestamp']}\n"
            f"Active Variant: [{variant_style}]{parsed_data['active_variant']}[/]\n"
            f"Status: [{status_style}]{parsed_data['status']}[/]\n"
            f"Memory ID: {parsed_data['memory_id']}",
            title="[b]System Status[/b]", border_style="blue", expand=False
        )

        # --- Performance Panel ---
        perf_table = Table(show_header=False, box=None, padding=(0,1), show_edge=False)
        perf_table.add_column(style="dim")
        perf_table.add_column(justify="right")
        loss = parsed_data.get('nm_update', {}).get('loss')
        grad = parsed_data.get('nm_update', {}).get('grad_norm')
        boost = parsed_data.get('qr_feedback', {}).get('boost_applied')
        perf_table.add_row("NM Loss:", f"{loss:.5f}" if isinstance(loss, float) else "[dim]N/A[/dim]")
        perf_table.add_row("NM Grad Norm:", f"{grad:.5f}" if isinstance(grad, float) else "[dim]N/A[/dim]")
        perf_table.add_row("QR Boost Applied:", f"{boost:.5f}" if isinstance(boost, float) else "[dim]N/A[/dim]")
        perf_panel = Panel(perf_table, title="[b]Performance[/b]", border_style="green", expand=False)

        # --- Selection Panel ---
        sel_info = parsed_data.get('selector_info', {})
        sel_table = Table(show_header=False, box=None, padding=(0,1), show_edge=False)
        sel_table.add_column(style="dim")
        sel_table.add_column(justify="left")
        sel_table.add_row("Selected:", f"[magenta]{sel_info.get('selected', 'N/A')}[/magenta] (Current: {sel_info.get('current', 'N/A')})")
        sel_table.add_row("Reason:", Text(sel_info.get('reason', 'N/A'), overflow="fold"))
        if 'perf_metrics_used' in sel_info:
             perf = sel_info['perf_metrics_used']
             avg_loss = f"{perf.get('avg_loss'):.3f}" if isinstance(perf.get('avg_loss'), (int, float)) else perf.get('avg_loss', 'N/A')
             avg_grad = f"{perf.get('avg_grad_norm'):.3f}" if isinstance(perf.get('avg_grad_norm'), (int, float)) else perf.get('avg_grad_norm', 'N/A')
             std_dev = f"{perf.get('std_dev_loss'):.3f}" if isinstance(perf.get('std_dev_loss'), (int, float)) else perf.get('std_dev_loss', 'N/A')
             perf_text = (f"Loss:{avg_loss} "
                          f"Grad:{avg_grad} "
                          f"StdD:{std_dev} "
                          f"Trend:{perf.get('trend_status', 'N/A')} "
                          f"Conf:{perf.get('confidence_level', 'N/A')} "
                          f"N:{perf.get('sample_count', 'N/A')}")
             sel_table.add_row("Perf Used:", perf_text)
        selection_panel = Panel(sel_table, title="[b]Variant Selection[/b]", border_style="magenta", expand=False)

        # --- LLM Guidance Panel ---
        llm_info = parsed_data.get('llm_info', {})
        llm_panel = Panel("[dim]No LLM Guidance Used[/dim]", title="[b]LLM Guidance[/b]", border_style="yellow", expand=False)
        if llm_info:
            llm_table = Table(show_header=False, box=None, padding=(0,1), show_edge=False)
            llm_table.add_column(style="dim")
            llm_table.add_column(justify="right")
            llm_table.add_row("Variant Hint:", f"Provided: {llm_info.get('variant_hint_provided', 'N/A')} -> Final: {llm_info.get('variant_hint_final', 'N/A')}")
            boost_mod_orig = llm_info.get('original_boost_mod')
            boost_mod_applied = llm_info.get('boost_modifier_applied')
            boost_orig_fmt = f"{boost_mod_orig:.3f}" if isinstance(boost_mod_orig, (int, float)) else 'N/A'
            boost_applied_fmt = f"{boost_mod_applied:.3f}" if isinstance(boost_mod_applied, (int, float)) else 'N/A'
            llm_table.add_row("Boost Mod:", f"Provided: {boost_orig_fmt} -> Applied: {boost_applied_fmt}")
            llm_table.add_row("Conf. Adjust:", f"{llm_info.get('confidence_level','N/A')} ({llm_info.get('adjustment_reason','N/A')})")
            llm_table.add_row("Focus Used:", str(llm_info.get('attention_focus_used', 'N/A')))
            llm_table.add_row("Tags Added:", str(llm_info.get('tags_added', [])))
            llm_panel = Panel(llm_table, title="[b]LLM Guidance Used[/b]", border_style="yellow", expand=False)

        # --- Adaptive Attention Panel ---
        adapt_params = parsed_data.get('adaptive_params', {})
        adapt_panel = Panel("[dim]N/A (Variant: NONE or No Metrics)[/dim]", title="[b]Adaptive Attention[/b]", border_style="cyan", expand=False)
        if adapt_params:
            adapt_table = Table(show_header=False, box=None, padding=(0,1), show_edge=False)
            adapt_table.add_column(style="dim")
            adapt_table.add_column(justify="right")
            adapt_table.add_row("Focus Mode:", str(adapt_params.get('focus_mode', 'N/A')))
            if adapt_params.get('context_limit') is not None:
                 adapt_table.add_row("Context Limit:", str(adapt_params['context_limit']))
            if adapt_params.get('temperature') is not None:
                 temp = adapt_params['temperature']
                 temp_fmt = f"{temp:.2f}" if isinstance(temp, (int, float)) else str(temp)
                 adapt_table.add_row("Temperature:", temp_fmt)
            if adapt_params.get('blend_factor') is not None: # MAL
                 blend = adapt_params['blend_factor']
                 blend_fmt = f"{blend:.2f}" if isinstance(blend, (int, float)) else str(blend)
                 adapt_table.add_row("Blend Factor:", blend_fmt)
            if adapt_params.get('gate_modifiers') is not None: # MAG
                 adapt_table.add_row("Gate Modifiers:", str(adapt_params['gate_modifiers']))
            if adapt_params.get('recency_bias') is not None: # MAC
                 adapt_table.add_row("Recency Bias:", str(adapt_params['recency_bias']))
            adapt_panel = Panel(adapt_table, title="[b]Adaptive Attention Params[/b]", border_style="cyan", expand=False)

        # --- Arrange Panels ---
        console.print(main_panel)
        console.print(Columns([perf_panel, selection_panel])) # Side-by-side
        console.print(llm_panel)
        console.print(adapt_panel)
        
        # Display variant statistics summary (if available)
        variant_stats = metrics.get("variant_stats", {})
        if variant_stats:
            counts = variant_stats.get("counts", {})
            total = variant_stats.get("total_responses", 0)
            surprise_metrics = variant_stats.get("surprise_metrics", {})
            
            stats_table = Table(title="[b]Variant Usage Statistics[/b]")
            stats_table.add_column("Variant", style="cyan")
            stats_table.add_column("Count", justify="right")
            stats_table.add_column("Percentage", justify="right")
            stats_table.add_column("Avg Loss", justify="right")
            stats_table.add_column("Avg Grad", justify="right")
            stats_table.add_column("Avg Boost", justify="right")
            
            for variant, count in counts.items():
                percentage = (count / total) * 100 if total > 0 else 0
                metrics_for_variant = surprise_metrics.get(variant, {})
                avg_loss = metrics_for_variant.get('avg_loss')
                avg_grad = metrics_for_variant.get('avg_grad_norm')
                avg_boost = metrics_for_variant.get('avg_boost')
                
                avg_loss_fmt = f"{avg_loss:.5f}" if isinstance(avg_loss, (int, float)) else "N/A"
                avg_grad_fmt = f"{avg_grad:.5f}" if isinstance(avg_grad, (int, float)) else "N/A"
                avg_boost_fmt = f"{avg_boost:.5f}" if isinstance(avg_boost, (int, float)) else "N/A"
                
                stats_table.add_row(
                    variant,
                    str(count),
                    f"{percentage:.1f}%",
                    avg_loss_fmt,
                    avg_grad_fmt,
                    avg_boost_fmt
                )
            
            console.print(stats_table)
        
        # Footer
        console.print(f"\n{'-' * console.width}")
        console.print(f"[dim]Press Ctrl+C to exit. Refreshing every {self.refresh_rate} seconds.[/dim]")
    
    async def run(self):
        """
        Run the dashboard, periodically fetching and displaying metrics.
        """
        self.is_running = True
        try:
            async with ClientSession() as session:
                while self.is_running:
                    # Fetch metrics
                    metrics = await self.fetch_metrics(session)
                    
                    # Store in history
                    if "error" not in metrics:
                        self.metrics_history.append(metrics)
                        if len(self.metrics_history) > self.max_history:
                            self.metrics_history.pop(0)
                    
                    # Display metrics
                    self.display_metrics(metrics)
                    
                    # Wait for refresh interval
                    await asyncio.sleep(self.refresh_rate)
        except asyncio.CancelledError:
            logger.info("Dashboard stopped via cancellation")
            self.is_running = False
        except Exception as e:
            logger.error(f"Error in dashboard run loop: {e}")
            self.is_running = False
    
    def stop(self):
        """
        Stop the dashboard.
        """
        self.is_running = False
        logger.info("Dashboard stopped")

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Dashboard for monitoring Titans variant performance in the Synthians Cognitive Architecture.'
    )
    parser.add_argument(
        '--url', '-u', type=str, default=None,
        help='URL of the Context Cascade Orchestrator API (default: http://localhost:8002 or CCE_URL env var)'
    )
    parser.add_argument(
        '--refresh', '-r', type=int, default=5,
        help='How often to refresh metrics in seconds (default: 5)'
    )
    parser.add_argument(
        '--debug', '-d', action='store_true',
        help='Enable debug mode with additional logging'
    )
    
    return parser.parse_args()

async def main_async():
    """
    Async entry point for the dashboard.
    """
    args = parse_arguments()
    
    # Set debug mode if requested
    global DEBUG
    if args.debug:
        DEBUG = True
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run the dashboard
    dashboard = VariantDiagnosticsDashboard(
        orchestrator_url=args.url,
        refresh_rate=args.refresh
    )
    
    try:
        await dashboard.run()
    except KeyboardInterrupt:
        dashboard.stop()
        print("\nDashboard stopped.")

def main():
    """
    Main entry point for the dashboard.
    """
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nDashboard stopped.")

if __name__ == '__main__':
    main()
