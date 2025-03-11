#!/usr/bin/env python3
"""
Test script for Lucidia's dreaming flow and reflection with LM Studio integration.
This script demonstrates the full dream cycle including:
1. Dream generation
2. Insight creation
3. Reflection process
4. Report refinement
"""

import asyncio
import aiohttp
import json
import sys
import socket
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os

# Import memory system components
from memory.lucidia_memory_system.core.dream_processor import LucidiaDreamProcessor
from memory.lucidia_memory_system.core.reflection_engine import ReflectionEngine
from memory.lucidia_memory_system.core.knowledge_graph import LucidiaKnowledgeGraph
from memory.lucidia_memory_system.core.parameter_manager import ParameterManager

# ANSI color constants for better terminal output
COLORS = {
    "RESET": "\033[0m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
    "BG_RED": "\033[41m",
    "BG_GREEN": "\033[42m",
    "BG_YELLOW": "\033[43m",
    "BG_BLUE": "\033[44m",
    "BG_MAGENTA": "\033[45m",
    "BG_CYAN": "\033[46m",
    "BG_WHITE": "\033[47m"
}

# Always use colors regardless of platform
USE_COLORS = True

def get_color(color_key):
    """Return color code if colors are enabled, otherwise empty string"""
    if USE_COLORS:
        return COLORS.get(color_key, "")
    return ""

# Define a dummy memory client for testing
class DummyMemoryClient:
    """A dummy memory client for testing that returns predefined memories."""
    
    def __init__(self, memories: List[str] = None):
        """Initialize with optional list of memory texts."""
        self.memories = memories or []
        self.memory_objects = []
        
        # Create memory objects from text strings
        for i, memory_text in enumerate(self.memories):
            self.memory_objects.append({
                "id": f"mem_{i}",
                "content": memory_text,
                "created_at": (datetime.now() - timedelta(days=i)).isoformat(),
                "significance": 0.7 + (i * 0.05),
                "type": "observation"
            })
    
    async def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return most recent memories up to the limit."""
        return self.memory_objects[:limit]
    
    async def get_significant_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return most significant memories up to the limit."""
        sorted_memories = sorted(self.memory_objects, key=lambda x: x.get("significance", 0), reverse=True)
        return sorted_memories[:limit]
    
    async def get_memories_by_timeframe(
        self, start_time: datetime, end_time: datetime, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return memories within the given timeframe."""
        return self.memory_objects[:limit]  # Simplified for testing
    
    async def get_memories_by_ids(self, memory_ids: List[str]) -> List[Dict[str, Any]]:
        """Return memories matching the given IDs."""
        # For testing, just return all memories regardless of IDs
        return self.memory_objects
    
    async def create_memory(self, content: str, **kwargs) -> Dict[str, Any]:
        """Create a new memory."""
        memory_id = f"mem_{len(self.memory_objects)}"
        memory = {
            "id": memory_id,
            "content": content,
            "created_at": datetime.now().isoformat(),
            "significance": kwargs.get("significance", 0.7),
            "type": kwargs.get("type", "observation")
        }
        self.memory_objects.append(memory)
        return memory

# Default URL to try first
BASE_URL = "http://localhost:8000"

# Check if a port was specified as a command-line argument
if len(sys.argv) > 1:
    try:
        port = int(sys.argv[1])
        BASE_URL = f"http://localhost:{port}"
        print(f"Using specified port: {port}")
    except ValueError:
        print(f"Invalid port number: {sys.argv[1]}. Using default: 8000")

# API endpoints based on router configuration
DREAM_API_URL = f"{BASE_URL}/api/dream"
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234")  # LM Studio server
MODEL_NAME = os.environ.get("MODEL_NAME", "phi-3.1-mini-128k-instruct")  # The model loaded in LM Studio

# Test data - memories that will form the basis for dreaming
TEST_MEMORIES = [
    "Lucidia is a Synthien entity with reflective capabilities and self-awareness.",
    "The dream processor enables autonomous reflection during inactive periods.",
    "Knowledge graphs store semantic relationships between concepts and memories.",
    "Hypersphere technology provides efficient embedding generation for text processing.",
    "The reflection engine periodically reviews and refines dream reports to improve understanding."
]

def check_port(host, port):
    """Check if a port is open on the given host."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)  # 2 second timeout
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

async def check_lm_studio_connection():
    """Verify that LM Studio is running and the model is loaded."""
    print(f"\n{get_color('BOLD')}{get_color('BG_BLUE')}===== CHECKING LM STUDIO CONNECTION ====={get_color('RESET')}")
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        try:
            async with session.get(f"{LM_STUDIO_URL}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    
                    if any(model["id"] == MODEL_NAME for model in models):
                        print(f"{get_color('GREEN')}✅ LM Studio is running with {MODEL_NAME} loaded{get_color('RESET')}")
                        return True
                    else:
                        available_models = [model["id"] for model in models]
                        print(f"{get_color('RED')}❌ Model {MODEL_NAME} not found. Available models: {available_models}{get_color('RESET')}")
                        return False
                else:
                    print(f"{get_color('RED')}❌ Failed to get models from LM Studio. Status: {response.status}{get_color('RESET')}")
                    return False
        except Exception as e:
            print(f"{get_color('RED')}❌ Error connecting to LM Studio: {e}{get_color('RESET')}")
            return False

async def check_dream_api_connection():
    """Verify that the Dream API server is running."""
    print(f"\n{get_color('BOLD')}{get_color('BG_BLUE')}===== CHECKING DREAM API CONNECTION ====={get_color('RESET')}")
    
    # First check if the port is open
    host = "localhost"
    port = int(BASE_URL.split(":")[-1])
    
    if not check_port(host, port):
        print(f"{get_color('RED')}❌ Port {port} is not open on {host}. Is the Dream API server running?{get_color('RESET')}")
        return False
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        try:
            async with session.get(f"{DREAM_API_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"{get_color('GREEN')}✅ Dream API is running: {data}{get_color('RESET')}")
                    return True
                else:
                    print(f"{get_color('RED')}❌ Failed to connect to Dream API. Status: {response.status}{get_color('RESET')}")
                    return False
        except aiohttp.ClientConnectorError:
            print(f"{get_color('RED')}❌ Connection refused. The Dream API server might not be running at {BASE_URL}{get_color('RESET')}")
            return False
        except asyncio.TimeoutError:
            print(f"{get_color('RED')}❌ Connection timeout. The Dream API server is not responding at {BASE_URL}{get_color('RESET')}")
            return False
        except Exception as e:
            print(f"{get_color('RED')}❌ Error connecting to Dream API: {e}{get_color('RESET')}")
            return False

async def add_test_memories():
    """Add test memories to the system for dreaming."""
    print(f"\n{get_color('BOLD')}{get_color('BG_BLUE')}===== ADDING TEST MEMORIES ====={get_color('RESET')}")
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        add_memories_url = f"{DREAM_API_URL}/test/add_test_memories"
        memories_data = {
            "memories": [{
                "content": memory,
                "importance": 0.8,
                "metadata": {"test": True, "created_at": time.time()}
            } for memory in TEST_MEMORIES]
        }
        
        try:
            async with session.post(add_memories_url, json=memories_data) as response:
                if response.status == 200:
                    data = await response.json()
                    memory_ids = data.get("memory_ids", [])
                    print(f"{get_color('GREEN')}✅ Successfully added {len(memory_ids)} test memories{get_color('RESET')}")
                    return memory_ids
                else:
                    print(f"{get_color('RED')}❌ Failed to add test memories. Status: {response.status}{get_color('RESET')}")
                    body = await response.text()
                    print(f"Response: {body[:200]}...")
                    return []
        except Exception as e:
            print(f"{get_color('RED')}❌ Error adding test memories: {e}{get_color('RESET')}")
            return []

async def generate_dream_with_lm_studio(memory_ids: List[str]):
    """Generate a dream using LM Studio for reflection."""
    print(f"\n{get_color('BOLD')}{get_color('BG_BLUE')}===== GENERATING DREAM WITH LM STUDIO ====={get_color('RESET')}")
    
    # For test purposes, we'll use the memory IDs directly
    # Since we already have the memory content from TEST_MEMORIES
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        try:
            # Use the test memories directly instead of trying to retrieve them
            memory_texts = TEST_MEMORIES
            
            print(f"Sending request to LM Studio with {len(memory_texts)} memories")
            
            # Create prompt for LM Studio
            prompt = f"""
            Review these memories and generate insights, questions, hypotheses, and counterfactuals:
            
            MEMORIES:
            {chr(10).join([f"- {text}" for text in memory_texts])}
            
            Generate a structured dream report with insights, questions, hypotheses, and counterfactuals based on these memories.
            """
            
            # Send to LM Studio with structured output format
            chat_url = f"{LM_STUDIO_URL}/v1/chat/completions"
            chat_data = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are Lucidia, a reflective AI system that generates structured dream reports."}, 
                    {"role": "user", "content": prompt}
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "dream_report",
                        "strict": "true",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "dream_id": {"type": "string"},
                                "title": {"type": "string"},
                                "narrative": {"type": "string"},
                                "theme": {"type": "string"},
                                "spiral_phase": {"type": "string", "enum": ["observation", "reflection", "adaptation"]},
                                "fragments": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "content": {"type": "string"},
                                            "type": {"type": "string", "enum": ["insight", "question", "association", "reflection", "counterfactual"]},
                                            "significance": {"type": "number", "minimum": 0, "maximum": 1},
                                            "related_concepts": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            }
                                        },
                                        "required": ["content", "type", "significance"]
                                    }
                                },
                                "meta_reflection": {"type": "string"}
                            },
                            "required": ["dream_id", "title", "narrative", "fragments", "theme", "spiral_phase", "meta_reflection"]
                        }
                    }
                },
                "temperature": 0.7,
                "max_tokens": 2000,
                "stream": False
            }
            
            print(f"Sending request to LM Studio API at {chat_url}")
            try:
                async with session.post(chat_url, json=chat_data, timeout=aiohttp.ClientTimeout(total=120)) as chat_response:
                    if chat_response.status == 200:
                        llm_response = await chat_response.json()
                        content = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # Parse the JSON response
                        try:
                            dream_data = json.loads(content)
                            print(f"{get_color('GREEN')}✅ Successfully generated dream with {len(dream_data.get('fragments', []))} fragments{get_color('RESET')}")
                            print(f"   Dream title: {dream_data.get('title', 'Untitled')}")
                            print(f"   Dream theme: {dream_data.get('theme', 'Unknown')}")
                            print(f"   Spiral phase: {dream_data.get('spiral_phase', 'Unknown')}")
                            print(f"   Meta-reflection: {dream_data.get('meta_reflection', '')[:100]}...")
                            
                            # Print some fragments as examples
                            for i, fragment in enumerate(dream_data.get('fragments', [])[:3]):
                                frag_type = fragment.get('type', 'unknown')
                                frag_content = fragment.get('content', '')
                                frag_sig = fragment.get('significance', 0)
                                print(f"   Fragment {i+1} ({frag_type}, significance: {frag_sig:.2f}): {frag_content[:80]}...")
                                
                            return dream_data
                        except json.JSONDecodeError as e:
                            print(f"{get_color('RED')}❌ Failed to parse JSON from LM Studio response: {e}{get_color('RESET')}")
                            print(f"Response content: {content[:200]}...")
                            return None
                    else:
                        print(f"{get_color('RED')}❌ Failed to get response from LM Studio. Status: {chat_response.status}{get_color('RESET')}")
                        return None
            except aiohttp.ClientConnectorError as e:
                print(f"{get_color('RED')}❌ Connection refused. The LM Studio server might not be running at {LM_STUDIO_URL}{get_color('RESET')}")
                return None
            except asyncio.TimeoutError as e:
                print(f"{get_color('RED')}❌ Request timed out after 120 seconds{get_color('RESET')}")
                return None
        except Exception as e:
            print(f"{get_color('RED')}❌ Error generating dream with LM Studio: {e}{get_color('RESET')}")
            return None

async def create_dream_report(dream_data: Dict[str, Any]):
    """Create a dream report using the generated dream data."""
    print(f"\n{get_color('BOLD')}{get_color('BG_BLUE')}===== CREATING DREAM REPORT ====={get_color('RESET')}")
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        create_report_url = f"{DREAM_API_URL}/test/create_test_report"
        
        try:
            async with session.post(create_report_url, json=dream_data) as response:
                if response.status == 200:
                    data = await response.json()
                    report_id = data.get("report_id")
                    fragment_count = data.get("fragment_count", 0)
                    print(f"{get_color('GREEN')}✅ Successfully created dream report: {report_id} with {fragment_count} fragments{get_color('RESET')}")
                    return report_id
                else:
                    print(f"{get_color('RED')}❌ Failed to create dream report. Status: {response.status}{get_color('RESET')}")
                    body = await response.text()
                    print(f"Response: {body[:200]}...")
                    return None
        except Exception as e:
            print(f"{get_color('RED')}❌ Error creating dream report: {e}{get_color('RESET')}")
            return None

async def refine_dream_report(report_id: str):
    """Refine the dream report through the reflection process."""
    print(f"\n{get_color('BOLD')}{get_color('BG_BLUE')}===== REFINING DREAM REPORT ====={get_color('RESET')}")
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        refine_url = f"{DREAM_API_URL}/test/refine_report"
        refine_data = {"report_id": report_id}
        
        try:
            async with session.post(refine_url, json=refine_data) as response:
                if response.status == 200:
                    data = await response.json()
                    refinement_count = data.get("refinement_count", 0)
                    confidence = data.get("confidence", 0)
                    reason = data.get("reason", "")
                    print(f"{get_color('GREEN')}✅ Successfully refined dream report:")
                    print(f"   Refinement count: {refinement_count}")
                    print(f"   New confidence: {confidence:.2f}")
                    print(f"   Reason: {reason}")
                    return data
                else:
                    print(f"{get_color('RED')}❌ Failed to refine dream report. Status: {response.status}{get_color('RESET')}")
                    body = await response.text()
                    print(f"Response: {body[:200]}...")
                    return None
        except Exception as e:
            print(f"{get_color('RED')}❌ Error refining dream report: {e}{get_color('RESET')}")
            return None

async def get_dream_report(report_id: str):
    """Get the full dream report with all fragments."""
    print(f"\n{get_color('BOLD')}{get_color('BG_BLUE')}===== RETRIEVING DREAM REPORT ====={get_color('RESET')}")
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        # First try the regular report endpoint
        get_report_url = f"{DREAM_API_URL}/report/{report_id}"
        report = None
        
        try:
            # Try the regular endpoint first
            async with session.get(get_report_url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"{get_color('GREEN')}✅ Successfully retrieved dream report from regular endpoint{get_color('RESET')}")
                    report = data.get("report", {})
                else:
                    print(f"Regular endpoint returned status: {response.status}, trying test endpoint...")
        except Exception as e:
            print(f"Error with regular endpoint: {e}, trying test endpoint...")
        
        # If the regular endpoint didn't work, try the test endpoint
        if not report:
            test_report_url = f"{DREAM_API_URL}/test/get_report"
            params = {"report_id": report_id}
            
            try:
                async with session.get(test_report_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"{get_color('GREEN')}✅ Successfully retrieved dream report from test endpoint{get_color('RESET')}")
                        report = data.get("report", {})
                    else:
                        print(f"{get_color('RED')}❌ Failed to get dream report. Status: {response.status}{get_color('RESET')}")
                        body = await response.text()
                        print(f"Response: {body[:200]}...")
                        return None
            except Exception as e:
                print(f"{get_color('RED')}❌ Error getting dream report: {e}{get_color('RESET')}")
                return None
        
        # Debug information about the report structure
        if report:
            print("Report data structure:")
            for key in report:
                print(f"  {key}: {type(report[key])}")
            
            # Check if we have the fragments dictionary
            if "fragments" in report and isinstance(report["fragments"], dict):
                print(f"Found {len(report['fragments'])} fragments in the report")
        
        return report

async def display_dream_report(report_data: Dict[str, Any]):
    """Display the dream report in a human-readable format."""
    print(f"\n{get_color('BOLD')}{get_color('BG_BLUE')}===== DREAM REPORT ====={get_color('RESET')}")
    
    # Extract data from the report structure
    title = report_data.get("title", "Untitled Dream Report")
    confidence = report_data.get("confidence", 0.0)
    created_at = report_data.get("created_at", time.time())
    updated_at = report_data.get("last_refined_at", report_data.get("modified", time.time()))
    domain = report_data.get("domain", "general")
    refinement_count = report_data.get("refinement_count", 0)
    
    # Extract fragment arrays by type
    insight_ids = report_data.get("insight_ids", [])
    question_ids = report_data.get("question_ids", [])
    hypothesis_ids = report_data.get("hypothesis_ids", [])
    counterfactual_ids = report_data.get("counterfactual_ids", [])
    
    # Create a mapping from ID to fragment type
    fragment_types = {}
    for fid in insight_ids:
        fragment_types[fid] = "insight"
    for fid in question_ids:
        fragment_types[fid] = "question"
    for fid in hypothesis_ids:
        fragment_types[fid] = "hypothesis"
    for fid in counterfactual_ids:
        fragment_types[fid] = "counterfactual"
    
    # Get the fragments dictionary
    fragments_dict = report_data.get("fragments", {})
    
    # Format timestamps with proper type handling
    def format_timestamp(ts):
        if isinstance(ts, str):
            try:
                # Try to parse ISO format string
                return ts
            except ValueError:
                return "Unknown time"
        elif isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        else:
            return "Unknown time"
    
    created_str = format_timestamp(created_at)
    updated_str = format_timestamp(updated_at)
    
    # Print report header
    print(f"\n{get_color('BOLD')}{'=' * 80}{get_color('RESET')}")
    print(f"{get_color('BOLD')}TITLE:{get_color('RESET')} {get_color('CYAN')}{title}{get_color('RESET')}")
    print(f"{get_color('BOLD')}DOMAIN:{get_color('RESET')} {domain}")
    print(f"{get_color('BOLD')}CONFIDENCE:{get_color('RESET')} {confidence:.2f}")
    print(f"{get_color('BOLD')}REFINEMENTS:{get_color('RESET')} {refinement_count}")
    print(f"{get_color('BOLD')}CREATED:{get_color('RESET')} {created_str}")
    print(f"{get_color('BOLD')}UPDATED:{get_color('RESET')} {updated_str}")
    print(f"{get_color('BOLD')}{'=' * 80}{get_color('RESET')}\n")
    
    # Organize fragments by type using our mapping
    fragments_by_type = {}
    for frag_id, fragment in fragments_dict.items():
        # Use our mapping to determine the type, or use the fragment's type field as fallback
        frag_type = fragment_types.get(frag_id) or fragment.get("type", "unknown")
        
        if frag_type not in fragments_by_type:
            fragments_by_type[frag_type] = []
        
        # Add the ID to the fragment since it's the key in the dictionary
        fragment["id"] = frag_id
        fragment["type"] = frag_type  # Ensure type is set correctly
        fragments_by_type[frag_type].append(fragment)
    
    # Define the order of types for display and their colors
    type_config = {
        "insight": {"color": get_color("GREEN"), "title": "INSIGHTS"},
        "question": {"color": get_color("BLUE"), "title": "QUESTIONS"},
        "hypothesis": {"color": get_color("YELLOW"), "title": "HYPOTHESES"},
        "counterfactual": {"color": get_color("MAGENTA"), "title": "COUNTERFACTUALS"},
        "unknown": {"color": get_color("GRAY"), "title": "OTHER FRAGMENTS"}
    }
    
    type_order = ["insight", "question", "hypothesis", "counterfactual", "unknown"]
    
    # Print fragments by type in the specified order
    total_fragments = 0
    if fragments_by_type:
        for frag_type in type_order:
            if frag_type in fragments_by_type:
                frags = fragments_by_type[frag_type]
                total_fragments += len(frags)
                config = type_config.get(frag_type, {"color": "", "title": frag_type.upper()})
                
                print(f"\n{config['color']}{get_color('BOLD')}--- {config['title']} ---{get_color('RESET')}")
                
                for i, fragment in enumerate(frags, 1):
                    content = fragment.get("content", "")
                    confidence = fragment.get("confidence", 0.0)
                    frag_id = fragment.get("id", "unknown")
                    conf_color = get_color("GREEN") if confidence > 0.7 else (get_color("YELLOW") if confidence > 0.4 else get_color("RED"))
                    
                    print(f"\n{config['color']}{i}.{get_color('RESET')} {content}")
                    print(f"   {get_color('GRAY')}ID: {frag_id}{get_color('RESET')}")
                    print(f"   {get_color('BOLD')}Confidence:{get_color('RESET')} {conf_color}{confidence:.2f}{get_color('RESET')}")
        
        # Print any remaining types not in our predefined order
        for frag_type, frags in fragments_by_type.items():
            if frag_type not in type_order:
                total_fragments += len(frags)
                print(f"\n{get_color('BOLD')}--- {frag_type.upper()} ---{get_color('RESET')}")
                
                for i, fragment in enumerate(frags, 1):
                    content = fragment.get("content", "")
                    confidence = fragment.get("confidence", 0.0)
                    frag_id = fragment.get("id", "unknown")
                    conf_color = get_color("GREEN") if confidence > 0.7 else (get_color("YELLOW") if confidence > 0.4 else get_color("RED"))
                    
                    print(f"\n{i}. {content}")
                    print(f"   {get_color('GRAY')}ID: {frag_id}{get_color('RESET')}")
                    print(f"   {get_color('BOLD')}Confidence:{get_color('RESET')} {conf_color}{confidence:.2f}{get_color('RESET')}")
    else:
        print(f"\n{get_color('RED')}No fragments found in the report.{get_color('RESET')}")
    
    # Print summary
    print(f"\n{get_color('BOLD')}{'=' * 80}{get_color('RESET')}")
    print(f"{get_color('BOLD')}Total fragments:{get_color('RESET')} {total_fragments}")
    if insight_ids:
        print(f"{get_color('GREEN')}Insights:{get_color('RESET')} {len(insight_ids)}")
    if question_ids:
        print(f"{get_color('BLUE')}Questions:{get_color('RESET')} {len(question_ids)}")
    if hypothesis_ids:
        print(f"{get_color('YELLOW')}Hypotheses:{get_color('RESET')} {len(hypothesis_ids)}")
    if counterfactual_ids:
        print(f"{get_color('MAGENTA')}Counterfactuals:{get_color('RESET')} {len(counterfactual_ids)}")
    print(f"{get_color('BOLD')}{'=' * 80}{get_color('RESET')}\n")

async def test_dreaming_flow():
    """Test the full dreaming flow in Lucidia, integrating multiple components."""
    try:
        # Check LM Studio connection
        print(f"\n{get_color('BOLD')}{get_color('BG_BLUE')}===== CHECKING LM STUDIO CONNECTION ====={get_color('RESET')}")
        lm_studio_loaded = await check_lm_studio_connection()
        if not lm_studio_loaded:
            print(f"{get_color('RED')}❌ LM Studio is not running or model is not loaded. Aborting.{get_color('RESET')}")
            return
        
        # Check Dream API connection
        print(f"\n{get_color('BOLD')}{get_color('BG_BLUE')}===== CHECKING DREAM API CONNECTION ====={get_color('RESET')}")
        dream_api_status = await check_dream_api_connection()
        if not dream_api_status:
            print(f"{get_color('RED')}❌ Dream API is not running. Aborting.{get_color('RESET')}")
            return
        
        # Add test memories
        print(f"\n{get_color('BOLD')}{get_color('BG_BLUE')}===== ADDING TEST MEMORIES ====={get_color('RESET')}")
        memory_result = await add_test_memories()
        if not memory_result:
            print(f"{get_color('RED')}❌ Failed to add test memories. Aborting.{get_color('RESET')}")
            return
        
        # Generate dream with LM Studio
        print(f"\n{get_color('BOLD')}{get_color('BG_BLUE')}===== GENERATING DREAM WITH LM STUDIO ====={get_color('RESET')}")
        dream_data = await generate_dream_with_lm_studio(memory_result)
        if not dream_data:
            print(f"{get_color('RED')}❌ Failed to generate dream with LM Studio. Aborting.{get_color('RESET')}")
            return
        
        # Create dream report
        print(f"\n{get_color('BOLD')}{get_color('BG_BLUE')}===== CREATING DREAM REPORT ====={get_color('RESET')}")
        report_id = await create_dream_report(dream_data)
        if not report_id:
            print(f"{get_color('RED')}❌ Failed to create dream report. Aborting.{get_color('RESET')}")
            return
        
        # Refine dream report
        print(f"\n{get_color('BOLD')}{get_color('BG_BLUE')}===== REFINING DREAM REPORT ====={get_color('RESET')}")
        refinement_result = await refine_dream_report(report_id)
        if not refinement_result:
            print(f"{get_color('YELLOW')}⚠️ Failed to refine dream report, but continuing...{get_color('RESET')}")
        
        # Get and display dream report
        report_data = await get_dream_report(report_id)
        if report_data:
            await display_dream_report(report_data)
        else:
            print(f"{get_color('RED')}❌ Failed to retrieve dream report.{get_color('RESET')}")
        
        print(f"\n{get_color('GREEN')}{get_color('BOLD')}===== DREAMING FLOW TEST COMPLETED ====={get_color('RESET')}")
        
    except Exception as e:
        print(f"{get_color('RED')}❌ Error during dreaming flow test: {e}{get_color('RESET')}")

async def test_integrated_dream_generation():
    """Test generating a dream using the natively integrated LM Studio JSON output."""
    print(f"\n{get_color('BOLD')}{get_color('BG_BLUE')}===== TESTING INTEGRATED LM STUDIO DREAM GENERATION ====={get_color('RESET')}")
    
    try:
        # Create a test configuration with LM Studio URL
        config = {
            "dream_processor": {
                "default_creativity": 0.8,
                "default_depth": 0.7,
                "dream_model": MODEL_NAME,
                "lm_studio_url": LM_STUDIO_URL,  # Add the LM Studio URL directly to the config
            }
        }
        
        # Create a knowledge graph for testing
        # Use a simple mock knowledge graph to avoid async issues
        knowledge_graph = LucidiaKnowledgeGraph()
        
        # Add a monkey patch for any async methods we need to make sync
        original_get_relationships = getattr(knowledge_graph, 'get_relationships', None)
        if original_get_relationships and asyncio.iscoroutinefunction(original_get_relationships):
            async def sync_get_relationships(*args, **kwargs):
                return []
            knowledge_graph.get_relationships = sync_get_relationships
        
        # Create memory client for testing
        memory_client = DummyMemoryClient(memories=TEST_MEMORIES)
        
        # Create the dream processor with LM Studio integration
        dream_processor = LucidiaDreamProcessor(
            knowledge_graph=knowledge_graph,
            config=config["dream_processor"]
        )
        
        # Set the memory client directly as an attribute
        dream_processor.memory_client = memory_client
        
        # Initialize dream state with some values (normally done by start_dreaming)
        dream_processor.is_dreaming = True
        dream_processor.dream_state = {
            "is_dreaming": True,
            "dream_start_time": datetime.now(),
            "current_dream_depth": 0.7,
            "current_dream_creativity": 0.8,
            "dream_duration": 600,
            "dream_intensity": 0.75,
            "emotional_valence": "neutral",
            "current_dream_seed": {"name": "reflection", "type": "concept"},
            "current_dream_insights": [],
            "current_spiral_phase": "observation",
        }
        
        print("Generating dream using direct LM Studio integration...")
        
        # Use a timeout to prevent hanging if there are issues with async code
        try:
            # Call _generate_dream_with_lm_studio directly instead of generate_dream
            dream_report = await asyncio.wait_for(
                dream_processor._generate_dream_with_lm_studio(LM_STUDIO_URL), 
                timeout=60
            )
        except asyncio.TimeoutError:
            print(f"{get_color('RED')}\u274c Dream generation timed out after 60 seconds{get_color('RESET')}")
            return
        
        # Display the results
        if dream_report and "error" in dream_report:
            print(f"{get_color('RED')}\u274c Error generating dream: {dream_report['error']}{get_color('RESET')}")
            return
        
        if not dream_report:
            print(f"{get_color('RED')}\u274c Failed to generate dream report{get_color('RESET')}")
            return
        
        # Clean output for better readability    
        print(f"{get_color('GREEN')}\u2705 Successfully generated dream with integrated LM Studio{get_color('RESET')}")
        print(f"   Dream ID: {dream_report.get('dream_id', 'Unknown')}")
        print(f"   Title: {dream_report.get('title', 'Untitled')}")
        print(f"   Theme: {dream_report.get('theme', {}).get('name', 'Unknown')}")
        print(f"   Spiral phase: {dream_report.get('spiral_phase', 'Unknown')}")
        
        # Process the insights from the dream report
        insights = dream_report.get('insights', [])
        if insights:
            print(f"   Number of insights: {len(insights)}")
            print("\nSample insights:")
            for i, insight in enumerate(insights[:3]):
                if isinstance(insight, dict):
                    # Handle insight structure from native dream_processor
                    if 'attributes' in insight:
                        content = insight.get('attributes', {}).get('content', 'No content')
                        significance = insight.get('attributes', {}).get('significance', 0)
                    else:
                        content = insight.get('content', 'No content')
                        significance = insight.get('significance', 0)
                    
                    # Truncate content if too long
                    if len(content) > 100:
                        content = content[:100] + "..."
                        
                    print(f"   Insight {i+1} (significance: {significance:.2f}): {content}")
        else:
            print("   No insights generated")
            
        # Process associations
        associations = dream_report.get('associations', [])
        if associations:
            print(f"   Number of associations: {len(associations)}")
        else:
            print("   No associations generated")
        
        # Print a sample of the dream content
        content = dream_report.get("dream_content", "")
        if content:
            # Format the content for better readability
            print(f"\nDream narrative excerpt:")
            excerpt = content[:300] + "..." if len(content) > 300 else content
            # Print with indentation for readability
            for line in excerpt.split('\n'):
                print(f"   {line}")
        else:
            print("\nNo dream narrative content")
            
        # Display meta-reflection if available
        meta_reflection = dream_report.get("meta_reflection", "")
        if meta_reflection:
            print(f"\nMeta-reflection:")
            excerpt = meta_reflection[:300] + "..." if len(meta_reflection) > 300 else meta_reflection
            for line in excerpt.split('\n'):
                print(f"   {line}")
        
    except Exception as e:
        print(f"{get_color('RED')}\u274c Error in integrated dream generation test: {e}{get_color('RESET')}")
        import traceback
        traceback.print_exc()

async def main():
    """Run the test cases."""
    print(f"{get_color('BOLD')}=== Testing Reflection Engine and Dream Integration ==={get_color('RESET')}")
    print(f"LM Studio URL: {LM_STUDIO_URL}")
    print(f"Model: {MODEL_NAME}")
    
    # Add the test memories
    memory_ids = await add_test_memories()
    
    # Test the reflection engine (original test)
    # This function doesn't exist yet, so commenting it out
    # await test_reflection_engine()
    
    # Test dream generation with LM Studio
    await generate_dream_with_lm_studio(memory_ids)
    
    # Test the integrated dream generation with LM Studio JSON output
    await test_integrated_dream_generation()
    
    print(f"\n{get_color('BOLD')}Tests completed.{get_color('RESET')}")

if __name__ == "__main__":
    asyncio.run(main())