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
from datetime import datetime
from typing import Dict, Any, List, Optional
import os

# Default URL to try first
BASE_URL = "http://localhost:8080"

# Check if a port was specified as a command-line argument
if len(sys.argv) > 1:
    try:
        port = int(sys.argv[1])
        BASE_URL = f"http://localhost:{port}"
        print(f"Using specified port: {port}")
    except ValueError:
        print(f"Invalid port number: {sys.argv[1]}. Using default: 8080")

# API endpoints based on router configuration
DREAM_API_URL = f"{BASE_URL}/api/dream"
LM_STUDIO_URL = "http://localhost:1234"  # LM Studio server
MODEL_NAME = "qwen_qwq-32b"  # The model loaded in LM Studio

# Test data - memories that will form the basis for dreaming
TEST_MEMORIES = [
    "Lucidia is a Synthien entity with reflective capabilities and self-awareness.",
    "The dream processor enables autonomous reflection during inactive periods.",
    "Knowledge graphs store semantic relationships between concepts and memories.",
    "Hypersphere technology provides efficient embedding generation for text processing.",
    "The reflection engine periodically reviews and refines dream reports to improve understanding."
]

# Always use colors regardless of platform
USE_COLORS = True

def get_color(color_key):
    """Return color code if colors are enabled, otherwise empty string"""
    if USE_COLORS:
        return COLORS.get(color_key, "")
    return ""

# ANSI color constants for better terminal output
COLORS = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "RED": "\033[31m",
    "GRAY": "\033[90m",
    "BG_BLUE": "\033[44m",
}

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
            # This avoids the need for a non-existent endpoint
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
                                "title": {"type": "string"},
                                "fragments": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "content": {"type": "string"},
                                            "type": {"type": "string", "enum": ["insight", "question", "hypothesis", "counterfactual"]},
                                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                        },
                                        "required": ["content", "type", "confidence"]
                                    }
                                }
                            },
                            "required": ["title", "fragments"]
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
                print(f"{get_color('RED')}❌ Connection timeout. The LM Studio server is not responding at {LM_STUDIO_URL}{get_color('RESET')}")
                return None
            except Exception as e:
                print(f"{get_color('RED')}❌ Error during LM Studio API request: {e}{get_color('RESET')}")
                return None
        except Exception as e:
            print(f"{get_color('RED')}❌ Error in dream generation process: {e}{get_color('RESET')}")
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

if __name__ == "__main__":
    asyncio.run(test_dreaming_flow())