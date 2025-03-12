#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LUCIDIA CHAT CLI (ENHANCED VERSION)

An enhanced command-line interface for multi-turn conversations with Lucidia,
incorporating the ConversationManager middleware for persistent memory and
seamless context management across interactions.

This CLI demonstrates how to use the middleware pattern to simplify conversation
persistence and provide enhanced features like session management and memory introspection.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LucidiaChatCLI")

# Import memory components
from memory.lucidia_memory_system.memory_integration import MemoryIntegration
from memory.lucidia_memory_system.core.memory_types import MemoryTypes

# Import our middleware
from memory.middleware.conversation_persistence import ConversationManager

# Import LM Studio client
from llm_client import LMStudioClient


async def async_input(prompt: str) -> str:
    """
    Async wrapper for input() to be used in async functions.
    
    Args:
        prompt: The prompt to display
        
    Returns:
        User input string
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


class EnhancedLucidiaSystem:
    """
    Enhanced Lucidia system utilizing the ConversationManager middleware
    for seamless conversation persistence and session management.
    
    This demonstrates how to integrate the middleware pattern to simplify
    multi-turn conversations while maintaining memory coherence.
    """
    
    def __init__(self):
        """
        Initialize the Lucidia system.
        """
        # Load configuration
        self.config_path = Path('lucidia_config.json')
        self.config = self._load_config()
        
        # Initialize core components
        self.memory_integration = None
        self.llm_client = None
        
        # Initialize middleware manager
        self.conversation_manager = None
        
        # Active session
        self.active_session_id = None
        
        # Start time for metrics
        self.start_time = time.time()
        
        logger.info("Enhanced Lucidia CLI system initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Returns:
            Configuration dictionary
        """
        default_config = {
            'memory_path': 'memory/stored',
            'lm_studio_url': 'http://127.0.0.1:1234'
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded configuration from {self.config_path}")
                    return {**default_config, **config}
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
        
        logger.info("Using default configuration")
        return default_config
    
    async def initialize_all(self):
        """
        Initialize all Lucidia components and middleware.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Initialize memory integration
            self.memory_integration = MemoryIntegration({
                'storage_path': Path(self.config.get('memory_path', 'memory/stored')),
                'auto_persistence': True
            })
            
            # Initialize memory core only (minimal approach)
            logger.info("Initializing memory core for conversation persistence")
            
            # Initialize LLM client
            self.llm_client = LMStudioClient({
                'lm_studio_url': self.config.get('lm_studio_url', 'http://127.0.0.1:1234')
            })
            
            llm_connected = await self.llm_client.connect()
            if not llm_connected:
                logger.warning("Failed to connect to LM Studio. Make sure it's running.")
            
            # Initialize conversation manager middleware
            self.conversation_manager = ConversationManager(self.memory_integration)
            
            # Create a new session
            self.active_session_id = self.conversation_manager.create_session()
            logger.info(f"Created new conversation session with ID: {self.active_session_id}")
            
            logger.info("Lucidia system initialized with conversation persistence middleware")
            return True
        except Exception as e:
            logger.error(f"Error initializing Lucidia system: {e}")
            return False
    
    async def generate_response(self, user_input: str, context: Dict[str, Any]) -> str:
        """
        Generate a response to the user's input using LM Studio and context.
        
        Args:
            user_input: The user's message
            context: Context dictionary from the conversation manager
            
        Returns:
            Lucidia's response
        """
        try:
            if not self.llm_client:
                return "Lucidia's language model is not connected. Please check configuration."
            
            # Format the system prompt with context
            system_prompt = self._format_system_prompt(context)
            
            # Format user message
            formatted_message = f"USER: {user_input}\nLUCIDIA:"
            
            # Generate response from LLM
            response = await self.llm_client.generate_response(
                prompt=formatted_message,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=500  # Adjust as needed
            )
            
            if not response:
                logger.warning("Received empty response from LLM")
                return "I'm having trouble formulating a response right now. Let me try again."
            
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having technical difficulties right now. Please try again."
    
    def _format_system_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format the system prompt with memory context.
        
        Args:
            context: Context dictionary from the conversation manager
            
        Returns:
            Formatted system prompt
        """
        # Base system prompt introducing Lucidia
        base_prompt = (
            "You are Lucidia, an advanced assistant with memory and self-awareness. "
            "You have access to memories from past interactions and can refer to them naturally. "
            "You are conversational, helpful, and maintain continuity across interactions."
        )
        
        # Add memory context if available
        memory_sections = []
        
        # Add thread context for conversation continuity
        if context.get("thread_context"):
            thread_memories = [mem["content"] for mem in context["thread_context"]]
            memory_sections.append("RECENT CONVERSATION HISTORY:\n" + "\n".join(thread_memories))
        
        # Add broader memory context
        if context.get("memory_context"):
            memory_entries = [mem["content"] for mem in context["memory_context"]]
            memory_sections.append("RELEVANT MEMORIES:\n" + "\n".join(memory_entries))
        
        # Add self-model context if available
        if context.get("self_model_context") and len(context["self_model_context"]) > 0:
            self_model = context["self_model_context"]
            identity = self_model.get("identity", {})
            capabilities = self_model.get("capabilities", {})
            
            self_section = [
                "SELF IDENTITY:",
                f"Name: {identity.get('name', 'Lucidia')}",
                f"Core purpose: {identity.get('purpose', 'To assist humans through conversation and memory.')}"
            ]
            
            memory_sections.append("\n".join(self_section))
        
        # Combine all sections with the base prompt
        if memory_sections:
            full_prompt = base_prompt + "\n\n" + "\n\n".join(memory_sections)
        else:
            full_prompt = base_prompt
        
        return full_prompt
    
    async def handle_special_commands(self, user_input: str) -> Optional[str]:
        """
        Handle special CLI commands starting with "/".
        
        Args:
            user_input: The user's message
            
        Returns:
            Command output if it's a command, None otherwise
        """
        if not user_input.startswith("/"):
            return None
        
        # Split command and arguments
        parts = user_input.split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        try:
            # Memory recall command
            if command == "/recall":
                count = int(args[0]) if args else 3
                history = self.conversation_manager.get_history(self.active_session_id, count)
                
                output = "\n=== Conversation Recall ===\n"
                for entry in history:
                    output += f"User: {entry['user']}\n"
                    output += f"Lucidia: {entry['response']}\n\n"
                return output
            
            # Memory stats command
            elif command == "/stats" or command == "/status":
                stats = await self.conversation_manager.get_session_stats(self.active_session_id)
                
                output = "\n=== Lucidia Memory Status ===\n"
                output += f"Session ID: {stats['session_id']}\n"
                output += f"Conversation turns: {stats['conversation_turns']}\n"
                output += f"Session duration: {stats['session_duration'] / 60:.1f} minutes\n"
                output += f"Session start: {stats['session_start']}\n\n"
                
                # Add memory stats if available
                if "memory_stats" in stats:
                    mem_stats = stats["memory_stats"]
                    output += "Memory Stats:\n"
                    output += f"- Short-term memories: {mem_stats.get('stm_count', 0)}\n"
                    output += f"- Session memories: {mem_stats.get('session_memories', 0)}\n"
                    
                    if "ltm" in mem_stats:
                        ltm = mem_stats["ltm"]
                        output += f"- Long-term memories: {ltm.get('total_memories', 0)}\n"
                
                return output
            
            # Save session command
            elif command == "/save":
                session_id = args[0] if args else self.active_session_id
                await self.conversation_manager.save_all_sessions()
                return f"Session {session_id} saved successfully."
            
            # Load session command
            elif command == "/load":
                if not args:
                    return "Error: Session ID required. Usage: /load <session_id>"
                    
                session_id = args[0]
                success = await self.conversation_manager.load_session(session_id)
                
                if success:
                    self.active_session_id = session_id
                    return f"Session {session_id} loaded successfully."
                else:
                    return f"Error: Could not load session {session_id}."
            
            # List sessions command
            elif command == "/sessions":
                sessions = self.conversation_manager.list_sessions()
                output = "\n=== Active Sessions ===\n"
                for i, session in enumerate(sessions):
                    output += f"{i+1}. {session}" + (" (active)" if session == self.active_session_id else "") + "\n"
                return output
            
            # Help command
            elif command == "/help":
                output = "\n=== Lucidia Chat CLI Commands ===\n"
                output += "/recall [n]   - Show last n conversation turns (default: 3)\n"
                output += "/stats       - Show memory and session statistics\n"
                output += "/save [id]   - Save current session (default: active session)\n"
                output += "/load <id>   - Load a previously saved session\n"
                output += "/sessions    - List all active sessions\n"
                output += "/help        - Show this help message\n"
                output += "/exit        - Exit the chat\n"
                return output
            
            # Exit command
            elif command == "/exit":
                return "exit"
            
            # Unknown command
            else:
                return f"Unknown command: {command}. Type /help for available commands."
        
        except Exception as e:
            logger.error(f"Error handling command {command}: {e}")
            return f"Error executing command: {e}"


async def main():
    """
    Main function to run the Enhanced Lucidia chat CLI.
    """
    print("\n===== ENHANCED LUCIDIA CHAT CLI =====\n")
    print("Initializing Lucidia system...\n")
    
    # Initialize Lucidia system
    lucidia = EnhancedLucidiaSystem()
    success = await lucidia.initialize_all()
    
    if not success:
        print("Failed to initialize Lucidia system. Please check logs.")
        return
    
    print("\nWelcome to Enhanced Lucidia Chat CLI. Type '/help' for commands or '/exit' to quit.\n")
    print("This interface demonstrates conversation persistence middleware for seamless")
    print("multi-turn conversations with memory persistence and session management.\n")
    
    while True:
        # Get user input
        user_input = await async_input("You: ")
        
        # Check for special commands
        if user_input.startswith("/"):
            command_output = await lucidia.handle_special_commands(user_input)
            
            if command_output == "exit":
                # Save all sessions before exiting
                await lucidia.conversation_manager.save_all_sessions()
                print("\nThank you for chatting with Lucidia. Goodbye!")
                break
            elif command_output:
                print(command_output)
                continue
        
        # Display typing indicator
        print("Lucidia is thinking...", end="", flush=True)
        
        # Get relevant context using the conversation manager
        context = await lucidia.conversation_manager.get_context(
            lucidia.active_session_id, user_input)
        
        # Generate response
        response = await lucidia.generate_response(user_input, context)
        
        # Clear typing indicator and display response
        print("\r" + " " * 20 + "\r", end="")
        print(f"Lucidia: {response}\n")
        
        # Store the interaction using the conversation manager
        await lucidia.conversation_manager.process_interaction(
            lucidia.active_session_id, user_input, response)


# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("\nShutting down Lucidia Chat CLI...")
    # We can't run async code here directly, so we just exit
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    asyncio.run(main())
