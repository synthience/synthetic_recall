#!/usr/bin/env python
# test_system_prompt.py

import sys

from voice_core.config.config import LLMConfig

def main():
    """Test the enhanced system prompt."""
    # Initialize LLM config
    config = LLMConfig()
    
    # Print the system prompt
    print("=== Lucidia's System Prompt ===")
    print(config.system_prompt)
    print("============================")

if __name__ == "__main__":
    main()
