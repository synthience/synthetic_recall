#!/usr/bin/env python
"""
Helper script to run LTM Converter with automatic dependency installation
"""

import os
import sys
import subprocess

def install_dependencies():
    """Install required dependencies"""
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    print(f"Installing dependencies from {requirements_file}...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
        print("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def run_ltm_converter():
    """Run the LTM converter with the command-line arguments"""
    ltm_converter_path = os.path.join(os.path.dirname(__file__), 'ltm_converter.py')
    print(f"Running LTM converter at {ltm_converter_path}...")
    cmd = [sys.executable, ltm_converter_path] + sys.argv[1:]
    try:
        subprocess.check_call(cmd)
        print("LTM converter completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running LTM converter: {e}")
        return False

if __name__ == "__main__":
    print("========== LTM Converter Wrapper ==========\n")
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies. Exiting.")
        sys.exit(1)
    
    # Step 2: Run the LTM converter
    if not run_ltm_converter():
        print("LTM converter execution failed. Exiting.")
        sys.exit(1)
    
    print("\n========== All tasks completed successfully ==========\n")
