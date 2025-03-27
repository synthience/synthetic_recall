#!/usr/bin/env python

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

def run_tests(args):
    """Run the Synthians Memory Core test suite with the specified options."""
    # Construct the pytest command
    cmd = ["pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add test selection options
    if args.markers:
        for marker in args.markers.split(","):
            cmd.append(f"-m {marker}")
    
    if args.module:
        cmd.append(args.module)
    
    if args.test:
        cmd.append(f"-k {args.test}")
    
    # Add parallel execution if specified
    if args.parallel:
        cmd.append(f"-xvs -n {args.parallel}")
    
    # Add report options
    if args.report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join("test_reports", f"report_{timestamp}")
        
        # Create the report directory if it doesn't exist
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Add HTML report
        cmd.append(f"--html={report_path}.html")
        
        # Add JUnit XML report for CI integration
        cmd.append(f"--junitxml={report_path}.xml")
    
    # Add asyncio mode
    cmd.append("--asyncio-mode=auto")
    
    # Join the command parts
    cmd_str = " ".join(cmd)
    print(f"Running: {cmd_str}")
    
    # Execute the command
    start_time = time.time()
    result = subprocess.run(cmd_str, shell=True)
    elapsed_time = time.time() - start_time
    
    print(f"\nTests completed in {elapsed_time:.2f} seconds with exit code {result.returncode}")
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Run Synthians Memory Core test suite")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-m", "--markers", help="Comma-separated list of markers to run (e.g., 'smoke,integration')")
    parser.add_argument("-k", "--test", help="Expression to filter tests by name")
    parser.add_argument("-t", "--module", help="Specific test module to run (e.g., 'test_api_health.py')")
    parser.add_argument("-p", "--parallel", type=int, help="Run tests in parallel with specified number of processes")
    parser.add_argument("-r", "--report", action="store_true", help="Generate HTML and XML test reports")
    parser.add_argument("--url", help="Override the API server URL (default: http://localhost:5010)")
    
    args = parser.parse_args()
    
    # Set environment variables
    if args.url:
        os.environ["SYNTHIANS_TEST_URL"] = args.url
    
    print("=== Synthians Memory Core Test Runner ===")
    print(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set the working directory to the script's directory
    original_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        return run_tests(args)
    finally:
        # Restore original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    sys.exit(main())
