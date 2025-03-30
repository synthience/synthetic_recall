#!/usr/bin/env python

"""
Simple script to check environment variables and write them to a file.
Useful for diagnosing container environment issues.
"""

import os
import json
import socket
from pathlib import Path
from datetime import datetime


def main():
    # Create a dictionary with all environment variables
    env_vars = dict(os.environ)
    
    # Add some system information
    env_vars['_hostname'] = socket.gethostname()
    env_vars['_timestamp'] = datetime.now().isoformat()
    
    # Check specific variables of interest
    cce_dev_mode = os.environ.get("CCE_DEV_MODE", "NOT_SET")
    titans_variant = os.environ.get("TITANS_VARIANT", "NOT_SET")
    
    # Print key variables to stdout
    print(f"\nCRITICAL ENVIRONMENT VARIABLES:\n")
    print(f"CCE_DEV_MODE = '{cce_dev_mode}'")
    print(f"TITANS_VARIANT = '{titans_variant}'\n")
    
    # Write to a file in the current directory
    output_path = Path("env_variables.json")
    with open(output_path, 'w') as f:
        json.dump(env_vars, f, indent=2, sort_keys=True)
    
    print(f"Wrote {len(env_vars)} environment variables to {output_path.absolute()}")


if __name__ == "__main__":
    main()
