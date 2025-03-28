#!/usr/bin/env python

import os
import sys
import subprocess

# Function to downgrade NumPy
def downgrade_numpy():
    print("Downgrading NumPy to 1.26.4 to fix binary incompatibility issues...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "numpy==1.26.4"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("Successfully downgraded NumPy")
        return True
    else:
        print(f"Error downgrading NumPy: {result.stderr}")
        return False

if __name__ == "__main__":
    # Downgrade NumPy first
    if downgrade_numpy():
        # Now run the actual server command with the remaining arguments
        if len(sys.argv) > 1:
            command = sys.argv[1:]
            print(f"Executing: {' '.join(command)}")
            # Execute the command with the updated environment
            os.execvp(command[0], command)
        else:
            print("No command specified to run after NumPy downgrade")
    else:
        print("Failed to downgrade NumPy, proceeding with caution...")
        if len(sys.argv) > 1:
            command = sys.argv[1:]
            print(f"Executing anyway: {' '.join(command)}")
            # Execute the command with the updated environment
            os.execvp(command[0], command)
