"""
Check Python path and installed packages
"""

import sys
import subprocess
import os

def main():
    # Print Python path
    print("Python executable:", sys.executable)
    print("\nPython path:")
    for path in sys.path:
        print(f"  - {path}")
    
    # Check for faster-whisper using pip
    print("\nChecking for faster-whisper using pip...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "faster-whisper"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("faster-whisper is installed:")
            print(result.stdout)
        else:
            print("faster-whisper is not installed according to pip")
            print(result.stderr)
    except Exception as e:
        print(f"Error checking pip: {e}")
    
    # Try to find the package manually
    print("\nSearching for faster-whisper in site-packages...")
    for path in sys.path:
        if "site-packages" in path:
            try:
                contents = os.listdir(path)
                faster_whisper_items = [item for item in contents if "faster" in item.lower() and "whisper" in item.lower()]
                if faster_whisper_items:
                    print(f"Found in {path}:")
                    for item in faster_whisper_items:
                        print(f"  - {item}")
            except Exception as e:
                print(f"Error checking {path}: {e}")
    
    # Check if we can import the package components
    print("\nTrying to import faster_whisper components...")
    try:
        import faster_whisper
        print("Successfully imported faster_whisper as a module")
        print(f"Module location: {faster_whisper.__file__}")
    except ImportError as e:
        print(f"Failed to import faster_whisper: {e}")
        
        # Try with underscores
        try:
            import faster_whisper
            print("Successfully imported faster_whisper (with underscore)")
        except ImportError as e:
            print(f"Failed to import faster_whisper (with underscore): {e}")

if __name__ == "__main__":
    main()
