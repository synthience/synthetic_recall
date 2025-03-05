"""
Test script to verify that all required packages are correctly installed.
This script attempts to import each package and reports success or failure.
"""

import sys
import importlib

def test_import(package_name):
    """Test importing a package and report success or failure."""
    try:
        # For packages with version specifiers, extract just the package name
        if '>=' in package_name:
            package_name = package_name.split('>=')[0]
        elif '==' in package_name:
            package_name = package_name.split('==')[0]
        
        # Special case for packages with @ notation
        if '@' in package_name:
            package_name = package_name.split('@')[0].strip()
            
        # Handle special cases
        if package_name == 'openai-whisper':
            package_name = 'whisper'
        elif package_name == 'ffmpeg-python':
            package_name = 'ffmpeg'
        elif package_name == 'faster-whisper':
            package_name = 'faster_whisper'  # Use underscore instead of hyphen
        elif package_name == 'df-nightly':
            package_name = 'df'  # DeepFilter package
        elif package_name == 'pyannote.audio':
            package_name = 'pyannote.audio'
            
        importlib.import_module(package_name)
        print(f"✅ Successfully imported {package_name}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {package_name}: {e}")
        return False

def test_df_import():
    """Test df-nightly specific imports and print available modules"""
    try:
        import df
        print("\nTesting df-nightly imports:")
        print(f"df version: {df.__version__ if hasattr(df, '__version__') else 'unknown'}")
        print("\nAvailable modules in df:")
        for item in dir(df):
            if not item.startswith('_'):
                print(f"- {item}")
                
        print("\nContents of df.enhance:")
        import df.enhance
        for item in dir(df.enhance):
            if not item.startswith('_'):
                print(f"- {item}")
    except Exception as e:
        print(f"Error testing df-nightly: {e}")

def main():
    # List of packages to test
    packages = [
        "torch",
        "torchaudio",
        "numpy",
        "soundfile",
        "scipy",
        "openai-whisper",
        "faster-whisper",
        "vosk",
        "webrtcvad",
        "torchvision",
        "librosa",
        "ffmpeg-python",
        "aiohttp",
        "loguru",
        "df-nightly",
        "pyannote.audio"
    ]
    
    print(f"Testing imports for {len(packages)} packages...\n")
    
    successful = 0
    failed = []
    
    for package in packages:
        if test_import(package):
            successful += 1
        else:
            failed.append(package)
    
    # Summary
    print("\n" + "="*80)
    print(f"Import Test Summary:")
    print(f"Successfully imported: {successful}/{len(packages)}")
    
    if failed:
        print(f"\nFailed imports ({len(failed)}):")
        for pkg in failed:
            print(f"  - {pkg}")
    else:
        print("\nAll packages imported successfully!")

if __name__ == "__main__":
    main()
    print("\n=== Testing df-nightly specifically ===")
    test_df_import()
