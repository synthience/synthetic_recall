"""
Test script to check the correct import and usage of deepfilternet
"""

# Try different import approaches
print("Attempting imports...")

try:
    import deepfilternet
    print("✅ Successfully imported deepfilternet")
    print(f"Module location: {deepfilternet.__file__}")
    print(f"Available attributes: {dir(deepfilternet)}")
except ImportError as e:
    print(f"❌ Failed to import deepfilternet: {e}")

try:
    from deepfilternet import DeepFilterNet
    print("✅ Successfully imported DeepFilterNet class")
except ImportError as e:
    print(f"❌ Failed to import DeepFilterNet: {e}")

try:
    import df
    print("✅ Successfully imported df")
    print(f"Module location: {df.__file__}")
    print(f"Available attributes: {dir(df)}")
except ImportError as e:
    print(f"❌ Failed to import df: {e}")

try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    print("✅ Successfully imported df.enhance functions")
except ImportError as e:
    print(f"❌ Failed to import df.enhance: {e}")

# Try to find the package in site-packages
import sys
import os

print("\nSearching for deepfilter in site-packages...")
for path in sys.path:
    if "site-packages" in path:
        try:
            contents = os.listdir(path)
            deepfilter_items = [item for item in contents if "deep" in item.lower() and "filter" in item.lower()]
            if deepfilter_items:
                print(f"Found in {path}:")
                for item in deepfilter_items:
                    print(f"  - {item}")
        except Exception as e:
            print(f"Error checking {path}: {e}")
