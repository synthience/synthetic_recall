# requirements.txt - Add these new dependencies to your existing requirements

# For core functionality
livekit==0.9.0
edge-tts==6.1.5
python-dotenv==1.0.0
websockets==11.0.3
torch>=1.13.0
torchaudio>=0.13.0
numpy>=1.20.0
aiohttp>=3.8.5

# For enhanced audio processing
soundfile>=0.12.1
scipy>=1.7.0

# For RAG implementation
python-Levenshtein>=0.21.0  # Efficient edit distance calculation

# For processing voice corrections
pydub>=0.25.1  # Audio processing

# For handling transcription effectively
re>=2.2.1  # Regular expressions

# Optional but recommended
cachetools>=5.3.1  # More advanced caching
uvloop>=0.17.0  # Faster event loop for asyncio
orjson>=3.9.0  # Faster JSON processing

# Installation script
# save as install_dependencies.py

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies for RAG implementation."""
    print("Installing dependencies for RAG implementation...")
    
    # Core dependencies
    core_deps = [
        "livekit>=0.9.0",
        "edge-tts>=6.1.5",
        "python-dotenv>=1.0.0",
        "websockets>=11.0.3",
        "torch>=1.13.0",
        "numpy>=1.20.0",
        "aiohttp>=3.8.5"
    ]
    
    # Audio processing
    audio_deps = [
        "soundfile>=0.12.1",
        "scipy>=1.7.0",
        "pydub>=0.25.1"
    ]
    
    # RAG specific
    rag_deps = [
        "python-Levenshtein>=0.21.0"
    ]
    
    # Optional optimizations
    opt_deps = [
        "cachetools>=5.3.1",
        "uvloop>=0.17.0;platform_system!='Windows'",  # uvloop not available on Windows
        "orjson>=3.9.0"
    ]
    
    # Install all dependencies
    all_deps = core_deps + audio_deps + rag_deps + opt_deps
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + all_deps)
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("\nTrying to install dependencies one by one...")
        
        # Try installing one by one to identify problematic packages
        for dep in all_deps:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                print(f"Successfully installed {dep}")
            except subprocess.CalledProcessError:
                print(f"Failed to install {dep}, continuing with others...")

if __name__ == "__main__":
    install_dependencies()
