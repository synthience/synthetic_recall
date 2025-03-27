#!/bin/bash
set -e

echo "=========================================="
echo "FAISS GPU Setup for Synthians Memory Core"
echo "=========================================="

# Make sure we have pip installed and updated
pip install -U pip

# Check for NVIDIA GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "[+] NVIDIA GPU detected, checking details..."
    nvidia-smi
    
    # Check CUDA availability
    if [ -d "/usr/local/cuda" ] || [ -d "/usr/cuda" ]; then
        echo "[+] CUDA installation found"
        
        # Install FAISS with GPU support
        echo "[+] Installing FAISS with GPU support..."
        # First uninstall any existing FAISS packages
        pip uninstall -y faiss faiss-cpu faiss-gpu || true
        
        # Install with no cache to ensure fresh download
        pip install --no-cache-dir faiss-gpu
        
        # Verify installation
        if python -c "import faiss; print(f'[+] FAISS-GPU {getattr(faiss, \"__version__\", \"unknown\")} installed successfully')" 2>/dev/null; then
            echo "[+] FAISS GPU installation verified"
            
            # Set GPU memory optimization variables
            export CUDA_VISIBLE_DEVICES=all
            export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
            echo "[+] GPU environment variables set"
        else
            echo "[!] Warning: FAISS-GPU installation verification failed, falling back to CPU version"
            pip uninstall -y faiss-gpu || true
            pip install --no-cache-dir faiss-cpu
        fi
    else
        echo "[!] CUDA installation not found, installing CPU version of FAISS"
        pip uninstall -y faiss faiss-gpu || true
        pip install --no-cache-dir faiss-cpu
    fi
else
    echo "[i] No NVIDIA GPU detected, installing CPU version of FAISS"
    pip uninstall -y faiss faiss-gpu || true
    pip install --no-cache-dir faiss-cpu
    
    # Verify installation
    if python -c "import faiss; print(f'[+] FAISS-CPU {getattr(faiss, \"__version__\", \"unknown\")} installed successfully')" 2>/dev/null; then
        echo "[+] FAISS CPU installation verified"
    else
        echo "[!] Error: FAISS-CPU installation failed"
        exit 1
    fi
fi

echo "=========================================="
echo "[+] FAISS setup completed successfully"  
echo "=========================================="
