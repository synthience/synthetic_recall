#!/usr/bin/env python

import logging
import os
import sys
import subprocess
import platform
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gpu_diagnostic")

class GPUDiagnostic:
    """A comprehensive GPU diagnostics tool for FAISS and PyTorch compatibility."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "cuda": {},
            "pytorch": {},
            "faiss": {},
            "compatibility": {}
        }
    
    def run_command(self, cmd):
        """Run a shell command and return output."""
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               shell=True, text=True, check=False)
            return result.stdout.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    def check_nvidia_smi(self):
        """Check NVIDIA driver and visible GPUs."""
        logger.info("Checking NVIDIA driver...")
        try:
            nvidia_smi = self.run_command("nvidia-smi")
            if "NVIDIA-SMI" in nvidia_smi:
                logger.info("✓ NVIDIA driver found")
                self.results["cuda"]["nvidia_driver"] = "available"
                self.results["cuda"]["nvidia_smi_output"] = nvidia_smi
                return True
            else:
                logger.warning("✗ NVIDIA driver not found or not responding")
                self.results["cuda"]["nvidia_driver"] = "not_available"
                return False
        except Exception as e:
            logger.error(f"Error checking NVIDIA driver: {str(e)}")
            self.results["cuda"]["nvidia_driver"] = f"error: {str(e)}"
            return False
    
    def check_nvcc(self):
        """Check CUDA compiler version."""
        logger.info("Checking NVCC version...")
        try:
            nvcc_version = self.run_command("nvcc --version")
            if "nvcc:" in nvcc_version:
                # Extract version
                for line in nvcc_version.split("\n"):
                    if "release" in line:
                        version = line.split("release")[1].strip().split(",")[0]
                        self.results["cuda"]["nvcc_version"] = version
                        logger.info(f"✓ NVCC version: {version}")
                        return version
            logger.warning("✗ NVCC not found")
            self.results["cuda"]["nvcc_version"] = "not_found"
            return None
        except Exception as e:
            logger.error(f"Error checking NVCC: {str(e)}")
            self.results["cuda"]["nvcc_version"] = f"error: {str(e)}"
            return None
    
    def check_pytorch(self):
        """Check PyTorch and its CUDA compatibility."""
        logger.info("Checking PyTorch...")
        try:
            import torch
            self.results["pytorch"]["version"] = torch.__version__
            logger.info(f"✓ PyTorch version: {torch.__version__}")
            
            # Check CUDA availability in PyTorch
            cuda_available = torch.cuda.is_available()
            self.results["pytorch"]["cuda_available"] = cuda_available
            if cuda_available:
                logger.info("✓ PyTorch CUDA is available")
                # Check CUDA version used by PyTorch
                cuda_version = torch.version.cuda
                self.results["pytorch"]["cuda_version"] = cuda_version
                logger.info(f"✓ PyTorch CUDA version: {cuda_version}")
                
                # Check device count and names
                device_count = torch.cuda.device_count()
                self.results["pytorch"]["device_count"] = device_count
                logger.info(f"✓ PyTorch sees {device_count} CUDA devices")
                
                devices = []
                for i in range(device_count):
                    name = torch.cuda.get_device_name(i)
                    devices.append({"id": i, "name": name})
                self.results["pytorch"]["devices"] = devices
                logger.info(f"✓ CUDA devices: {', '.join([d['name'] for d in devices])}")
                
                return True
            else:
                logger.warning("✗ PyTorch CUDA is NOT available")
                return False
        except ImportError:
            logger.warning("✗ PyTorch is not installed")
            self.results["pytorch"]["installed"] = False
            return False
        except Exception as e:
            logger.error(f"Error checking PyTorch: {str(e)}")
            self.results["pytorch"]["error"] = str(e)
            return False
    
    def check_torchvision(self):
        """Check TorchVision compatibility."""
        logger.info("Checking TorchVision...")
        try:
            import torchvision
            self.results["pytorch"]["torchvision_version"] = torchvision.__version__
            logger.info(f"✓ TorchVision version: {torchvision.__version__}")
            
            # Test a basic torchvision operation
            try:
                # Try to use a transform (a common operation)
                from torchvision import transforms
                _ = transforms.ToTensor()
                logger.info("✓ TorchVision transforms working correctly")
                self.results["pytorch"]["torchvision_transforms"] = "working"
                return True
            except Exception as e:
                logger.warning(f"✗ TorchVision transforms error: {str(e)}")
                self.results["pytorch"]["torchvision_transforms"] = f"error: {str(e)}"
                return False
        except ImportError:
            logger.warning("✗ TorchVision is not installed")
            self.results["pytorch"]["torchvision_installed"] = False
            return False
        except Exception as e:
            logger.error(f"Error checking TorchVision: {str(e)}")
            self.results["pytorch"]["torchvision_error"] = str(e)
            return False
    
    def check_faiss(self):
        """Check FAISS installation and GPU support."""
        logger.info("Checking FAISS...")
        try:
            import faiss
            self.results["faiss"]["installed"] = True
            
            # Check FAISS version
            version = getattr(faiss, "__version__", "unknown")
            self.results["faiss"]["version"] = version
            logger.info(f"✓ FAISS version: {version}")
            
            # Check GPU support
            has_gpu = hasattr(faiss, "StandardGpuResources")
            self.results["faiss"]["gpu_support"] = has_gpu
            
            if has_gpu:
                logger.info("✓ FAISS has GPU support")
                # Try to create a small index to verify GPU works
                try:
                    import numpy as np
                    dimension = 64
                    logger.info("Testing FAISS GPU index creation...")
                    
                    # First test CPU index as baseline
                    cpu_index = faiss.IndexFlatL2(dimension)
                    xb = np.random.random((10, dimension)).astype('float32')
                    cpu_index.add(xb)
                    logger.info("✓ FAISS CPU index creation successful")
                    
                    # Now test GPU index
                    logger.info("Attempting to create FAISS GPU resources...")
                    res = faiss.StandardGpuResources()
                    logger.info("✓ FAISS GPU resources created")
                    
                    # Configure resource to be safer
                    try:
                        # This might not be available in all FAISS versions
                        if hasattr(faiss, "GpuResourcesConfig"):
                            gpu_config = faiss.GpuResourcesConfig()
                            gpu_config.pinned_memory = True
                            gpu_config.temp_memory = 64 * 1024 * 1024  # 64 MB
                            res.setConfig(gpu_config)
                            logger.info("✓ FAISS GPU resources configured with custom options")
                        else:
                            logger.info("FAISS GpuResourcesConfig not available")
                    except Exception as e:
                        logger.warning(f"Couldn't configure GPU resources: {str(e)}")
                        self.results["faiss"]["config_error"] = str(e)
                    
                    logger.info("Attempting to move index to GPU...")
                    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                    logger.info("✓ FAISS GPU index creation successful")
                    
                    # Try to add vectors to GPU index
                    xq = np.random.random((1, dimension)).astype('float32')
                    gpu_index.add(xq)
                    logger.info("✓ FAISS GPU index add operation successful")
                    
                    # Try to search
                    D, I = gpu_index.search(xq, 5)
                    logger.info("✓ FAISS GPU index search operation successful")
                    
                    self.results["faiss"]["gpu_working"] = True
                    return True
                except Exception as e:
                    logger.error(f"Error testing FAISS GPU: {str(e)}")
                    self.results["faiss"]["gpu_error"] = str(e)
                    self.results["faiss"]["gpu_working"] = False
                    return False
            else:
                logger.warning("✗ FAISS doesn't have GPU support")
                return False
            
        except ImportError:
            logger.warning("✗ FAISS is not installed")
            self.results["faiss"]["installed"] = False
            return False
        except Exception as e:
            logger.error(f"Error checking FAISS: {str(e)}")
            self.results["faiss"]["error"] = str(e)
            return False
    
    def check_compatibility(self):
        """Check compatibility between CUDA, PyTorch, and FAISS."""
        # Compare PyTorch CUDA version with system CUDA version
        if "cuda_version" in self.results["pytorch"] and "nvcc_version" in self.results["cuda"]:
            pt_cuda = self.results["pytorch"]["cuda_version"]
            nvcc_cuda = self.results["cuda"]["nvcc_version"]
            
            # Simple string comparison, could be improved for minor version differences
            if pt_cuda == nvcc_cuda:
                logger.info(f"✓ PyTorch CUDA version ({pt_cuda}) matches system CUDA version ({nvcc_cuda})")
                self.results["compatibility"]["pytorch_cuda_match"] = True
            else:
                logger.warning(f"✗ PyTorch CUDA version ({pt_cuda}) doesn't match system CUDA version ({nvcc_cuda})")
                self.results["compatibility"]["pytorch_cuda_match"] = False
                self.results["compatibility"]["pytorch_cuda_mismatch"] = f"PyTorch: {pt_cuda}, NVCC: {nvcc_cuda}"
        
        # Check for shared symbols between PyTorch and FAISS
        if self.results["pytorch"].get("cuda_available", False) and self.results["faiss"].get("gpu_support", False):
            if self.results["faiss"].get("gpu_working", False):
                logger.info("✓ FAISS and PyTorch GPU integration is working")
                self.results["compatibility"]["faiss_pytorch_gpu_integration"] = "working"
            else:
                logger.warning("✗ FAISS and PyTorch GPU integration is NOT working")
                self.results["compatibility"]["faiss_pytorch_gpu_integration"] = "not_working"
                if "gpu_error" in self.results["faiss"]:
                    error = self.results["faiss"]["gpu_error"]
                    if "undefined symbol" in error.lower():
                        logger.error("✗ Symbol mismatch detected between FAISS and CUDA runtime")
                        self.results["compatibility"]["symbol_mismatch"] = True
                        # Try to extract the symbol name
                        import re
                        symbol_match = re.search(r"undefined symbol: ([^\s]+)", error.lower())
                        if symbol_match:
                            symbol = symbol_match.group(1)
                            self.results["compatibility"]["mismatched_symbol"] = symbol
    
    def check_shared_libraries(self):
        """Check shared libraries that might be related to CUDA."""
        logger.info("Checking shared libraries...")
        try:
            # Check LD_LIBRARY_PATH
            ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
            self.results["cuda"]["ld_library_path"] = ld_library_path
            logger.info(f"LD_LIBRARY_PATH: {ld_library_path}")
            
            # Check for libcuda.so
            libcuda_path = self.run_command("find /usr -name 'libcuda.so*' 2>/dev/null")
            self.results["cuda"]["libcuda_path"] = libcuda_path
            if libcuda_path:
                logger.info(f"✓ Found libcuda.so at: {libcuda_path}")
            else:
                logger.warning("✗ libcuda.so not found in /usr")
            
            # Check for PyTorch CUDA libs
            pytorch_cuda_libs = self.run_command("find /usr/local -name '*cuda*.so*' 2>/dev/null")
            self.results["pytorch"]["cuda_libs"] = pytorch_cuda_libs
            if pytorch_cuda_libs:
                logger.info(f"✓ Found PyTorch CUDA libraries")
            else:
                logger.warning("✗ No PyTorch CUDA libraries found in /usr/local")
            
            # Check CUDA_VISIBLE_DEVICES
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            self.results["cuda"]["cuda_visible_devices"] = cuda_visible_devices
            logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
            
            # Check if we have LD_PRELOAD
            ld_preload = os.environ.get("LD_PRELOAD", "")
            self.results["cuda"]["ld_preload"] = ld_preload
            if ld_preload:
                logger.info(f"LD_PRELOAD: {ld_preload}")
            else:
                logger.info("LD_PRELOAD not set")
                
        except Exception as e:
            logger.error(f"Error checking shared libraries: {str(e)}")
            self.results["cuda"]["shared_libs_error"] = str(e)
    
    def recommend_fixes(self):
        """Recommend fixes based on diagnostic results."""
        logger.info("\n===== RECOMMENDATIONS =====")
        
        recommendations = []
        
        # Check if PyTorch CUDA is mismatched with NVCC
        if self.results["compatibility"].get("pytorch_cuda_match") is False:
            pt_cuda = self.results["pytorch"].get("cuda_version")
            nvcc_cuda = self.results["cuda"].get("nvcc_version")
            rec = f"Reinstall PyTorch with CUDA {nvcc_cuda} support: \n"
            rec += f"pip uninstall -y torch torchvision torchaudio\n"
            rec += f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{nvcc_cuda.replace('.', '')}"
            recommendations.append(rec)
        
        # Check if FAISS GPU is not working
        if self.results["faiss"].get("gpu_working") is False:
            if "undefined symbol" in str(self.results["faiss"].get("gpu_error", "")).lower():
                rec = "Fix symbol mismatch by setting LD_PRELOAD:\n"
                rec += "Add this to your Dockerfile or docker-compose.yml:\n"
                rec += "ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1"
                recommendations.append(rec)
            
            rec = "Consider reinstalling FAISS with correct CUDA version:\n"
            rec += "pip uninstall -y faiss-gpu\n"
            rec += "pip install faiss-gpu"
            recommendations.append(rec)
        
        # Check if PyTorch or TorchVision has issues
        if self.results["pytorch"].get("torchvision_transforms") == "error":
            rec = "TorchVision appears to have CUDA compatibility issues. Try reinstalling with matching CUDA version."
            recommendations.append(rec)
        
        # Always recommend the fallback handler
        rec = "Ensure your code includes proper fallback when GPU initialization fails:\n"
        rec += """```python
try:
    # Try GPU
    res = faiss.StandardGpuResources()
    # Set lower temp memory to avoid CUDA OOM
    res.setTempMemory(64 * 1024 * 1024)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    logger.info("Using GPU FAISS index")
    return gpu_index
except Exception as e:
    logger.warning(f"GPU index failed, falling back to CPU: {e}")
    return cpu_index
```"""
        recommendations.append(rec)
        
        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"\nRecommendation {i}:\n{rec}")
        
        self.results["recommendations"] = recommendations
        
        if not recommendations:
            logger.info("No issues found that require fixes!")
    
    def run_diagnostics(self):
        """Run all diagnostic checks."""
        logger.info("\n===== Starting GPU diagnostics =====\n")
        
        # System and environment checks
        logger.info(f"Running on {platform.platform()}")
        logger.info(f"Python version: {sys.version}")
        
        # Check NVIDIA driver and CUDA
        driver_ok = self.check_nvidia_smi()
        if driver_ok:
            self.check_nvcc()
            self.check_shared_libraries()
        
        # Check PyTorch and TorchVision
        pytorch_ok = self.check_pytorch()
        if pytorch_ok:
            self.check_torchvision()
        
        # Check FAISS
        faiss_ok = self.check_faiss()
        
        # Check compatibility between components
        if driver_ok and pytorch_ok and faiss_ok:
            self.check_compatibility()
        
        # Provide recommendations
        self.recommend_fixes()
        
        logger.info("\n===== GPU diagnostics completed =====\n")
        return self.results

def main():
    diagnostic = GPUDiagnostic()
    diagnostic.run_diagnostics()
    return 0

if __name__ == "__main__":
    sys.exit(main())
