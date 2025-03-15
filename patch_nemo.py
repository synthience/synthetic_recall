#!/usr/bin/env python3
"""
Patch script to fix ModelFilter import in NeMo
"""
import os
import sys

def patch_nemo_common():
    """Patch the NeMo common.py file to fix ModelFilter import"""
    nemo_common_path = "/usr/local/lib/python3.10/dist-packages/nemo/core/classes/common.py"
    
    if not os.path.exists(nemo_common_path):
        print(f"Error: {nemo_common_path} not found")
        return False
    
    # Read the file
    with open(nemo_common_path, 'r') as f:
        content = f.read()
    
    # Replace the problematic import
    if "from huggingface_hub import HfApi, HfFolder, ModelFilter, hf_hub_download" in content:
        content = content.replace(
            "from huggingface_hub import HfApi, HfFolder, ModelFilter, hf_hub_download",
            "from huggingface_hub import HfApi, HfFolder, hf_hub_download\n# ModelFilter is deprecated, using direct parameters instead"
        )
        
        # Replace any ModelFilter usage with direct parameters
        if "filter=ModelFilter(" in content:
            content = content.replace("filter=ModelFilter(", "")
            content = content.replace(")", "")
        
        # Write the modified content back
        with open(nemo_common_path, 'w') as f:
            f.write(content)
        
        print(f"Successfully patched {nemo_common_path}")
        return True
    else:
        print(f"Import statement not found in {nemo_common_path}")
        return False

def patch_nemo_pretrained():
    """Patch the NeMo pretrained.py file to fix ModelFilter usage"""
    nemo_pretrained_path = "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/pretrained.py"
    
    if not os.path.exists(nemo_pretrained_path):
        print(f"Error: {nemo_pretrained_path} not found")
        return False
    
    # Read the file
    with open(nemo_pretrained_path, 'r') as f:
        content = f.read()
    
    # Replace any ModelFilter usage with direct parameters
    if "filter=ModelFilter(" in content:
        content = content.replace("filter=ModelFilter(", "")
        content = content.replace(")", "")
        
        # Write the modified content back
        with open(nemo_pretrained_path, 'w') as f:
            f.write(content)
        
        print(f"Successfully patched {nemo_pretrained_path}")
        return True
    else:
        print(f"ModelFilter usage not found in {nemo_pretrained_path}")
        return False

def main():
    """Main function to patch NeMo files"""
    success = patch_nemo_common()
    if success:
        patch_nemo_pretrained()
    else:
        print("Failed to patch NeMo common.py, skipping other files")
        sys.exit(1)

if __name__ == "__main__":
    main()
