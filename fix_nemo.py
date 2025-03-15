#!/usr/bin/env python3
"""
Fix script for NeMo to handle ModelFilter issue
"""

import os
import re

# Define the ModelFilter class that's missing
modelfilter_code = '''
# Add ModelFilter class that's missing from newer huggingface_hub versions
class ModelFilter:
    """Filter for listing models on the Hugging Face Hub."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"ModelFilter({', '.join(f'{k}={v}' for k, v in self.kwargs.items())})"
'''

# Path to the file we need to modify
common_py_path = "/usr/local/lib/python3.10/dist-packages/nemo/core/classes/common.py"

# Check if the file exists
if not os.path.exists(common_py_path):
    print(f"Error: {common_py_path} not found")
    exit(1)

# Read the file content
with open(common_py_path, 'r') as f:
    content = f.read()

# Check if ModelFilter is referenced
if "ModelFilter" in content:
    # Find the import line for huggingface_hub
    huggingface_import = re.search(r'from\s+huggingface_hub\s+import.*', content)
    
    if huggingface_import:
        import_line = huggingface_import.group(0)
        
        # Remove ModelFilter from the import statement
        new_import_line = re.sub(r'\bModelFilter\s*,\s*', '', import_line)
        new_import_line = re.sub(r'\s*,\s*ModelFilter\b', '', new_import_line)
        
        # Insert our ModelFilter class after the modified import
        modified_content = content[:huggingface_import.start()] + new_import_line + '\n' + modelfilter_code + content[huggingface_import.end():]
        
        # Write the modified content back to the file
        with open(common_py_path, 'w') as f:
            f.write(modified_content)
        
        print(f"Successfully patched {common_py_path} - removed ModelFilter from import and added class definition")
    else:
        # If we can't find the import, just add it at the top of the file
        modified_content = "from huggingface_hub import HfApi, HfFolder, hf_hub_download\n" + modelfilter_code + content
        
        # Write the modified content back to the file
        with open(common_py_path, 'w') as f:
            f.write(modified_content)
        
        print(f"Added ModelFilter class at the top of {common_py_path}")
else:
    print(f"ModelFilter is not referenced in {common_py_path}")
