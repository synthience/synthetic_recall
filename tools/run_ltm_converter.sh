#!/bin/bash

# Install required dependencies first
pip install -r tools/requirements.txt

# Run the LTM converter with all the passed arguments
python tools/ltm_converter.py "$@"
