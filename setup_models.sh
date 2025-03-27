#!/bin/bash

# Setup script for model directories in Lucidia
# Creates all necessary directories for Docker volume mounts

echo "Setting up model directories for Lucidia..."

# Define required directories
DIRECTORIES=(
  "./models"
  "./models/emotion"
  "./memory"
  "./memory/stored"
  "./memory/stored/synthians"
)

# Create each directory
for dir in "${DIRECTORIES[@]}"; do
  if [ ! -d "$dir" ]; then
    echo "Creating directory: $dir"
    mkdir -p "$dir"
  else
    echo "Directory already exists: $dir"
  fi
done

# Create a .gitkeep file in each directory to ensure it's tracked by git
for dir in "${DIRECTORIES[@]}"; do
  if [ ! -f "$dir/.gitkeep" ]; then
    touch "$dir/.gitkeep"
    echo "Created .gitkeep in $dir"
  fi
done

echo ""
echo "Setup complete! Directory structure is ready for Docker."
echo ""
echo "To use custom emotion models:"
echo "1. Place model files in ./models/emotion/"
echo "2. Start Docker with: docker-compose up -d"
echo ""
echo "The EmotionAnalyzer will automatically:"
echo "- Use local models if found in ./models/emotion/"
echo "- Download models from HuggingFace if not found locally"
echo "- Cache downloaded models for future use"
