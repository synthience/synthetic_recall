#!/bin/bash

# Setup script for emotion models directory structure
# This ensures volume mount points exist for Docker without rebuilding

# Define the directories to create
MODEL_DIRS=(
  "./models"
  "./models/emotion"
  "./data/models"
  "./data/models/emotion"
)

# Create each directory if it doesn't exist
for dir in "${MODEL_DIRS[@]}"; do
  if [ ! -d "$dir" ]; then
    echo "Creating directory: $dir"
    mkdir -p "$dir"
  else
    echo "Directory already exists: $dir"
  fi
done

echo "\nEmotion model directories are ready."
echo "You can now place model files in ./models/emotion/"
echo "These will be available in Docker at /app/models/emotion"
echo "\nThe EmotionAnalyzer will load them automatically,"
echo "or download them if not found locally."
