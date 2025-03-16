#!/bin/bash

# Update script for migrating from HPCSIGFlowManager to HPCQRFlowManager
echo "Starting QuickRecal integration update..."

# Create backup directory
BACKUP_DIR="./backup_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup directory at $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Backup important files before modification
cp -r ./memory/lucidia_memory_system/core/memory_core.py "$BACKUP_DIR/"
cp -r ./memory/lucidia_memory_system/core/long_term_memory.py "$BACKUP_DIR/"
cp -r ./memory/lucidia_memory_system/core/spiral_phases.py "$BACKUP_DIR/"
cp -r ./memory/lucidia_memory_system/core/dream_processor.py "$BACKUP_DIR/"
cp -r ./memory/lucidia_memory_system/core/Self/self_model.py "$BACKUP_DIR/"
cp -r ./server/tensor_server.py "$BACKUP_DIR/"
cp -r ./server/hpc_server.py "$BACKUP_DIR/"
cp -r ./server/memory_bridge.py "$BACKUP_DIR/"

echo "Files backed up successfully to $BACKUP_DIR"

# Restart the containers with the new configuration
echo "Restarting services with QuickRecal integration..."
docker-compose down
docker-compose up -d

echo "QuickRecal integration update completed successfully!"
echo "The system is now using the HPCQRFlowManager for memory processing."
echo "Service names have been updated to reflect the new approach."
echo "Backup of original files available at: $BACKUP_DIR"
