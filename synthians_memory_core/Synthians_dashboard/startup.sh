#!/bin/bash

# Log startup information
echo "Starting Synthians Dashboard in dev mode..."
echo "Environment: $NODE_ENV"
echo "Ports: $PORT"
echo "Backend URLs:"
echo "  Memory Core: $MEMORY_CORE_URL"
echo "  Neural Memory: $NEURAL_MEMORY_URL"
echo "  CCE: $CCE_URL"

# Run the application in dev mode
npm run dev
