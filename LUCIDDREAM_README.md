# Luciddream Docker Setup

## Overview

This document explains how to set up and run the Luciddream Docker container, which integrates with the existing HPC and Tensor servers to provide dream processing capabilities for the Lucidia memory system.

## Architecture

The Luciddream system uses a Docker-based architecture with the following components:

1. **Main container (nemo_sig_v3)** - Already existing, running multiple services:
   - `tensor_server.py`: WebSocket server for embedding generation (port 5001)
   - `hpc_server.py`: WebSocket server for high-performance computing (port 5005)

2. **Luciddream container** - New container that will:
   - Run `dream_api_server.py`: FastAPI server for dream processing (port 8080)
   - Connect to the tensor and HPC servers via WebSockets
   - Provide HTTP API endpoints for dream processing (port 8081)

3. **Docker network (lucid-net)** - Enables communication between containers

## Prerequisites

- Docker Engine installed
- Docker Compose installed
- The main `nemo_sig_v3` container is already running
- NVIDIA GPU with appropriate drivers (for the main container)

## Setup Instructions

### Option 1: Using Docker Compose

1. Make sure the main `nemo_sig_v3` container is running

2. Run the Luciddream container using Docker Compose:

```bash
docker-compose -f luciddream-docker-compose.yml up -d
```

### Option 2: Building and Running Manually

1. Build the Luciddream Docker image:

```bash
docker build -t luciddream:latest -f Dockerfile.luciddream .
```

2. Run the container, connecting it to the existing Docker network:

```bash
docker run -d \
  --name luciddream \
  --network lucid-net \
  -p 8080:8080 \
  -p 8081:8081 \
  -v ./:/workspace/project \
  -v ./data:/workspace/data \
  luciddream:latest
```

## Verifying the Setup

1. Check if the container is running:

```bash
docker ps
```

2. Check the logs to ensure proper initialization:

```bash
docker logs luciddream
```

3. Test the API health endpoint:

```bash
curl http://localhost:8080/api/dream/health
```

## API Endpoints

The Luciddream container exposes the following API endpoints:

- **POST /api/dream/start** - Start a dream processing session
- **GET /api/dream/status** - Get the status of dream processing
- **POST /api/dream/stop** - Stop the current dream session
- **POST /api/dream/consolidate** - Consolidate similar memories
- **POST /api/dream/optimize** - Optimize memory storage
- **POST /api/dream/insights** - Generate insights from memories
- **POST /api/dream/self-reflection** - Run a self-reflection session
- **GET /api/dream/self-model** - Retrieve data from the self model
- **GET /api/dream/knowledge-graph** - Get knowledge graph information
- **POST /api/dream/knowledge-graph** - Add to the knowledge graph
- **GET /api/dream/health** - Health check endpoint

## Environment Variables

The following environment variables can be configured:

- `TENSOR_SERVER_URL` - URL of the tensor server (default: `ws://nemo_sig_v3:5001`)
- `HPC_SERVER_URL` - URL of the HPC server (default: `ws://nemo_sig_v3:5005`)
- `DREAM_API_PORT` - Port for the dream API (default: `8080`)
- `LOG_LEVEL` - Logging level (default: `INFO`)
- `STORAGE_PATH` - Path for storing data (default: `/workspace/data`)
- `LLM_API_ENDPOINT` - Endpoint for the LLM API (default: `http://host.docker.internal:1234/v1`)
- `LLM_MODEL` - Model to use for LLM (default: `qwen_qwq-32b`)

## Troubleshooting

### Connection Issues

If the Luciddream container cannot connect to the tensor or HPC servers:

1. Ensure both servers are running in the `nemo_sig_v3` container
2. Check that both containers are on the same Docker network (`lucid-net`)
3. Verify the WebSocket URLs are correct in the environment variables

### API Errors

If you encounter errors when calling the API endpoints:

1. Check the container logs for error messages
2. Ensure the required components (knowledge graph, self model, world model) are properly initialized
3. Verify that the LLM service is accessible

## Important Notes

- The Luciddream container depends on the `nemo_sig_v3` container for embedding generation and HPC processing
- The container mounts the project directory as a volume, so code changes will be reflected without rebuilding
- Data is persisted in the `/workspace/data` directory, which is mounted as a volume
