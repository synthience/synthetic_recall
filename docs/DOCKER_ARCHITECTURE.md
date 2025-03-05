# Lucid Recall Docker Architecture

## Container Structure

### 1. Core Processing Container (nemo_sig_v3)
- **Image**: lucid-recall-core
- **Base**: NVIDIA Container with CUDA support
- **Entrypoint**: /opt/nvidia/nvidia_entrypoint.sh
- **Mounts**:
  - Project: `G:\Development FAST\Lucid Recall FAST 1.2:/workspace/project`
  - Models: `H:\Models:/workspace/models`
- **Exposed Ports**:
  - 6006 (TensorBoard)
  - 8888 (Jupyter)
  - 5000 (Tensor Server)
  - 5004 (HPC Server)
- **Network Mode**: bridge
- **GPU Support**: Enabled via NVIDIA runtime

### 2. Port Forward Container (port-forward-5000)
- **Image**: alpine/socat
- **Purpose**: WebSocket port forwarding for Tensor Server
- **Command**: `socat TCP-LISTEN:5000,fork TCP:target:5000`
- **Port Mapping**: 5000:5000
- **Links**: Connected to nemo_sig_v3 as "target"
- **Network Mode**: bridge
- **Restart Policy**: no

### 3. Port Forward Container (port-forward-5004)
- **Image**: alpine/socat
- **Purpose**: WebSocket port forwarding for HPC Server
- **Command**: `socat TCP-LISTEN:5004,fork TCP:target:5004`
- **Port Mapping**: 5004:5004
- **Links**: Connected to nemo_sig_v3 as "target"
- **Network Mode**: bridge
- **Restart Policy**: no

## Network Configuration

### Bridge Network
- All containers on docker0 bridge network
- Internal IPs:
  - nemo_sig_v3: 172.17.0.2
  - port-forward-5000: 172.17.0.3
  - port-forward-5004: 172.17.0.4

### Port Forwarding
```
Client -> port-forward-5000:5000 -> nemo_sig_v3:5000 (Tensor Server)
Client -> port-forward-5004:5004 -> nemo_sig_v3:5004 (HPC Server)
```

## Volume Mounts

### Project Mount
```
Host: G:\Development FAST\Lucid Recall FAST 1.2
Container: /workspace/project
Purpose: Source code and runtime files
```

### Models Mount
```
Host: H:\Models
Container: /workspace/models
Purpose: ML models and weights
```

## Resource Configuration

### GPU Access
- NVIDIA runtime enabled
- Direct GPU access in nemo_sig_v3
- No GPU access needed for port forwards

### Memory Configuration
- nemo_sig_v3: No limit (runtime dependent)
- port-forward-5000: Default limits
- port-forward-5004: Default limits

## Security Configuration

### Container Isolation
- All containers run non-privileged
- Default seccomp profile
- No extra capabilities
- Standard AppArmor profile

### Network Security
- Internal bridge network
- Port forwarding via socat
- No direct container exposure
- Host port binding for WebSocket access

## Startup Dependencies

### Container Start Order
1. nemo_sig_v3 must start first
2. port-forward-5000 and port-forward-5004 start after
3. Links ensure proper networking

### Health Checks
- Port availability on 5000 and 5004
- WebSocket server status
- GPU availability in nemo_sig_v3

## Maintenance Procedures

### Container Updates
```bash
# Update core container
docker pull lucid-recall-core:latest
docker stop nemo_sig_v3
docker rm nemo_sig_v3
# Start new container with same config

# Update port forwards
docker pull alpine/socat:latest
docker stop port-forward-5000 port-forward-5004
docker rm port-forward-5000 port-forward-5004
# Start new containers with same config
```

### Log Management
- All containers use json-file logging
- Default log rotation
- Logs accessible via docker logs
- Container logs in standard docker location

## Development Notes

### Container Access
```bash
# Access core container
docker exec -it nemo_sig_v3 bash

# Check port forward logs
docker logs port-forward-5000
docker logs port-forward-5004
```

### Port Management
```bash
# Verify port forwarding
docker exec port-forward-5000 netstat -tulpn
docker exec port-forward-5004 netstat -tulpn

# Check core container ports
docker exec nemo_sig_v3 netstat -tulpn
```

### Resource Monitoring
```bash
# Monitor GPU usage
docker exec nemo_sig_v3 nvidia-smi

# Check container stats
docker stats nemo_sig_v3 port-forward-5000 port-forward-5004