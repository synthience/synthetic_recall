# Lucid Recall Launch Guide

## Pre-Launch Verification

### 1. Container Status
```powershell
# Verify container is running
$container = docker ps | findstr nemo_sig_v2
if (-not $container) {
    Write-Host 'ERROR: Container nemo_sig_v2 not found running' -ForegroundColor Red
    exit 1
}

# Expected Output:
# [container_id] nemo_sig_v2 ... (running)
```

### 2. Port Availability
```powershell
# Check required ports
$ports = @(5000, 5004, 3000)
foreach ($port in $ports) {
    $test = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue
    if ($test.TcpTestSucceeded) {
        Write-Host "ERROR: Port $port is already in use" -ForegroundColor Red
        exit 1
    }
}
```

### 3. Resource Verification
- GPU Memory Available
- System Memory Free
- Required Processes Stopped
- Network Ports Clear

## Launch Sequence

### 1. Using start-LucidRecall-core.ps1
```powershell
# Execute the launch script
./start-LucidRecall-core.ps1

# Script will:
# 1. Verify container status
# 2. Check port availability
# 3. Launch Memory System
# 4. Launch HPC Server
# 5. Start Next.js interface
```

### 2. Manual Launch Sequence

#### Memory System Launch (Port 5000)
```powershell
# Launch tensor_server.py in container
Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `"docker exec -it nemo_sig_v2 sh -c 'cd /workspace/project && PYTHONPATH=/workspace/project python3 managers/python/nemo_integration/tensor_server.py'`"" -WindowStyle Normal

# Success Indicators:
# - "WebSocket server listening on 0.0.0.0:5000"
# - "GPU initialized: [GPU NAME]"
# - "Initialized TensorServer with Memory System"
```

#### HPC Server Launch (Port 5004)
```powershell
# Launch hpc_server.py in container
Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `"docker exec -it nemo_sig_v2 sh -c 'cd /workspace/project && PYTHONPATH=/workspace/project python3 managers/python/nemo_integration/hpc_server.py'`"" -WindowStyle Normal

# Success Indicators:
# - "HPC Server running on ws://0.0.0.0:5004"
# - "HPC Server initialized"
# - "Starting client connection handler"
```

#### Interface Launch (Port 3000)
```powershell
# Start Next.js interface
Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `"npm run dev`"" -WindowStyle Normal

# Success Indicators:
# - "ready - started server on 0.0.0.0:3000"
# - Interface accessible in browser
```

## Success Verification

### 1. Service Status Check
```powershell
# Memory System (Port 5000)
Test-NetConnection -ComputerName localhost -Port 5000

# HPC Server (Port 5004)
Test-NetConnection -ComputerName localhost -Port 5004

# Interface (Port 3000)
Test-NetConnection -ComputerName localhost -Port 3000
```

### 2. Component Verification
- Memory System: ws://localhost:5000
  * WebSocket connection active
  * GPU initialization complete
  * Memory operations responsive

- HPC Server: ws://localhost:5004
  * WebSocket connection active
  * Client handler initialized
  * Processing pipeline ready

- Next.js Interface: http://localhost:3000
  * Web interface accessible
  * WebSocket connections established
  * Components loading correctly

### 3. Log Verification
```powershell
# Memory System Logs
docker exec nemo_sig_v2 tail -f /workspace/project/logs/memory_system.log

# HPC Server Logs
docker exec nemo_sig_v2 tail -f /workspace/project/logs/hpc_server.log

# Interface Logs
# Available in the terminal running npm run dev
```

## Troubleshooting

### 1. Container Issues
```powershell
# Check container status
docker ps | findstr nemo_sig_v2

# Container logs
docker logs nemo_sig_v2

# Restart container if needed
docker restart nemo_sig_v2
```

### 2. Port Conflicts
```powershell
# List processes using ports
netstat -ano | findstr "5000 5004 3000"

# Kill process by PID if needed
Stop-Process -Id [PID] -Force
```

### 3. Service Recovery
- Memory System:
  * Check GPU availability
  * Verify container access
  * Restart tensor_server.py

- HPC Server:
  * Verify Memory System connection
  * Check processing pipeline
  * Restart hpc_server.py

- Interface:
  * Clear Next.js cache if needed
  * Verify WebSocket connections
  * Restart npm run dev

## Shutdown Procedure

### 1. Graceful Shutdown
```powershell
# Stop services in reverse order:
# 1. Close Next.js interface (Ctrl+C in interface window)
# 2. Stop HPC Server (Ctrl+C in HPC window)
# 3. Stop Memory System (Ctrl+C in Memory System window)
```

### 2. Verification
```powershell
# Verify ports are released
foreach ($port in @(5000, 5004, 3000)) {
    $test = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue
    if (-not $test.TcpTestSucceeded) {
        Write-Host "Port $port successfully released"
    }
}
```

### 3. Resource Cleanup
```powershell
# Clear GPU memory
docker exec nemo_sig_v2 python3 -c "import torch; torch.cuda.empty_cache()"

# Remove temporary files
Remove-Item -Path ".next/cache/*" -Recurse -Force