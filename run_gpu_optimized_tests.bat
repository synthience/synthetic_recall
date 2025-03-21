@echo off
SETLOCAL EnableDelayedExpansion

echo HPC-QR Flow Manager GPU Optimization and Testing Suite
echo ===================================================
echo.

REM Check if NVIDIA GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || (
    echo ERROR: PyTorch with CUDA support is required.
    exit /b 1
)

REM Create results directory if it doesn't exist
mkdir test_results 2>NUL

echo Checking GPU information...
python tools/gpu_resource_manager.py info --detailed

echo.
echo Choose an operation:
echo 1. Run damping factor analysis
echo 2. Run batch size optimization tests
echo 3. Run GPU-optimized tensor server
echo 4. Monitor GPU usage
echo 5. Run all tests in Docker container
echo.

set /p operation=Enter your choice (1-5): 

if "%operation%"=="1" (
    echo.
    echo Running damping factor analysis...
    python tools/damping_factor_analysis.py --embedding-count 1000 --scaling-factors 0.1 0.5 1.0
    
    echo.
    echo Results saved to test_results directory.
)

if "%operation%"=="2" (
    echo.
    echo Running batch size optimization tests...
    python tools/hpcqr_stress_test.py --batch-sizes 1 8 16 32 64 128 256 --embedding-counts 100 500 1000
    
    echo.
    echo Results saved to test_results directory.
)

if "%operation%"=="3" (
    echo.
    echo Starting GPU-optimized tensor server...
    python tools/gpu_optimized_tensor_server.py
)

if "%operation%"=="4" (
    echo.
    echo Monitoring GPU usage...
    set /p duration=Enter monitoring duration in seconds (default: 60): 
    
    if "!duration!"=="" set duration=60
    
    python tools/gpu_resource_manager.py monitor --interval 1 --duration !duration! --output test_results/gpu_monitoring.json
    
    echo.
    echo Monitoring data saved to test_results/gpu_monitoring.json
)

if "%operation%"=="5" (
    echo.
    echo Running all tests in Docker container...
    
    echo Building Docker image...
    docker-compose -f docker-compose.hpcqr-test.yml build
    
    echo.
    echo Running stress tests...
    docker-compose -f docker-compose.hpcqr-test.yml run hpcqr-stress-test
    
    echo.
    echo Running damping factor analysis...
    docker-compose -f docker-compose.hpcqr-test.yml run damping-analysis
    
    echo.
    echo Tests completed. Results are in the test_results directory.
)

echo.
echo Done!
pause
