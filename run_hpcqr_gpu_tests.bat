@echo off
echo Running HPC-QR Flow Manager GPU-accelerated tests...

REM Create results directory if it doesn't exist
mkdir test_results 2>NUL

echo Building Docker image...
docker-compose -f docker-compose.hpcqr-test.yml build

echo.
echo Running stress tests to identify optimal batch sizes...
docker-compose -f docker-compose.hpcqr-test.yml run hpcqr-stress-test

echo.
echo Running damping factor analysis...
docker-compose -f docker-compose.hpcqr-test.yml run damping-analysis

echo.
echo Tests completed. Results are in the test_results directory.
echo Use 'docker-compose -f docker-compose.hpcqr-test.yml up tensor-server' to start the tensor server with GPU acceleration.

echo.
pause
