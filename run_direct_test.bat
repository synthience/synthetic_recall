@echo off
echo ===== Running Direct FAISS Memory Tests in Docker =====

REM Run the direct test in the Docker container
docker exec -it synthians_core bash -c "cd /workspace/project && python docker_test.py"

if %ERRORLEVEL% EQU 0 (
  echo.
  echo [92m✓ FAISS Memory Integration Tests: PASSED[0m
  echo ===== Test successfully completed =====
) else (
  echo.
  echo [91m✗ FAISS Memory Integration Tests: FAILED[0m
  echo ===== Test failed =====
)

pause
