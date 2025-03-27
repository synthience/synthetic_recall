@echo off
echo ===== Running Quick FAISS Memory Test in Docker =====

REM Run the quick test within the Docker container's network
docker exec -it synthians_core bash -c "cd /workspace/project && python -c \"import sys, asyncio; from quick_test import test_memory_system; sys.exit(0 if asyncio.run(test_memory_system()) else 1)\""

if %ERRORLEVEL% EQU 0 (
  echo.
  echo [92m^u2713 FAISS Memory System Test: PASSED[0m
  echo ===== Test successfully completed =====
) else (
  echo.
  echo [91m^u2717 FAISS Memory System Test: FAILED[0m
  echo ===== Test failed =====
)

pause
