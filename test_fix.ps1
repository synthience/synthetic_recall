# Test script to verify memory system fix

Write-Host "\n[Step 1] Running Docker to test the memory system..." -ForegroundColor Cyan

cd "$PSScriptRoot"

# First run the docker_test.py script to see if our fix worked
Write-Host "\nRunning memory system test..." -ForegroundColor Yellow
python docker_test.py

# Check the exit code to see if the test passed
if ($LASTEXITCODE -eq 0) {
    Write-Host "\n✅ Tests PASSED! The fix resolved the memory retrieval issue." -ForegroundColor Green
} else {
    Write-Host "\n❌ Tests FAILED. Further investigation needed." -ForegroundColor Red
}

Write-Host "\nTest complete." -ForegroundColor Cyan
