@echo off
echo Testing Emotion Analyzer Integration
cd %~dp0..

echo.
echo Choose a test to run:
echo 1. Test EmotionMixin integration (clean output)
echo 2. Test EmotionMixin integration (verbose output)
echo 3. Test with correct format and delay
echo 4. Test manually
echo 5. Test with HPC fallback
echo.

set /p choice=Enter your choice (1-5): 

if "%choice%"=="1" (
    python utils\test_emotion_analyzer.py --integration-clean
) else if "%choice%"=="2" (
    python utils\test_emotion_analyzer.py --integration
) else if "%choice%"=="3" (
    python utils\test_emotion_analyzer.py --with-delay
) else if "%choice%"=="4" (
    python utils\test_emotion_analyzer.py --manual
) else if "%choice%"=="5" (
    python utils\test_emotion_analyzer.py --no-analyzer
) else (
    echo Invalid choice
    exit /b 1
)
