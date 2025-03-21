@echo off
echo Running LTM converter with automatic emotion data fixing...

docker run --gpus all --rm -v .:/app -w /app --add-host=host.docker.internal:host-gateway lucid-recall-dist-hpcqr-stress-test python tools/run_ltm_with_emotion_fix.py --ltm-path memory/stored/ltm

echo Done!
