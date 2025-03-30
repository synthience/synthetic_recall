#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Parse command line arguments
FAST_MODE=false
for arg in "$@"; do
  case $arg in
    --fast)
      FAST_MODE=true
      echo "⚡ Fast mode enabled: Skipping optional delays and non-critical operations"
      shift
      ;;
    *)
      # Unknown option
      ;;
  esac
done

# Configuration
COMPOSE_FILE="docker-compose.test.yml"
TEST_DIR="tests" # Directory containing pytest tests
VARIANTS=("NONE" "MAC" "MAG" "MAL")
# Define URLs based on network_mode: host
MC_HEALTH_URL="http://localhost:5010/health"
NM_HEALTH_URL="http://localhost:8001/health"
CCE_HEALTH_URL="http://localhost:8002/" # Add /health if CCE gets one
HEALTH_TIMEOUT=120 # Max seconds to wait for services to be healthy
HEALTH_INTERVAL=5  # Seconds between health checks

# Optimize settings for fast mode
if [ "$FAST_MODE" = true ]; then
  HEALTH_INTERVAL=2  # Faster checking in fast mode
  # We keep HEALTH_TIMEOUT the same to avoid false failures
fi

# --- Helper Function ---
wait_for_service() {
  local url=$1
  local service_name=$2
  local elapsed=0
  echo "Waiting for $service_name at $url to be healthy..."
  while ! curl -sf "$url" > /dev/null; do
    if [ $elapsed -ge $HEALTH_TIMEOUT ]; then
      echo "Error: Timeout waiting for $service_name at $url"
      docker-compose -f "$COMPOSE_FILE" logs "$service_name" # Show logs on failure
      exit 1
    fi
    echo "($elapsed s) - $service_name not ready yet, sleeping $HEALTH_INTERVAL s..."
    sleep $HEALTH_INTERVAL
    elapsed=$((elapsed + HEALTH_INTERVAL))
  done
  echo "$service_name is healthy!"
}

# --- Main Test Execution ---
overall_status=0

cleanup() {
  echo "--- Cleaning up Docker environment ---"
  export TITANS_VARIANT="" # Unset for safety
  
  if [ "$FAST_MODE" = true ]; then
    # Fast mode: no volume removal for speed, just stop containers
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    echo "Docker environment stopped (fast mode - volumes preserved)."
  else
    # Normal mode: full cleanup including volumes
    docker-compose -f "$COMPOSE_FILE" down -v --remove-orphans
    echo "Docker environment stopped and volumes removed."
  fi
}
# Ensure cleanup runs on exit or interrupt
trap cleanup EXIT INT TERM

# Create test data directories if they don't exist
mkdir -p ./test_data/mc_storage ./test_data/nm_logs ./test_data/cce_logs

for variant in "${VARIANTS[@]}"; do
  echo ""
  echo "======================================================"
  echo " Preparing to test TITANS_VARIANT=$variant"
  echo "======================================================"

  # Set the variant for the CCE container environment
  export TITANS_VARIANT=$variant
  echo "Exported TITANS_VARIANT=$TITANS_VARIANT for docker-compose"

  # Bring up services for this variant test run
  echo "--- Starting services for $variant test ---"
  # Use --force-recreate to ensure CCE picks up the new env var
  
  if [ "$FAST_MODE" = true ] && [ "$variant" != "${VARIANTS[0]}" ]; then
    # Fast mode & not first variant: reuse containers where possible for speed
    docker-compose -f "$COMPOSE_FILE" up -d --no-recreate
    # Just restart the CCE container to pick up new variant
    docker-compose -f "$COMPOSE_FILE" restart context-cascade-engine
    echo "Services starting in fast mode (reusing containers where possible)..."
  else
    # First variant or normal mode: ensure clean environment
    docker-compose -f "$COMPOSE_FILE" up -d --force-recreate --remove-orphans # --build
    echo "Services starting..."
  fi

  # Wait for all services to be healthy
  wait_for_service "$MC_HEALTH_URL" "Memory Core"
  wait_for_service "$NM_HEALTH_URL" "Neural Memory"
  wait_for_service "$CCE_HEALTH_URL" "CCE" # Adjust endpoint if needed
  echo "All services healthy for $variant test."

  # Run the specific test file for this variant
  TEST_FILE="${TEST_DIR}/test_variant_${variant,,}.py" # Lowercase variant name
  if [ ! -f "$TEST_FILE" ]; then
      # Handle NONE variant if no specific test file exists
      if [ "$variant" == "NONE" ]; then
          echo "Skipping specific test file for NONE variant (assuming covered by others or basic tests)."
          # Optional: Run a general integration test here if desired
          # pytest "${TEST_DIR}/test_basic_integration.py" -v -s
          pytest_exit_code=0
      else
          echo "Warning: Test file $TEST_FILE not found. Skipping tests for $variant."
          pytest_exit_code=0 # Don't fail the whole run if a test file is missing yet
      fi
  else
      echo "--- Running pytest for $variant ($TEST_FILE) ---"
      pytest "$TEST_FILE" -v -s --asyncio-mode=auto
      pytest_exit_code=$?
  fi

  if [ $pytest_exit_code -ne 0 ]; then
    echo "Error: Pytest failed for variant $variant with exit code $pytest_exit_code"
    overall_status=1 # Mark the overall run as failed
    # Optional: Stop further tests on first failure? Add 'exit 1' here.
  else
    echo "Pytest successful for variant $variant."
  fi

  # Bring down services between variants (unless in fast mode and not the last variant)
  if [ "$FAST_MODE" = true ] && [ "$variant" != "${VARIANTS[-1]}" ]; then
    # Fast mode & not last variant: just unset the environment variable
    # but keep containers running for the next test
    echo "--- Fast mode: Keeping services running for next variant ---"
    export TITANS_VARIANT=""
  else
    # Last variant or normal mode: full cleanup
    echo "--- Stopping services after $variant test ---"
    if [ "$FAST_MODE" = true ]; then
      # Fast mode, last variant: no volume removal
      docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    else
      # Normal mode: full cleanup
      docker-compose -f "$COMPOSE_FILE" down -v --remove-orphans
    fi
    echo "Services stopped."
    export TITANS_VARIANT=""
  fi

  # Optional short pause between variant runs (skip in fast mode)
  if [ "$FAST_MODE" != true ]; then
    sleep 2
  fi
done

echo ""
echo "======================================================"
if [ $overall_status -eq 0 ]; then
  echo "✅ All variant tests completed successfully."
else
  echo "❌ Some variant tests failed."
fi
echo "======================================================"

exit $overall_status
