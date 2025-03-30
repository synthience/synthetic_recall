# Stop services if running
Write-Output "Stopping Docker containers if running..."
docker-compose down

# Make the entrypoint script executable
Write-Output "Fixing script permissions..."
docker run --rm -v ".:/workspace" -w /workspace alpine sh -c "chmod +x faiss_setup_entrypoint.sh"

# Start the containers
Write-Output "Starting Docker containers with our changes..."
docker-compose up -d

# Wait for the containers to be fully started
Write-Output "Waiting for containers to start (15 seconds)..."
Start-Sleep -Seconds 15

# Check if the API server is running
Write-Output "Checking service status..."
$MEMORY_CORE_ID = docker ps --filter "name=synthians_core" --format "{{.ID}}"
$TRAINER_SERVER_ID = docker ps --filter "name=trainer-server" --format "{{.ID}}"

if (-not $MEMORY_CORE_ID) {
    Write-Output "u274c Container synthians_core is not running. Check docker-compose logs."
} else {
    Write-Output "u2705 Container synthians_core is running."
}

if (-not $TRAINER_SERVER_ID) {
    Write-Output "u274c Container trainer-server is not running. Check docker-compose logs."
} else {
    Write-Output "u2705 Container trainer-server is running."
}

# Get logs to see if initialization completed
Write-Output "\nMost recent synthians_core logs:"
docker logs synthians_core --tail 20

Write-Output "\nMost recent trainer-server logs:"
docker logs trainer-server --tail 20

# Run the docker_test.py script if both containers are running
if ($MEMORY_CORE_ID -and $TRAINER_SERVER_ID) {
    Write-Output "\nRunning memory system test..."
    docker exec $MEMORY_CORE_ID python /workspace/project/docker_test.py
} else {
    Write-Output "\nSkipping test since at least one container is not running."
}

Write-Output "\nFor more detailed logs, run: docker logs synthians_core or docker logs trainer-server"
