# synthians-dashboard-tester.ps1
# Purpose: Automate testing for Synthians Cognitive Dashboard Phase 5.9.2

# Configuration Variables
$composeFile = "..\..\docker-compose.yml" # Path relative to the dashboard directory
$dashboardPath = "c:\Users\danny\OneDrive\Documents\AI_Conversations\lucid-recall-dist\lucid-recall-dist\synthians_memory_core\Synthians_dashboard"
$dashboardUrl = "http://localhost:5000"
$apiBaseUrl = "http://localhost:5020" # Memory Core port
$neuralMemoryUrl = "http://localhost:8001" # Neural Memory port
$cceUrl = "http://localhost:8002" # CCE port

# Define checkmark and X mark that work in PowerShell
$checkMark = "[OK]"
$xMark = "[FAIL]"

Write-Host "=== Synthians Cognitive Dashboard Phase 5.9.2 Testing Tool ===" -ForegroundColor Cyan
Write-Host "This script will help automate portions of the testing process." -ForegroundColor Cyan

function Test-ServiceHealth {
    param ([string]$explainabilityFlag)
    
    Write-Host "`n[STEP 1] Checking Docker Service Health..." -ForegroundColor Magenta
    
    # Define services with their container names and ports
    $services = @(
        @{name="synthians_core"; containerName="synthians_core"; port="5020:5010"; alias="memory-core"; k8sLabel="app=synthians_core"; k8sPort=5010},
        @{name="trainer-server"; containerName="trainer-server"; port="8001:8001"; alias="neural-memory"; k8sLabel="app=trainer-server"; k8sPort=8001},
        @{name="context-cascade-orchestrator"; containerName="context-cascade-orchestrator"; port="8002:8002"; alias="cce"; k8sLabel="app=context-cascade-orchestrator"; k8sPort=8002}
    )
    $allRunning = $true
    
    # Try both kubectl and docker commands to check environment
    $k8sRunning = $false
    $dockerRunning = $false
    
    try {
        $k8sOutput = kubectl get pods 2>$null
        if ($null -ne $k8sOutput) {
            $k8sRunning = $true
            Write-Host "Kubernetes environment detected" -ForegroundColor Green
        }
    } catch {
        $k8sRunning = $false
    }
    
    try {
        $dockerOutput = docker ps 2>$null
        if ($null -ne $dockerOutput) {
            $dockerRunning = $true
        }
    } catch {
        $dockerRunning = $false
    }
    
    if (-not $k8sRunning -and -not $dockerRunning) {
        Write-Host "Neither Kubernetes nor Docker appears to be running or accessible" -ForegroundColor Red
        Write-Host "Please ensure either Docker Desktop or kubectl is properly configured" -ForegroundColor Red
        return $false
    }
    
    Write-Host "Detecting services..." -ForegroundColor Yellow
    
    foreach ($service in $services) {
        $serviceName = $service.name
        $containerName = $service.containerName
        $alias = $service.alias
        $serviceDetected = $false
        
        # First try Kubernetes
        if ($k8sRunning) {
            Write-Host "Checking for $serviceName in Kubernetes..." -ForegroundColor Gray
            $k8sStatus = kubectl get pods | Select-String $serviceName
            if ($null -ne $k8sStatus) {
                $isRunning = $k8sStatus -match "Running"
                $serviceDetected = $true
            }
        }
        
        # If not found in K8s or K8s not running, try Docker
        if (-not $serviceDetected -and $dockerRunning) {
            Write-Host "Checking for $containerName in Docker..." -ForegroundColor Gray
            $dockerStatus = docker ps --filter "name=$containerName" --format "{{.Status}}"
            if ($null -ne $dockerStatus -and $dockerStatus -ne "") {
                $isRunning = $dockerStatus -match "Up"
                $serviceDetected = $true
            }
        }
        
        # Process results
        if ($serviceDetected) {
            if ($isRunning) {
                Write-Host "$checkMark $serviceName ($alias) is running" -ForegroundColor Green
                
                # Special check for memory core explainability flag
                if ($serviceName -eq "synthians_core" -and $explainabilityFlag -ne "unknown") {
                    $explainValue = ""
                    if ($k8sRunning) {
                        $podName = kubectl get pods | Select-String $serviceName | ForEach-Object { $_ -replace '\s+', ' ' } | ForEach-Object { ($_ -split ' ')[0] }
                        if ($null -ne $podName) {
                            Write-Host "  Checking logs in pod $podName" -ForegroundColor Gray
                            $explainValue = kubectl logs --tail 100 $podName 2>$null | Select-String "ENABLE_EXPLAINABILITY"
                        }
                    } else {
                        Write-Host "  Checking logs in container $containerName" -ForegroundColor Gray
                        $explainValue = docker logs --tail 100 $containerName 2>$null | Select-String "ENABLE_EXPLAINABILITY"
                    }
                    
                    Write-Host "  ENABLE_EXPLAINABILITY setting: $explainValue" -ForegroundColor Yellow
                    
                    if (($explainabilityFlag -eq "true" -and $explainValue -notmatch "true") -or 
                        ($explainabilityFlag -eq "false" -and $explainValue -notmatch "false")) {
                        Write-Host "  WARNING: ENABLE_EXPLAINABILITY might not be set to $explainabilityFlag" -ForegroundColor Red
                    }
                }
            } else {
                Write-Host "$xMark $serviceName ($alias) is detected but NOT running" -ForegroundColor Red
                $allRunning = $false
            }
        } else {
            Write-Host "$xMark $serviceName ($alias) was not found" -ForegroundColor Red
            $allRunning = $false
        }
    }
    
    if (-not $allRunning) {
        if ($k8sRunning) {
            Write-Host "`nSome services are not running in Kubernetes. Check their status with:" -ForegroundColor Yellow
            Write-Host "kubectl get pods" -ForegroundColor Yellow
        } else {
            Write-Host "`nSome services are not running. Start them with:" -ForegroundColor Yellow
            Write-Host "docker-compose -f $composeFile up -d" -ForegroundColor Yellow
        }
        
        $proceed = Read-Host "Would you like to continue anyway? (y/n)"
        if ($proceed -ne "y") {
            return $false
        }
    }
    
    return $true
}

function Test-DashboardRunning {
    Write-Host "`n[STEP 2] Checking Dashboard Status..." -ForegroundColor Magenta
    
    try {
        $response = Invoke-WebRequest -Uri $dashboardUrl -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "$checkMark Dashboard is accessible at $dashboardUrl" -ForegroundColor Green
            return $true
        }
    } catch {
        Write-Host "$xMark Dashboard is not accessible at $dashboardUrl" -ForegroundColor Red
        
        # Check if dashboard container is running
        $dashboardRunning = $false
        try {
            $k8sDashboard = kubectl get pods | Select-String "dashboard"
            if ($null -ne $k8sDashboard -and $k8sDashboard -match "Running") {
                $dashboardRunning = $true
                Write-Host "Dashboard pod appears to be running but web UI is not accessible" -ForegroundColor Yellow
            }
        } catch { }
        
        try {
            $dockerDashboard = docker ps --filter "name=synthians-dashboard" --format "{{.Status}}"
            if ($null -ne $dockerDashboard -and $dockerDashboard -match "Up") {
                $dashboardRunning = $true
                Write-Host "Dashboard container appears to be running but web UI is not accessible" -ForegroundColor Yellow
            }
        } catch { }
        
        if (-not $dashboardRunning) {
            $start = Read-Host "Would you like to start the dashboard locally? (y/n)"
            if ($start -eq "y") {
                Write-Host "Starting dashboard at $dashboardPath..." -ForegroundColor Yellow
                
                # Set environment variables and start the dashboard
                $startCmd = "cd $dashboardPath; `$env:MEMORY_CORE_URL='$apiBaseUrl'; `$env:NEURAL_MEMORY_URL='$neuralMemoryUrl'; `$env:CCE_URL='$cceUrl'; npm run dev"
                Start-Process powershell -ArgumentList "-NoExit -Command $startCmd"
                
                # Wait a bit for it to start
                Write-Host "Waiting 10 seconds for dashboard to start..." -ForegroundColor Yellow
                Start-Sleep -Seconds 10
                
                # Check again
                return Test-DashboardRunning
            }
        } else {
            Write-Host "Dashboard service is running but the web UI is not accessible. Check for errors in the container logs." -ForegroundColor Yellow
        }
    }
    
    return $false
}

function New-TestData {
    Write-Host "`n[STEP 3] Creating Test Data..." -ForegroundColor Magenta
    
    # Skip if user doesn't want to create test data
    $createData = Read-Host "Would you like to create test memories and assemblies? (y/n)"
    if ($createData -ne "y") {
        Write-Host "Skipping test data creation..." -ForegroundColor Yellow
        return
    }
    
    # Headers for API calls
    $headers = @{
        "Content-Type" = "application/json"
    }
    
    try {
        # Create test memories
        Write-Host "Creating test memories..." -ForegroundColor Yellow
        
        $memory1 = Invoke-RestMethod -Uri "$apiBaseUrl/api/memories" -Method Post -Headers $headers `
                    -Body '{"content": "This is test memory 1 for Phase 5.9.2 testing.", "tags": ["test", "phase5.9.2"]}' `
                    -ErrorAction Stop
                    
        $memory2 = Invoke-RestMethod -Uri "$apiBaseUrl/api/memories" -Method Post -Headers $headers `
                    -Body '{"content": "This is test memory 2 for Phase 5.9.2 testing.", "tags": ["test", "phase5.9.2"]}' `
                    -ErrorAction Stop
        
        # Create an assembly with these memories
        Write-Host "Creating test assembly..." -ForegroundColor Yellow
        $assembly = Invoke-RestMethod -Uri "$apiBaseUrl/api/assemblies" -Method Post -Headers $headers `
                    -Body "{`"name`":`"Test Assembly 5.9.2`", `"description`":`"Assembly for dashboard testing`", `"memory_ids`":[`"$($memory1.id)`",`"$($memory2.id)`"], `"tags`":[`"test`", `"dashboard`"]}" `
                    -ErrorAction Stop
        
        Write-Host "$checkMark Successfully created test data:" -ForegroundColor Green
        Write-Host "  Memory 1 ID: $($memory1.id)" -ForegroundColor Cyan
        Write-Host "  Memory 2 ID: $($memory2.id)" -ForegroundColor Cyan
        Write-Host "  Assembly ID: $($assembly.id)" -ForegroundColor Cyan
        
        # Attempt to create a second assembly for merge testing
        Write-Host "Creating second test assembly..." -ForegroundColor Yellow
        $memory3 = Invoke-RestMethod -Uri "$apiBaseUrl/api/memories" -Method Post -Headers $headers `
                    -Body '{"content": "This is test memory 3 for merge testing.", "tags": ["test", "merge"]}' `
                    -ErrorAction Stop
                    
        $assembly2 = Invoke-RestMethod -Uri "$apiBaseUrl/api/assemblies" -Method Post -Headers $headers `
                    -Body "{`"name`":`"Merge Test Assembly`", `"description`":`"Assembly for merge testing`", `"memory_ids`":[`"$($memory3.id)`"], `"tags`":[`"test`", `"merge`"]}" `
                    -ErrorAction Stop
        
        Write-Host "  Second Assembly ID: $($assembly2.id)" -ForegroundColor Cyan
        
        # Attempt to trigger a merge
        $triggerMerge = Read-Host "Would you like to attempt to trigger a merge between the assemblies? (y/n)"
        if ($triggerMerge -eq "y") {
            Write-Host "Attempting to trigger merge..." -ForegroundColor Yellow
            try {
                $mergeBody = "{`"source_assembly_ids`":[`"$($assembly.id)`",`"$($assembly2.id)`"]}" 
                $merge = Invoke-RestMethod -Uri "$apiBaseUrl/api/request_merge" -Method Post -Headers $headers `
                        -Body $mergeBody `
                        -ErrorAction Stop
                Write-Host "$checkMark Merge requested successfully" -ForegroundColor Green
                Write-Host "  Merge request ID: $($merge.id)" -ForegroundColor Cyan
            } catch {
                Write-Host "$xMark Failed to trigger merge: $_" -ForegroundColor Red
            }
        }
        
    } catch {
        Write-Host "$xMark Failed to create test data: $_" -ForegroundColor Red
    }
}

function Set-ExplainabilityFlag {
    param ([string]$desiredState)
    
    Write-Host "`n[TOGGLING FLAG] Setting ENABLE_EXPLAINABILITY=$desiredState..." -ForegroundColor Magenta
    
    # Check if running in Kubernetes
    $k8sRunning = $false
    try {
        $k8sOutput = kubectl get pods 2>$null
        if ($null -ne $k8sOutput) {
            $k8sRunning = $true
        }
    } catch {}
    
    if ($k8sRunning) {
        Write-Host "Kubernetes environment detected. You'll need to modify the deployment or ConfigMap for memory-core." -ForegroundColor Yellow
        Write-Host "This is typically done with kubectl edit deployment memory-core or similar commands." -ForegroundColor Yellow
        Read-Host "Press Enter when you've made the change" | Out-Null
    } else {
        # Stop the services
        Write-Host "Stopping Docker services..." -ForegroundColor Yellow
        docker-compose -f $composeFile down
        
        # Need to modify the environment variable or .env file
        Write-Host "Please manually edit your docker-compose.yml or .env file to set ENABLE_EXPLAINABILITY=$desiredState for memory-core service" -ForegroundColor Yellow
        Read-Host "Press Enter when you've made the change" | Out-Null
        
        # Restart services
        Write-Host "Restarting services with new configuration..." -ForegroundColor Yellow
        docker-compose -f $composeFile up -d
    }
    
    # Wait for services to be ready
    Write-Host "Waiting for services to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    
    # Verify the setting took effect
    Test-ServiceHealth -explainabilityFlag $desiredState
}

function Open-TestUrls {
    Write-Host "`n[OPENING TEST URLS] Opening browser windows for test pages..." -ForegroundColor Magenta
    
    $urls = @(
        "$dashboardUrl/overview",
        "$dashboardUrl/memory-core",
        "$dashboardUrl/logs",
        "$dashboardUrl/assemblies",
        "$dashboardUrl/config"
    )
    
    foreach ($url in $urls) {
        Write-Host "Opening: $url" -ForegroundColor Yellow
        Start-Process $url
        Start-Sleep -Seconds 2 # Brief delay between opening windows
    }
    
    # If we have a test assembly ID from earlier, open it too
    $assemblyId = Read-Host "Enter a test assembly ID to open its detail page (or press Enter to skip)"
    if ($assemblyId) {
        $assemblyUrl = "$dashboardUrl/assemblies/$assemblyId"
        Write-Host "Opening: $assemblyUrl" -ForegroundColor Yellow
        Start-Process $assemblyUrl
    }
}

function Start-TestScenario {
    param ([string]$explainabilityFlag)
    
    # Check services
    $servicesOk = Test-ServiceHealth -explainabilityFlag $explainabilityFlag
    if (-not $servicesOk) { 
        Write-Host "Service health check failed or was skipped. Some tests may not work correctly." -ForegroundColor Yellow
    }
    
    # Check dashboard
    $dashboardOk = Test-DashboardRunning
    if (-not $dashboardOk) { 
        Write-Host "Dashboard is not running. Cannot proceed with UI testing." -ForegroundColor Red
        return 
    }
    
    # Optionally create test data
    New-TestData
    
    # Open test URLs for manual testing
    Open-TestUrls
    
    Write-Host "`n=== Manual Testing Required ===" -ForegroundColor White -BackgroundColor DarkBlue
    Write-Host "Please manually test the features according to the test plan." -ForegroundColor Cyan
    Write-Host "The browser windows have been opened to the relevant pages." -ForegroundColor Cyan
    Write-Host "Document your findings in the test checklist." -ForegroundColor Cyan
}

# Main Menu
function Show-Menu {
    Write-Host "`n=== Synthians Testing Menu ===" -ForegroundColor Green
    Write-Host "1. Run test scenario with ENABLE_EXPLAINABILITY=true" -ForegroundColor White
    Write-Host "2. Run test scenario with ENABLE_EXPLAINABILITY=false" -ForegroundColor White
    Write-Host "3. Toggle explainability flag" -ForegroundColor White
    Write-Host "4. Create test data only" -ForegroundColor White
    Write-Host "5. Check service health" -ForegroundColor White
    Write-Host "6. Open test URLs" -ForegroundColor White
    Write-Host "Q. Quit" -ForegroundColor White
    
    $choice = Read-Host "Enter your choice"
    
    switch ($choice) {
        "1" { Start-TestScenario -explainabilityFlag "true" }
        "2" { Start-TestScenario -explainabilityFlag "false" }
        "3" { 
            $newState = Read-Host "Enter desired state (true/false)"
            if ($newState -eq "true" -or $newState -eq "false") {
                Set-ExplainabilityFlag -desiredState $newState
            } else {
                Write-Host "Invalid input. Must be 'true' or 'false'." -ForegroundColor Red
            }
        }
        "4" { New-TestData }
        "5" { Test-ServiceHealth -explainabilityFlag "unknown" }
        "6" { Open-TestUrls }
        "Q" { return $false }
        "q" { return $false }
        default { Write-Host "Invalid option, please try again" -ForegroundColor Red }
    }
    
    return $true
}

# Main loop
$continue = $true
while ($continue) {
    $continue = Show-Menu
}

Write-Host "Testing completed. Thank you for using the Synthians Dashboard Testing Tool." -ForegroundColor Green