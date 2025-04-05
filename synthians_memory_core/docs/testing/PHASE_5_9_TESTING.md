# Phase 5.9 Testing Strategy (Revised)

This document outlines the testing strategy for the new explainability and diagnostics features planned for Phase 5.9, updated to reflect the revised implementation approach.

## Overview

Phase 5.9 introduces several new components and APIs related to explainability and diagnostics. Testing these features requires a combination of unit tests, integration tests, and end-to-end tests to ensure correctness, performance, and resilience.

## Key Components to Test

1. **Explainability Module**:
   - Activation explanation
   - Merge explanation (with revised merge tracking)
   - Lineage tracing (with cycle detection)

2. **Diagnostics Module**:
   - MergeTracker (append-only with event reconciliation)
   - Runtime configuration exposure (strict allow-list)
   - Activation statistics (periodic persistence)

3. **API Endpoints**:
   - `GET /assemblies/{id}/explain_activation?memory_id={memory_id}`
   - `GET /assemblies/{id}/explain_merge`
   - `GET /assemblies/{id}/lineage`
   - `GET /diagnostics/merge_log`
   - `GET /config/runtime/{service_name}`

4. **Feature Flag**:
   - `ENABLE_EXPLAINABILITY` flag behavior

## Unit Testing

### Explainability Module

```python
# Example test for activation explanation
async def test_activation_explanation():
    # Setup: Create a mock geometry manager and persistence
    geometry_manager = MockGeometryManager()
    persistence = MockMemoryPersistence()
    
    # Configure mock to return specific values
    geometry_manager.calculate_similarity.return_value = 0.85
    
    # Create the explainer function/module under test
    from synthians_memory_core.explainability.activation import generate_activation_explanation
    
    # Mock assembly and memory data
    assembly_id = "asm_test"
    memory_id = "mem_test"
    assembly_data = {
        "id": assembly_id,
        "composite_embedding": [0.1, 0.2, 0.3]
    }
    memory_data = {
        "id": memory_id,
        "embedding": [0.2, 0.3, 0.4]
    }
    
    # Mock the persistence to return our test data
    persistence.load_assembly.return_value = assembly_data
    persistence.load_memory.return_value = memory_data
    
    # Execute
    explanation = await generate_activation_explanation(
        assembly_id=assembly_id,
        memory_id=memory_id,
        persistence=persistence,
        geometry_manager=geometry_manager,
        config={"ASSEMBLY_ACTIVATION_THRESHOLD": 0.8}
    )
    
    # Assert
    assert explanation.assembly_id == assembly_id
    assert explanation.memory_id == memory_id
    assert explanation.calculated_similarity == 0.85
    assert explanation.activation_threshold == 0.8
    assert explanation.passed_threshold == True
    
    # Verify mocks were called correctly
    persistence.load_assembly.assert_called_once_with(assembly_id)
    persistence.load_memory.assert_called_once_with(memory_id)
    geometry_manager.calculate_similarity.assert_called_once_with(
        assembly_data["composite_embedding"], memory_data["embedding"]
    )
```

### Merge Tracker (Revised for Append-Only Strategy)

```python
# Example test for merge logging with the revised append-only strategy
async def test_merge_tracker_logging():
    # Setup: Create a merge tracker with a temporary log file
    temp_dir = tempfile.TemporaryDirectory()
    log_path = os.path.join(temp_dir.name, "merge_log.jsonl")
    
    merge_tracker = MergeTracker(log_path=log_path)
    
    # Execute: Log a merge event
    merge_event_id = await merge_tracker.log_merge_creation_event(
        source_assembly_ids=["asm_source1", "asm_source2"],
        target_assembly_id="asm_target",
        similarity=0.92,
        threshold=0.9
    )
    
    # Assert: Verify log file contains the expected merge creation entry
    async with aiofiles.open(log_path, "r") as f:
        content = await f.read()
        log_entries = [json.loads(line) for line in content.strip().split("\n")]
        
        assert len(log_entries) == 1
        entry = log_entries[0]
        assert entry["event_type"] == "merge_creation"
        assert len(entry["merge_event_id"]) > 0
        assert entry["source_assembly_ids"] == ["asm_source1", "asm_source2"]
        assert entry["target_assembly_id"] == "asm_target"
        assert entry["similarity_at_merge"] == 0.92
        assert entry["merge_threshold"] == 0.9
        # No explicit cleanup_status in merge_creation events
        
    # Now test the cleanup status update
    await merge_tracker.update_cleanup_status(merge_event_id, "completed")
    
    # Verify log now has both events
    async with aiofiles.open(log_path, "r") as f:
        content = await f.read()
        log_entries = [json.loads(line) for line in content.strip().split("\n")]
        
        assert len(log_entries) == 2
        cleanup_entry = log_entries[1]
        assert cleanup_entry["event_type"] == "cleanup_status_update"
        assert cleanup_entry["target_merge_event_id"] == merge_event_id
        assert cleanup_entry["new_status"] == "completed"
        
    # Test the reconciled event reading
    reconciled_entries = await merge_tracker.get_reconciled_log_entries(limit=10)
    assert len(reconciled_entries) == 1
    reconciled = reconciled_entries[0]
    assert reconciled["merge_event_id"] == merge_event_id
    assert reconciled["final_cleanup_status"] == "completed"
    
    # Cleanup
    temp_dir.cleanup()
```

### Configuration Service (Revised with Strict Allow-List)

```python
# Example test for configuration sanitization with strict allow-list
def test_config_sanitization():
    # Setup: Define the allow-list
    SAFE_CONFIG_KEYS_MEMORY_CORE = [
        "embedding_dim",
        "assembly_activation_threshold",
        "enable_explainability"
    ]
    
    # Create a configuration with both safe and unsafe keys
    full_config = {
        "embedding_dim": 768,
        "assembly_activation_threshold": 0.8,
        "database_password": "secret",  # Should be filtered out
        "api_key": "private_key",       # Should be filtered out
        "enable_explainability": True,
        "internal_cache_size": 1000     # Should be filtered out
    }
    
    # Execute
    from synthians_memory_core.api.diagnostics_routes import get_safe_config
    sanitized = get_safe_config(
        service_name="memory-core", 
        full_config=full_config,
        safe_keys_map={
            "memory-core": SAFE_CONFIG_KEYS_MEMORY_CORE
        }
    )
    
    # Assert: Verify only safe keys are included
    assert len(sanitized) == 3
    assert "embedding_dim" in sanitized
    assert "assembly_activation_threshold" in sanitized
    assert "enable_explainability" in sanitized
    assert "database_password" not in sanitized
    assert "api_key" not in sanitized
    assert "internal_cache_size" not in sanitized
```

## Integration Testing (Revised for New Implementation)

Integration tests should verify that components work together correctly:

```python
# Example integration test for merge explanation with revised components
async def test_merge_explanation_integration():
    # Setup: Create mocks for persistence and merge tracker
    persistence = MockMemoryPersistence()
    merge_tracker = MockMergeTracker()
    
    # Configure persistence mock to return a specific assembly
    assembly = {
        "id": "asm_merged",
        "name": "Merged Assembly",
        "composite_embedding": [0.1, 0.2, 0.3],
        "memory_ids": ["mem1", "mem2"],
        "merged_from": ["asm_source1", "asm_source2"]
    }
    persistence.load_assembly.return_value = assembly
    
    # Configure merge tracker to return specific merge log entries (for reconciliation)
    merge_tracker.get_reconciled_log_entries.return_value = [{
        "merge_event_id": "merge_123",
        "creation_timestamp": "2025-04-01T12:00:00Z",
        "source_assembly_ids": ["asm_source1", "asm_source2"],
        "target_assembly_id": "asm_merged",
        "similarity_at_merge": 0.95,
        "merge_threshold": 0.9,
        "final_cleanup_status": "completed",
        "cleanup_timestamp": "2025-04-01T12:05:00Z"
    }]
    
    # Create a real explainer function with mocked dependencies
    from synthians_memory_core.explainability.merge import generate_merge_explanation
    
    # Execute
    explanation = await generate_merge_explanation(
        assembly_id="asm_merged",
        persistence=persistence,
        merge_tracker=merge_tracker
    )
    
    # Assert
    assert explanation.target_assembly_id == "asm_merged"
    assert explanation.merge_event_id == "merge_123"
    assert explanation.source_assembly_ids == ["asm_source1", "asm_source2"]
    assert explanation.similarity_at_merge == 0.95
    assert explanation.threshold_at_merge == 0.9
    assert explanation.reconciled_cleanup_status == "completed"
    
    # Verify mocks were called correctly
    persistence.load_assembly.assert_called_once_with("asm_merged")
    merge_tracker.get_reconciled_log_entries.assert_called_once()
```

## API Testing (Revised for New Endpoints)

```python
# Example API test for explain_activation endpoint
async def test_explain_activation_api():
    # Setup: Create a test client with mocked dependencies
    app = create_test_app(
        memory_core=MockMemoryCore(),
        explainers=MockExplainabilityModule(),
        config={"ENABLE_EXPLAINABILITY": True}
    )
    client = TestClient(app)
    
    # Mock the explanation result
    mock_explanation = ExplainActivationData(
        assembly_id="asm_test",
        memory_id="mem_test",
        check_timestamp="2025-04-01T12:00:00Z",
        trigger_context="retrieval_query:test",
        calculated_similarity=0.85,
        activation_threshold=0.8,
        passed_threshold=True,
        notes="Similarity exceeded threshold"
    )
    
    # Configure the mock to return our test data
    app.dependency_overrides[get_explainability_module] = lambda: MockExplainabilityModule()
    app.dependency_overrides[get_explainability_module]().generate_activation_explanation.return_value = mock_explanation
    
    # Execute
    response = client.get("/assemblies/asm_test/explain_activation?memory_id=mem_test")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert data["explanation"]["assembly_id"] == "asm_test"
    assert data["explanation"]["memory_id"] == "mem_test"
    assert data["explanation"]["calculated_similarity"] == 0.85
    assert data["explanation"]["passed_threshold"] == True
```

## End-to-End Testing

End-to-end tests should verify the complete flow works correctly:

```python
@pytest.mark.asyncio
async def test_phase_5_9_end_to_end():
    # Setup: Create a real Memory Core with all components
    config = {
        "ENABLE_EXPLAINABILITY": True,
        "merge_log_path": "temp_merge_log.jsonl",
        "merge_log_max_entries": 100,
        "assembly_metrics_persist_interval": 1.0,  # 1 second for faster testing
    }
    
    memory_core = SynthiansMemoryCore(
        config=config,
        # Use real components for an E2E test
    )
    
    # Start the API server
    api = MemoryCoreAPI(memory_core=memory_core)
    server = await api.start(test_mode=True)
    
    try:
        # Create test client
        async with AsyncClient(base_url=f"http://{server.host}:{server.port}") as client:
            # 1. Create two test assemblies
            asm1_id = await memory_core.create_assembly(name="Test Assembly 1")
            asm2_id = await memory_core.create_assembly(name="Test Assembly 2")
            
            # 2. Add memories to both assemblies
            mem1_id = await memory_core.create_memory(
                text="This is a test memory for assembly 1",
                metadata={"source": "test"}
            )
            await memory_core.add_memory_to_assembly(asm1_id, mem1_id)
            
            mem2_id = await memory_core.create_memory(
                text="This is a test memory for assembly 2",
                metadata={"source": "test"}
            )
            await memory_core.add_memory_to_assembly(asm2_id, mem2_id)
            
            # 3. Force a merge between the assemblies
            merged_id = await memory_core.merge_assemblies([asm1_id, asm2_id])
            
            # 4. Wait for the merge cleanup to complete
            await asyncio.sleep(2)  # Allow async cleanup to complete
            
            # 5. Test activation explanation API
            activation_response = await client.get(
                f"/assemblies/{merged_id}/explain_activation?memory_id={mem1_id}"
            )
            assert activation_response.status_code == 200
            activation_data = activation_response.json()
            assert activation_data["success"] == True
            assert activation_data["explanation"]["assembly_id"] == merged_id
            assert activation_data["explanation"]["memory_id"] == mem1_id
            assert "calculated_similarity" in activation_data["explanation"]
            
            # 6. Test merge explanation API
            merge_response = await client.get(f"/assemblies/{merged_id}/explain_merge")
            assert merge_response.status_code == 200
            merge_data = merge_response.json()
            assert merge_data["success"] == True
            assert merge_data["explanation"]["target_assembly_id"] == merged_id
            assert sorted(merge_data["explanation"]["source_assembly_ids"]) == sorted([asm1_id, asm2_id])
            
            # 7. Test lineage API
            lineage_response = await client.get(f"/assemblies/{merged_id}/lineage")
            assert lineage_response.status_code == 200
            lineage_data = lineage_response.json()
            assert lineage_data["success"] == True
            assert lineage_data["target_assembly_id"] == merged_id
            assert len(lineage_data["lineage"]) == 3  # merged + 2 source assemblies
            
            # 8. Test merge log API
            merge_log_response = await client.get("/diagnostics/merge_log")
            assert merge_log_response.status_code == 200
            merge_log_data = merge_log_response.json()
            assert merge_log_data["success"] == True
            assert len(merge_log_data["reconciled_log_entries"]) >= 1
            # Find our merge in the log
            found_merge = False
            for entry in merge_log_data["reconciled_log_entries"]:
                if entry["target_assembly_id"] == merged_id:
                    found_merge = True
                    assert entry["final_cleanup_status"] == "completed"
                    break
            assert found_merge, "Our test merge was not found in the merge log"
            
            # 9. Test runtime config API
            config_response = await client.get("/config/runtime/memory-core")
            assert config_response.status_code == 200
            config_data = config_response.json()
            assert config_data["success"] == True
            assert config_data["service"] == "memory-core"
            # Ensure only safe keys are exposed
            for unsafe_key in ["database_password", "api_key"]:
                assert unsafe_key not in config_data["config"]
    
    finally:
        # Cleanup
        await server.shutdown()
        # Remove temporary files
        if os.path.exists("temp_merge_log.jsonl"):
            os.remove("temp_merge_log.jsonl")
        if os.path.exists("stats/assembly_activation_stats.json"):
            os.remove("stats/assembly_activation_stats.json")
```

## Performance Testing

```python
@pytest.mark.asyncio
async def test_explainability_performance():
    # Setup: Create a memory core with real components
    memory_core = create_real_memory_core()
    
    # Add a significant number of memories and assemblies
    assembly_ids = []
    for i in range(20):  # Create 20 base assemblies
        asm_id = await memory_core.create_assembly(name=f"Assembly {i}")
        assembly_ids.append(asm_id)
        
        # Add 5 memories to each assembly
        for j in range(5):
            mem_id = await memory_core.create_memory(
                text=f"This is memory {j} for assembly {i}",
                metadata={"index": j}
            )
            await memory_core.add_memory_to_assembly(asm_id, mem_id)
    
    # Create a deep merge hierarchy (5 levels)
    merged_ids = assembly_ids.copy()
    for level in range(5):
        if len(merged_ids) < 2:
            break
            
        new_merged_ids = []
        for i in range(0, len(merged_ids), 2):
            if i + 1 < len(merged_ids):
                merged_id = await memory_core.merge_assemblies([merged_ids[i], merged_ids[i+1]])
                new_merged_ids.append(merged_id)
            else:
                new_merged_ids.append(merged_ids[i])  # Odd assembly out
                
        merged_ids = new_merged_ids
    
    # Now test performance of lineage tracing
    final_assembly_id = merged_ids[0]  # The top-level merged assembly
    
    # Measure time to trace lineage
    start_time = time.time()
    lineage = await memory_core.trace_assembly_lineage(final_assembly_id)
    lineage_time = time.time() - start_time
    
    # Assert on reasonable performance
    assert lineage_time < 1.0, f"Lineage tracing took too long: {lineage_time:.3f} seconds"
    assert len(lineage) > 10, "Lineage should include many assemblies"
    
    # Also test merge log performance
    start_time = time.time()
    merge_log = await memory_core.get_merge_log(limit=50)
    merge_log_time = time.time() - start_time
    
    # Assert on reasonable performance
    assert merge_log_time < 0.5, f"Merge log retrieval took too long: {merge_log_time:.3f} seconds"
```

## Security Testing

```python
@pytest.mark.asyncio
async def test_config_api_security():
    # Setup a memory core with a mix of sensitive and non-sensitive config
    config = {
        "embedding_dim": 768,
        "assembly_activation_threshold": 0.8,
        "database_password": "very_secret_password",
        "api_key": "super_secret_api_key_12345",
        "internal_secret": "do_not_expose_this",
        "enable_explainability": True
    }
    
    memory_core = SynthiansMemoryCore(config=config)
    api = MemoryCoreAPI(memory_core=memory_core)
    server = await api.start(test_mode=True)
    
    try:
        # Create test client
        async with AsyncClient(base_url=f"http://{server.host}:{server.port}") as client:
            # Test runtime config API
            response = await client.get("/config/runtime/memory-core")
            assert response.status_code == 200
            data = response.json()
            
            # Verify only safe keys are exposed
            assert "embedding_dim" in data["config"]
            assert "assembly_activation_threshold" in data["config"]
            assert "enable_explainability" in data["config"]
            
            # Verify sensitive keys are NOT exposed
            assert "database_password" not in data["config"]
            assert "api_key" not in data["config"]
            assert "internal_secret" not in data["config"]
            
            # Attempt path traversal or other injection attacks
            response = await client.get("/config/runtime/../../secrets")
            assert response.status_code in [400, 404], "Should reject path traversal attempts"
            
            response = await client.get("/config/runtime/memory-core; cat /etc/passwd")
            assert response.status_code in [400, 404], "Should reject command injection attempts"
    
    finally:
        # Cleanup
        await server.shutdown()
```

## Regression Testing

Regression tests should verify that existing functionality still works with the new features enabled or disabled:

```python
@pytest.mark.asyncio
async def test_core_functionality_with_explainability_disabled():
    # Setup: Create a memory core with explainability disabled
    config = {"ENABLE_EXPLAINABILITY": False}
    memory_core = SynthiansMemoryCore(config=config)
    
    # Test basic operations still work
    assembly_id = await memory_core.create_assembly(name="Test Assembly")
    memory_id = await memory_core.create_memory(text="Test memory")
    await memory_core.add_memory_to_assembly(assembly_id, memory_id)
    
    # Verify assembly was created and memory was added
    assembly = await memory_core.get_assembly(assembly_id)
    assert assembly is not None
    assert memory_id in assembly.memory_ids
    
    # Test API endpoints with explainability disabled
    api = MemoryCoreAPI(memory_core=memory_core)
    server = await api.start(test_mode=True)
    
    try:
        async with AsyncClient(base_url=f"http://{server.host}:{server.port}") as client:
            # Explainability endpoints should return 404 or appropriate error
            response = await client.get(f"/assemblies/{assembly_id}/explain_activation?memory_id={memory_id}")
            assert response.status_code in [404, 501]  # Not Found or Not Implemented
            
            # But core endpoints should still work
            response = await client.get(f"/assemblies/{assembly_id}")
            assert response.status_code == 200
    finally:
        await server.shutdown()
```

## Test Coverage Targets

- **Unit Tests**: ≥ 90% coverage of new code in explainability and diagnostics modules
- **Integration Tests**: Cover all key component interactions
- **API Tests**: 100% coverage of new endpoints
- **End-to-End Tests**: Cover all major user flows

## Monitoring and Debugging During Testing

1. **Log Level**: Set `LOG_LEVEL=DEBUG` during testing to capture detailed information
2. **Tracing**: Enable detailed tracing of API calls and component interactions when debugging test failures
3. **Memory Profiling**: Monitor memory usage during performance testing

These comprehensive tests will help ensure the reliability and correctness of the Phase 5.9 features.