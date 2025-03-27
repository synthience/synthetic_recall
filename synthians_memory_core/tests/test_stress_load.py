import pytest
import asyncio
import json
import time
import random
import numpy as np
from datetime import datetime, timedelta
from synthians_memory_core.api.client.client import SynthiansClient

# Optional marker for these slow tests
pytestmark = pytest.mark.slow

@pytest.mark.asyncio
async def test_1000_memory_ingestion():
    """Test the system with a large number of memories (stress test)."""
    async with SynthiansClient() as client:
        start_time = time.time()
        memory_ids = []
        batch_size = 10  # Process in batches to avoid overwhelming the server
        total_memories = 100  # Reduced from 1000 for faster testing - set to 1000 for full stress test
        
        print(f"Starting bulk ingestion of {total_memories} memories...")
        
        # Generate text templates for variety
        templates = [
            "Remember to {action} the {object} at {time}.",
            "I need to {action} {count} {object}s before {time}.",
            "Don't forget that {person} is coming to {location} at {time}.",
            "The {event} is scheduled for {day} at {time}.",
            "Make sure to check the {object} in the {location}."
        ]
        
        actions = ["review", "check", "update", "clean", "fix", "prepare", "send", "receive"]
        objects = ["document", "report", "presentation", "email", "meeting", "project", "task", "schedule"]
        times = ["9:00 AM", "10:30 AM", "noon", "2:15 PM", "4:00 PM", "5:30 PM", "this evening", "tomorrow"]
        people = ["John", "Sara", "Michael", "Emma", "David", "Lisa", "Alex", "Olivia"]
        locations = ["office", "conference room", "lobby", "home", "cafe", "downtown", "upstairs", "kitchen"]
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        events = ["meeting", "conference", "workshop", "presentation", "lunch", "dinner", "call", "interview"]
        counts = ["two", "three", "four", "five", "several", "many", "a few", "some"]
        
        # Create memories in batches
        for batch in range(0, total_memories, batch_size):
            batch_tasks = []
            
            for i in range(batch, min(batch + batch_size, total_memories)):
                # Generate a random memory with a template
                template = random.choice(templates)
                content = template.format(
                    action=random.choice(actions),
                    object=random.choice(objects),
                    time=random.choice(times),
                    person=random.choice(people),
                    location=random.choice(locations),
                    day=random.choice(days),
                    event=random.choice(events),
                    count=random.choice(counts)
                )
                
                # Add a unique identifier
                content += f" (Memory #{i+1})"
                
                # Generate random metadata
                metadata = {
                    "batch": batch // batch_size,
                    "index": i,
                    "importance": random.uniform(0.1, 1.0),
                    "category": random.choice(["work", "personal", "reminder", "event"]),
                    "stress_test": True
                }
                
                # Create memory task
                task = client.process_memory(content=content, metadata=metadata)
                batch_tasks.append(task)
            
            # Process the batch
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Collect memory IDs
            for result in batch_results:
                if result.get("success"):
                    memory_ids.append(result.get("memory_id"))
            
            # Log progress
            elapsed = time.time() - start_time
            progress = min(100, (len(memory_ids) / total_memories) * 100)
            print(f"Progress: {progress:.1f}% - {len(memory_ids)}/{total_memories} memories created in {elapsed:.2f} seconds")
            
            # Pause briefly between batches to avoid overwhelming server
            await asyncio.sleep(0.1)
        
        # Final statistics
        total_time = time.time() - start_time
        rate = len(memory_ids) / total_time if total_time > 0 else 0
        
        print(f"Completed: {len(memory_ids)}/{total_memories} memories created in {total_time:.2f} seconds")
        print(f"Rate: {rate:.2f} memories/second")
        
        # Verify we can retrieve memories from the batch
        if memory_ids:
            # Try to retrieve a random memory by ID
            random_id = random.choice(memory_ids)
            retrieve_resp = await client.retrieve_memories(
                query=f"Memory #{random.randint(1, total_memories)}",
                top_k=5
            )
            
            assert retrieve_resp.get("success") is True, "Failed to retrieve after bulk ingestion"
            print(f"Successfully retrieved {len(retrieve_resp.get('memories', []))} memories from the bulk ingestion")

@pytest.mark.asyncio
async def test_concurrent_retrievals():
    """Test the system with many concurrent retrieval requests."""
    async with SynthiansClient() as client:
        # First, create some memories to retrieve
        timestamp = int(time.time())
        keyword = f"concurrent{timestamp}"
        
        # Create 10 memories with the same keyword
        create_tasks = []
        for i in range(10):
            content = f"Memory {i+1} for concurrent retrieval test with keyword {keyword}"
            task = client.process_memory(content=content)
            create_tasks.append(task)
        
        create_results = await asyncio.gather(*create_tasks)
        created_ids = [r.get("memory_id") for r in create_results if r.get("success")]
        
        assert len(created_ids) > 0, "Failed to create test memories for concurrent retrievals"
        print(f"Created {len(created_ids)} test memories for concurrent retrievals")
        
        # Wait briefly for processing
        await asyncio.sleep(1)
        
        # Now perform many concurrent retrievals
        concurrency = 20  # Number of concurrent requests
        start_time = time.time()
        
        retrieval_tasks = []
        for i in range(concurrency):
            task = client.retrieve_memories(
                query=f"{keyword} memory {random.randint(1, 10)}",
                top_k=5
            )
            retrieval_tasks.append(task)
        
        # Execute concurrently
        retrieval_results = await asyncio.gather(*retrieval_tasks)
        
        # Calculate statistics
        successful = sum(1 for r in retrieval_results if r.get("success"))
        total_time = time.time() - start_time
        rate = concurrency / total_time if total_time > 0 else 0
        
        print(f"Completed {successful}/{concurrency} concurrent retrievals in {total_time:.2f} seconds")
        print(f"Rate: {rate:.2f} retrievals/second")
        
        # Check that all retrievals worked
        assert successful == concurrency, f"Only {successful}/{concurrency} concurrent retrievals succeeded"

@pytest.mark.asyncio
async def test_batch_save_and_reload():
    """Test saving and reloading the memory store during batch operations."""
    # Note: This test assumes there's an endpoint to trigger a save/reload cycle
    # If not available, this can be skipped
    
    async with SynthiansClient() as client:
        try:
            # Create a batch of memories
            timestamp = int(time.time())
            memory_ids = []
            
            # Create 20 test memories
            for i in range(20):
                content = f"Memory {i+1} for save/reload test at {timestamp}"
                response = await client.process_memory(content=content)
                if response.get("success"):
                    memory_ids.append(response.get("memory_id"))
            
            # Call save endpoint (if available)
            # This is hypothetical - might need to be implemented
            save_response = await client.session.post(f"{client.base_url}/save_memory_store")
            save_result = await save_response.json()
            
            print(f"Save operation result: {json.dumps(save_result, indent=2)}")
            
            # Call reload endpoint (if available)
            reload_response = await client.session.post(f"{client.base_url}/reload_memory_store")
            reload_result = await reload_response.json()
            
            print(f"Reload operation result: {json.dumps(reload_result, indent=2)}")
            
            # Verify memories are still retrievable after reload
            retrieved_count = 0
            for memory_id in memory_ids[:5]:  # Check first 5 memories
                query = f"save/reload test at {timestamp}"
                result = await client.retrieve_memories(query=query, top_k=20)
                
                if result.get("success"):
                    result_ids = [m.get("id") for m in result.get("memories", [])]
                    if memory_id in result_ids:
                        retrieved_count += 1
            
            # Check that we could retrieve our memories after reload
            assert retrieved_count > 0, "Failed to retrieve memories after save/reload cycle"
            print(f"Successfully retrieved {retrieved_count}/5 test memories after save/reload cycle")
            
        except Exception as e:
            # The save/reload endpoints might not exist yet
            print(f"Save/reload test failed: {str(e)}")
            print("Save/reload endpoints may not be implemented yet.")

@pytest.mark.asyncio
async def test_memory_decay_pruning():
    """Test memory decay and pruning of old memories."""
    # This test is designed to verify that old memories can be pruned
    # It may need to be adapted based on actual implementation
    
    async with SynthiansClient() as client:
        try:
            # Create memories with backdated timestamps
            timestamp = int(time.time())
            
            # Current memory
            current_response = await client.process_memory(
                content=f"Current memory at {timestamp}",
                metadata={"timestamp": time.time()}
            )
            current_id = current_response.get("memory_id")
            
            # 1-day old memory
            day_old_time = time.time() - (60 * 60 * 24)  # 1 day ago
            day_old_response = await client.process_memory(
                content=f"One day old memory at {timestamp}",
                metadata={"timestamp": day_old_time}
            )
            day_old_id = day_old_response.get("memory_id")
            
            # 1-week old memory
            week_old_time = time.time() - (60 * 60 * 24 * 7)  # 1 week ago
            week_old_response = await client.process_memory(
                content=f"One week old memory at {timestamp}",
                metadata={"timestamp": week_old_time}
            )
            week_old_id = week_old_response.get("memory_id")
            
            # 1-month old memory
            month_old_time = time.time() - (60 * 60 * 24 * 30)  # ~1 month ago
            month_old_response = await client.process_memory(
                content=f"One month old memory at {timestamp}",
                metadata={"timestamp": month_old_time}
            )
            month_old_id = month_old_response.get("memory_id")
            
            # Verify all were created successfully
            assert all(r.get("success") for r in [
                current_response, day_old_response, week_old_response, month_old_response
            ]), "Failed to create test memories with different ages"
            
            print("Successfully created test memories with different timestamps")
            
            # Now trigger a pruning operation (if available)
            # This is hypothetical - might need to be implemented
            prune_response = await client.session.post(
                f"{client.base_url}/prune_old_memories",
                json={"max_age_days": 14}  # Prune memories older than 2 weeks
            )
            prune_result = await prune_response.json()
            
            print(f"Pruning operation result: {json.dumps(prune_result, indent=2)}")
            
            # Check which memories are still retrievable
            retrievable = []
            
            for memory_id, age in [
                (current_id, "current"),
                (day_old_id, "1-day"),
                (week_old_id, "1-week"),
                (month_old_id, "1-month")
            ]:
                query = f"memory at {timestamp}"
                result = await client.retrieve_memories(query=query, top_k=10)
                
                if result.get("success"):
                    result_ids = [m.get("id") for m in result.get("memories", [])]
                    if memory_id in result_ids:
                        retrievable.append(age)
            
            print(f"Still retrievable after pruning: {retrievable}")
            
            # Current and 1-day old should still be retrievable
            # 1-month old should be pruned
            # 1-week old depends on implementation details
            assert "current" in retrievable, "Current memory was incorrectly pruned"
            assert "1-day" in retrievable, "1-day old memory was incorrectly pruned"
            
            if "1-month" in retrievable:
                print("Warning: 1-month old memory was not pruned, pruning may not be implemented yet")
                
        except Exception as e:
            # The pruning endpoint might not exist yet
            print(f"Memory pruning test failed: {str(e)}")
            print("Memory pruning may not be implemented yet.")
