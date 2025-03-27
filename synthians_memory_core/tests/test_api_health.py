import pytest
import asyncio
import json
from synthians_memory_core.api.client.client import SynthiansClient

@pytest.mark.asyncio
async def test_health_and_stats():
    """Test basic health check and stats endpoints."""
    async with SynthiansClient() as client:
        # Test health endpoint
        health = await client.health_check()
        assert health.get("status") == "healthy", "Health check failed"
        assert "uptime_seconds" in health, "Health response missing uptime"
        assert "version" in health, "Health response missing version"
        
        # Test stats endpoint
        stats = await client.get_stats()
        assert stats.get("success") is True, "Stats endpoint failed"
        assert "api_server" in stats, "Stats missing api_server information"
        assert "memory_count" in stats.get("api_server", {}), "Stats missing memory count"
        
        # Output results for debugging
        print(f"Health check: {json.dumps(health, indent=2)}")
        print(f"Stats: {json.dumps(stats, indent=2)}")

@pytest.mark.asyncio
async def test_api_smoke_test():
    """Test all API endpoints to ensure they respond correctly."""
    async with SynthiansClient() as client:
        # Test embedding generation
        embed_resp = await client.generate_embedding("Test embedding generation")
        assert embed_resp.get("success") is True, "Embedding generation failed"
        assert "embedding" in embed_resp, "No embedding returned"
        assert "dimension" in embed_resp, "No dimension information"
        
        # Test emotion analysis
        emotion_resp = await client.analyze_emotion("I am feeling very happy today")
        assert emotion_resp.get("success") is True, "Emotion analysis failed"
        assert "emotions" in emotion_resp, "No emotions returned"
        assert "dominant_emotion" in emotion_resp, "No dominant emotion identified"
        
        # Test QuickRecal calculation
        qr_resp = await client.calculate_quickrecal(text="Testing QuickRecal API")
        assert qr_resp.get("success") is True, "QuickRecal calculation failed"
        assert "quickrecal_score" in qr_resp, "No QuickRecal score returned"
        
        # Test contradiction detection
        contradict_resp = await client.detect_contradictions(threshold=0.7)
        assert contradict_resp.get("success") is True, "Contradiction detection failed"
