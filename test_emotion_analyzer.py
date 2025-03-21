import asyncio
import json
import websockets
import sys

async def test_websocket_api():
    """Test the Emotion Analyzer WebSocket API."""
    try:
        uri = "ws://localhost:5007"
        async with websockets.connect(uri) as websocket:
            # Health check
            await websocket.send(json.dumps({"type": "health_check"}))
            response = await websocket.recv()
            print("\n=== Health Check Response ===\n")
            health_response = json.loads(response)
            print(json.dumps(health_response, indent=2))
            
            if health_response.get("status") != "ok":
                print("❌ Health check failed!")
                return False
            
            # Test with a happy text
            happy_text = "I'm so excited and happy about this new project! It's going to be amazing!"
            await websocket.send(json.dumps({"type": "analyze", "text": happy_text}))
            response = await websocket.recv()
            print("\n=== Happy Text Analysis ===\n")
            result = json.loads(response)
            print(json.dumps(result, indent=2))
            
            # Test with a sad text
            sad_text = "I'm feeling very sad and disappointed about the news today."
            await websocket.send(json.dumps({"type": "analyze", "text": sad_text}))
            response = await websocket.recv()
            print("\n=== Sad Text Analysis ===\n")
            result = json.loads(response)
            print(json.dumps(result, indent=2))
            
            # Test with an angry text
            angry_text = "I'm so frustrated and angry about how this was handled!"
            await websocket.send(json.dumps({"type": "analyze", "text": angry_text}))
            response = await websocket.recv()
            print("\n=== Angry Text Analysis ===\n")
            result = json.loads(response)
            print(json.dumps(result, indent=2))
            
            # Test with a neutral text
            neutral_text = "The report contains information about the quarterly earnings."
            await websocket.send(json.dumps({"type": "analyze", "text": neutral_text}))
            response = await websocket.recv()
            print("\n=== Neutral Text Analysis ===\n")
            result = json.loads(response)
            print(json.dumps(result, indent=2))
            
            print("\n✅ All tests completed successfully!\n")
            return True
            
    except Exception as e:
        print(f"\n❌ Error testing WebSocket API: {e}\n")
        return False

async def test_with_custom_text(custom_text):
    """Test the Emotion Analyzer with custom text input."""
    try:
        uri = "ws://localhost:5007"
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps({"type": "analyze", "text": custom_text}))
            response = await websocket.recv()
            print("\n=== Custom Text Analysis ===\n")
            print(f"Text: {custom_text}\n")
            result = json.loads(response)
            print(json.dumps(result, indent=2))
            
            # Extract and display the dominant emotions
            if "dominant_detailed" in result:
                print(f"\nDominant detailed emotion: {result['dominant_detailed']['emotion']} "
                      f"(Confidence: {result['dominant_detailed']['confidence']:.2f})")
            
            if "dominant_primary" in result:
                print(f"Dominant primary emotion: {result['dominant_primary']['emotion']} "
                      f"(Confidence: {result['dominant_primary']['confidence']:.2f})")
            
            return True
    except Exception as e:
        print(f"\n❌ Error analyzing custom text: {e}\n")
        return False

async def test_qr_calculator_integration():
    """Test the integration of Emotion Analyzer with QR Calculator."""
    import torch
    from server.qr_calculator import UnifiedQuickRecallCalculator, QuickRecallMode, QuickRecallFactor
    import numpy as np
    import time
    
    print("\n=== Testing QR Calculator Integration with Emotion Analyzer ===\n")
    
    # Initialize QR calculator
    calculator_config = {
        'embedding_dim': 768,
        'mode': QuickRecallMode.HPC_QR,
        'device': 'cpu',
        'emotional_keywords': ['happy', 'sad', 'angry', 'excited', 'love', 'hate', 'fear'],
        'emotional_intensifiers': ['very', 'extremely', 'so', 'incredibly']
    }
    
    # Set high weight for emotion to see its effect
    factor_weights = {
        QuickRecallFactor.EMOTION: 0.5,  # Set very high to see the impact
        QuickRecallFactor.IMPORTANCE: 0.1,
        QuickRecallFactor.RECENCY: 0.1,
        QuickRecallFactor.PERSONAL: 0.1,
        QuickRecallFactor.COHERENCE: 0.1,
        QuickRecallFactor.SURPRISE: 0.1,
    }
    
    calculator = UnifiedQuickRecallCalculator(calculator_config)
    calculator.set_factor_weights(factor_weights)
    
    # Create a mock embedding
    mock_embedding = torch.randn(768).normalize(p=2, dim=0)
    
    # Test texts with different emotions
    test_texts = [
        "I'm feeling very happy and excited about this new project!",
        "This is a neutral statement about the weather today.",
        "I'm so angry and frustrated with how this situation was handled!",
        "I'm deeply saddened by the news of what happened."
    ]
    
    for text in test_texts:
        # Test with text only
        context = {"timestamp": time.time()}
        score = await calculator.calculate(text, context=context)
        print(f"Text: '{text}'")
        print(f"Score with text only: {score:.4f}\n")
        
        # Test with embedding + text in context
        context = {"timestamp": time.time(), "original_text": text}
        score = await calculator.calculate(mock_embedding, context=context)
        print(f"Score with embedding + text in context: {score:.4f}\n")
        print("-" * 80)

async def main():
    print("\n==== Emotion Analyzer Test ====\n")
    
    # Check if custom text was provided as command line argument
    if len(sys.argv) > 1 and sys.argv[1] == "--test-qr":
        await test_qr_calculator_integration()
    elif len(sys.argv) > 1:
        custom_text = " ".join(sys.argv[1:])
        await test_with_custom_text(custom_text)
    else:
        # Run the standard tests
        await test_websocket_api()
    
if __name__ == "__main__":
    asyncio.run(main())
