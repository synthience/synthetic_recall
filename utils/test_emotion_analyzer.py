#!/usr/bin/env python
# Utility to test the emotion analyzer integration

import os
import sys
import json
import asyncio
import argparse
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core.enhanced_memory_client import EnhancedMemoryClient

async def test_emotion_analyzer(text: str, use_analyzer: bool = True):
    """Test emotion analyzer integration with a text input."""
    # Configure environment variables for testing if needed
    if use_analyzer:
        os.environ['EMOTION_ANALYZER_HOST'] = os.getenv('EMOTION_ANALYZER_HOST', 'localhost')
        os.environ['EMOTION_ANALYZER_PORT'] = os.getenv('EMOTION_ANALYZER_PORT', '5007')
    
    # Create memory client
    memory_client = EnhancedMemoryClient(
        tensor_server_url="ws://localhost:5003",  # Use default or change as needed
        hpc_server_url="ws://localhost:5004",    # Use default or change as needed
        session_id="test-emotion-session"
    )
    
    # Analyze text for emotional context
    print(f"Analyzing text: '{text}'")
    result = await memory_client.detect_emotional_context(text)
    
    # Pretty print the result
    print("\nEmotion Analysis Result:")
    print(json.dumps(result, indent=2))
    
    # Show the primary emotion and sentiment
    print(f"\nPrimary emotion: {result.get('current_emotion', 'unknown')}")
    print(f"Sentiment score: {result.get('sentiment', 0.0):.2f}")
    
    # Show all detected emotions if available
    if 'emotions' in result and result['emotions']:
        print("\nDetected emotions with confidence:")
        for emotion, confidence in sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {confidence:.2f}")

async def manual_test_emotion_analyzer(text: str):
    """Manually test the emotion analyzer WebSocket connection."""
    import websockets
    
    # Configure the endpoint
    host = os.getenv('EMOTION_ANALYZER_HOST', 'localhost')
    port = os.getenv('EMOTION_ANALYZER_PORT', '5007')
    endpoint = f"ws://{host}:{port}/ws"
    
    print(f"Connecting to emotion analyzer at {endpoint}...")
    try:
        # Connect to the WebSocket with a timeout using asyncio
        async with asyncio.timeout(5):
            async with websockets.connect(endpoint) as websocket:
                # Create the payload - ensure text field is properly formatted
                text_value = text.strip()  # Remove any whitespace
                if not text_value:
                    text_value = "This is a test message with emotion."  # Default text if empty
                    
                payload = {
                    "type": "analyze",
                    "data": {
                        "text": text_value,
                        "analysis_type": "emotion"
                    }
                }
                
                # Send the request
                print(f"Sending request: {payload}")
                await websocket.send(json.dumps(payload))
                
                # Get the response
                response = await websocket.recv()
                print(f"\nReceived response: {response}")
                
                # Parse the JSON response
                result = json.loads(response)
                print("\nEmotion Analysis Result:")
                print(json.dumps(result, indent=2))
                return result
    except Exception as e:
        print(f"\nError connecting to emotion analyzer: {e}")
        return None

async def manual_test_with_both_types(text: str):
    """Test both emotion and emotional_context analysis types."""
    import websockets
    
    # Configure the endpoint
    host = os.getenv('EMOTION_ANALYZER_HOST', 'localhost')
    port = os.getenv('EMOTION_ANALYZER_PORT', '5007')
    endpoint = f"ws://{host}:{port}/ws"
    
    print(f"\n===== TESTING BOTH ANALYSIS TYPES =====\n")
    
    # Prepare clean text
    text_value = text.strip()
    if not text_value:
        text_value = "I am feeling excited and happy about completing this project!"
    
    results = {}
    
    # Test both types
    for analysis_type in ["emotion", "emotional_context"]:
        print(f"\n----- Testing {analysis_type} -----")
        
        try:
            # Connect to the WebSocket with a timeout
            async with asyncio.timeout(5):
                async with websockets.connect(endpoint) as websocket:
                    # Create payload
                    payload = {
                        "type": "analyze",
                        "data": {
                            "text": text_value,
                            "analysis_type": analysis_type
                        }
                    }
                    
                    # Send request
                    print(f"Sending request for {analysis_type}:\n{json.dumps(payload, indent=2)}")
                    await websocket.send(json.dumps(payload))
                    
                    # Get response
                    response = await websocket.recv()
                    print(f"\nReceived response:\n{response}")
                    
                    # Parse response
                    result = json.loads(response)
                    
                    # Store result
                    results[analysis_type] = result
        except Exception as e:
            print(f"\nError testing {analysis_type}: {e}")
            results[analysis_type] = {"error": str(e)}
    
    return results

async def test_plain_payload(text: str):
    """Test with a very simple payload structure."""
    import websockets
    
    # Configure the endpoint
    host = os.getenv('EMOTION_ANALYZER_HOST', 'localhost')
    port = os.getenv('EMOTION_ANALYZER_PORT', '5007')
    endpoint = f"ws://{host}:{port}/ws"
    
    print(f"\n===== TESTING SIMPLIFIED PAYLOAD =====\n")
    
    try:
        # Connect to the WebSocket with a timeout
        async with asyncio.timeout(5):
            async with websockets.connect(endpoint) as websocket:
                # Create a simple payload
                simple_payload = {"text": text}
                
                # Send request
                print(f"Sending simple payload:\n{json.dumps(simple_payload, indent=2)}")
                await websocket.send(json.dumps(simple_payload))
                
                # Get response
                response = await websocket.recv()
                print(f"\nReceived response:\n{response}")
                
                # Parse response
                result = json.loads(response)
                print("\nParsed result:")
                print(json.dumps(result, indent=2))
                return result
    except Exception as e:
        print(f"\nError with simple payload: {e}")
        return {"error": str(e)}

async def test_correct_format(text: str):
    """Test with the format expected by the emotion analyzer."""
    import websockets
    
    # Configure the endpoint
    host = os.getenv('EMOTION_ANALYZER_HOST', 'localhost')
    port = os.getenv('EMOTION_ANALYZER_PORT', '5007')
    endpoint = f"ws://{host}:{port}/ws"
    
    print(f"\n===== TESTING CORRECT FORMAT =====\n")
    
    try:
        # Connect to the WebSocket with a timeout
        async with asyncio.timeout(10):  # Increased timeout
            async with websockets.connect(endpoint) as websocket:
                # First try a health check
                health_check = {"type": "health_check"}
                print(f"Sending health check:\n{json.dumps(health_check, indent=2)}")
                await websocket.send(json.dumps(health_check))
                
                # Get response
                response = await websocket.recv()
                print(f"\nReceived health check response:\n{response}")
                
                # Now try the analyze endpoint with the correct format
                if text.strip():
                    payload = {
                        "type": "analyze",
                        "text": text.strip()
                    }
                    
                    print(f"\nSending analyze request:\n{json.dumps(payload, indent=2)}")
                    await websocket.send(json.dumps(payload))
                    
                    # Get response
                    response = await websocket.recv()
                    print(f"\nReceived analyze response:\n{response}")
                    
                    # Parse response
                    result = json.loads(response)
                    print("\nParsed result:")
                    print(json.dumps(result, indent=2))
                    return result
    except Exception as e:
        print(f"\nError with correct format: {e}")
        return {"error": str(e)}

async def test_with_delay(text: str):
    """Test with the correct format and delay between messages."""
    import websockets
    
    # Configure the endpoint
    host = os.getenv('EMOTION_ANALYZER_HOST', 'localhost')
    port = os.getenv('EMOTION_ANALYZER_PORT', '5007')
    endpoint = f"ws://{host}:{port}/ws"
    
    print(f"\n===== TESTING WITH DELAY =====\n")
    
    try:
        # Connect to the WebSocket with a timeout
        async with asyncio.timeout(10):  # Increased timeout
            async with websockets.connect(endpoint) as websocket:
                # First try a health check
                health_check = {"type": "health_check"}
                print(f"Sending health check:\n{json.dumps(health_check, indent=2)}")
                await websocket.send(json.dumps(health_check))
                
                # Get response
                response = await websocket.recv()
                print(f"\nReceived health check response:\n{response}")
                
                # Brief delay
                await asyncio.sleep(1)
                
                # Now try the analyze endpoint with the correct format
                if text.strip():
                    payload = {
                        "type": "analyze",
                        "text": text.strip()
                    }
                    
                    print(f"\nSending analyze request:\n{json.dumps(payload, indent=2)}")
                    await websocket.send(json.dumps(payload))
                    
                    # Wait for response with a longer timeout
                    try:
                        async with asyncio.timeout(5):
                            response = await websocket.recv()
                            print(f"\nReceived analyze response:\n{response}")
                            
                            # Parse response
                            result = json.loads(response)
                            print("\nParsed result:")
                            print(json.dumps(result, indent=2))
                            return result
                    except asyncio.TimeoutError:
                        print("\nTimeout waiting for analyze response!")
                        return {"error": "timeout"}
    except Exception as e:
        print(f"\nError with delay test: {e}")
        return {"error": str(e)}

async def test_integration():
    """Test the integration between EmotionMixin and the emotion analyzer."""
    from memory_core.emotion import EmotionMixin
    
    print(f"\n===== TESTING MIXIN INTEGRATION =====\n")
    
    # Create a simple EmotionMixin instance for testing
    class TestEmotionMixin(EmotionMixin):
        def _get_current_timestamp(self):
            return time.time()
            
        async def _connect_to_hpc(self):
            # Return a mock connection that always raises an exception
            class MockConnection:
                async def close(self):
                    pass
                    
                async def detect_emotion(self, text):
                    raise Exception("HPC not available")
                    
                async def detect_emotional_context(self, text):
                    raise Exception("HPC not available")
            return MockConnection()
    
    # Create an instance with a random name
    test_mixin = TestEmotionMixin()
    
    # Test with a happy text
    happy_text = "I am feeling really excited and happy about this project's success!"
    try:
        print(f"Testing detect_emotion with: {happy_text}")
        emotion = await test_mixin.detect_emotion(happy_text)
        print(f"Detected emotion: {emotion}")
        
        print(f"\nTesting emotional_context with: {happy_text}")
        context = await test_mixin.detect_emotional_context(happy_text)
        print(f"Emotional context: {json.dumps(context, indent=2)}")
        
        # Test with a sad text
        sad_text = "I am feeling quite sad and disappointed about the results."
        print(f"\nTesting detect_emotion with: {sad_text}")
        emotion = await test_mixin.detect_emotion(sad_text)
        print(f"Detected emotion: {emotion}")
        
        print(f"\nTesting emotional_context with: {sad_text}")
        context = await test_mixin.detect_emotional_context(sad_text)
        print(f"Emotional context: {json.dumps(context, indent=2)}")
        
        return True
    except Exception as e:
        print(f"Error testing integration: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integration_clean():
    """Test the integration between EmotionMixin and the emotion analyzer with clean output."""
    from memory_core.emotion import EmotionMixin
    import logging
    
    print(f"\n===== TESTING MIXIN INTEGRATION (CLEAN OUTPUT) =====\n")
    
    # Temporarily disable logging
    logging.disable(logging.WARNING)
    
    # Create a simple EmotionMixin instance for testing
    class TestEmotionMixin(EmotionMixin):
        def _get_current_timestamp(self):
            return time.time()
            
        async def _connect_to_hpc(self):
            # Return a mock connection that always raises an exception
            class MockConnection:
                async def close(self):
                    pass
                    
                async def detect_emotion(self, text):
                    raise Exception("HPC not available")
                    
                async def detect_emotional_context(self, text):
                    raise Exception("HPC not available")
            return MockConnection()
    
    # Create an instance
    test_mixin = TestEmotionMixin()
    
    try:
        # Test various emotional texts
        texts = [
            "I am feeling really excited and happy about this project's success!",
            "I am feeling quite sad and disappointed about the results.",
            "I'm so angry about how this situation was handled!",
            "I'm a bit nervous about the upcoming presentation.",
            "I'm surprised by the unexpected turn of events."
        ]
        
        print("Testing emotion detection:\n")
        print("{:<60} {:<15}".format("Text", "Emotion"))
        print("-" * 75)
        
        for text in texts:
            emotion = await test_mixin.detect_emotion(text)
            print("{:<60} {:<15}".format(text[:57] + "..." if len(text) > 57 else text, emotion))
        
        print("\n\nTesting emotional context:\n")
        print("{:<60} {:<15} {:<10}".format("Text", "Emotion", "Sentiment"))
        print("-" * 85)
        
        for text in texts:
            context = await test_mixin.detect_emotional_context(text)
            print("{:<60} {:<15} {:<10.2f}".format(
                text[:57] + "..." if len(text) > 57 else text,
                context["emotional_state"],
                context["sentiment"]
            ))
            
        # Re-enable logging
        logging.disable(logging.NOTSET)
        return True
    except Exception as e:
        # Re-enable logging
        logging.disable(logging.NOTSET)
        print(f"Error testing integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test the emotion analyzer integration")
    parser.add_argument("--text", help="Text to analyze for emotional context", default="I am feeling excited about this test!")
    parser.add_argument("--no-analyzer", action="store_true", help="Disable emotion analyzer and use HPC fallback")
    parser.add_argument("--manual", action="store_true", help="Directly test the WebSocket connection")
    parser.add_argument("--both-types", action="store_true", help="Test both emotion and emotional_context analysis types")
    parser.add_argument("--plain-payload", action="store_true", help="Test with a simplified payload structure")
    parser.add_argument("--correct-format", action="store_true", help="Test with the format expected by the emotion analyzer")
    parser.add_argument("--with-delay", action="store_true", help="Test with the correct format and delay between messages")
    parser.add_argument("--integration", action="store_true", help="Test the integration between EmotionMixin and the emotion analyzer")
    parser.add_argument("--integration-clean", action="store_true", help="Test the integration between EmotionMixin and the emotion analyzer with clean output")
    args = parser.parse_args()
    
    if args.integration_clean:
        asyncio.run(test_integration_clean())
    elif args.integration:
        asyncio.run(test_integration())
    elif args.both_types:
        asyncio.run(manual_test_with_both_types(args.text))
    elif args.plain_payload:
        asyncio.run(test_plain_payload(args.text))
    elif args.correct_format:
        asyncio.run(test_correct_format(args.text))
    elif args.with_delay:
        asyncio.run(test_with_delay(args.text))
    elif args.manual:
        asyncio.run(manual_test_emotion_analyzer(args.text))
    else:
        asyncio.run(test_emotion_analyzer(args.text, not args.no_analyzer))

if __name__ == "__main__":
    main()
