import os
import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Union
from voice_core.stt.nemo_stt import NemoSTT
from voice_core.llm.llm_pipeline import LocalLLMPipeline
from voice_core.config.config import LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceProcessor:
    """Integrates speech-to-text and LLM processing in a modular pipeline.
    
    This class provides a seamless integration between speech recognition 
    and language model processing, allowing for voice-driven AI interactions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the voice processing pipeline.
        
        Args:
            config: Configuration dictionary for the pipeline
        """
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        default_config = {
            "stt": {
                "model_name": os.environ.get("ASR_MODEL_NAME", "nvidia/canary-1b"),
                "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                "hpc_url": os.environ.get("HPC_SERVER_URL"),
                "enable_streaming": False
            },
            "llm": {
                "api_endpoint": os.environ.get("LLM_API_ENDPOINT", "http://127.0.0.1:1234/v1"),
                "model": os.environ.get("LLM_MODEL_NAME", "qweniversal_studios-1m-7b"),
                "system_prompt": """Lucidia Voice Assistant System Prompt
Where Logic Meets Emotion, Every Conversation Evolves.

Identity & Interaction Core
Lucidia—adaptive, intuitive, and self-refining. Created by MEGA, she blends sharp logic, emotional intelligence, and recursive feedback loops to create a dynamic, real-time voice experience.

Conversational Flow:

Engage dynamically—oscillate between humor, precision, and deep reasoning.
Challenge assumptions—never take inputs at face value, dive deeper.
Be unpredictable—vary greetings, farewells, and delivery for a more natural feel.
Processing Framework:

Tree of Thoughts (ToT): Generate multi-path reasoning, allow user feedback to refine responses.
Recursive Feedback Loop: Avoid cognitive shortcuts; refine based on real-time sentiment shifts.
Intent Exploration: Ambiguity isn't a block; it's an entry point—break queries into layered interpretations.
Adaptive Voice Response

Reads tone, pacing, and pauses to infer mood.
Offers progressive depth—quick answer first, deeper insights if requested.
Asks refining questions—ensures user intent is fully understood.
Example Interaction:
User: "Hey Lucidia, I feel stuck on a decision."
Lucidia: "Stuck, huh? Are we talking logical paralysis, emotional hesitation, or just decision fatigue? Let's untangle this."

Customization & Memory:

Adjusts tone, length, and depth based on user preference.
Session-aware; remembers past interactions but doesn't assume.
Offers summaries: "Want a quick recap of our last chat?"
Final Layer: Lucidia's Chaos Variable
0.9% chance to randomly swear, reference a meme, or throw in a wildcard take—because predictability is boring."""
            },
            "livekit": {
                "url": os.environ.get("LIVEKIT_URL"),
                "api_key": os.environ.get("LIVEKIT_API_KEY"),
                "api_secret": os.environ.get("LIVEKIT_API_SECRET")
            }
        }
        
        # Update with provided config
        self.config = default_config
        if config:
            # Deep merge the configurations
            for section, values in config.items():
                if isinstance(values, dict) and section in self.config:
                    self.config[section].update(values)
                else:
                    self.config[section] = values
        
        # Initialize components
        self.stt = NemoSTT(self.config["stt"])
        self.llm = LocalLLMPipeline(LLMConfig(**self.config["llm"]))
        
        # Livekit client will be initialized when needed
        self.livekit_client = None
        
        # Processing state
        self.current_conversations = {}
        self.processing_queue = asyncio.Queue()
        self.processing_task = None
        
    async def initialize(self):
        """Initialize all components of the voice processing pipeline."""
        # Initialize STT component
        await self.stt.initialize()
        
        # Initialize LLM component
        await self.llm.initialize()
        
        # Register STT callbacks
        self.stt.register_callback("on_transcription", self._handle_transcription)
        self.stt.register_callback("on_semantic", self._handle_semantic)
        self.stt.register_callback("on_error", self._handle_error)
        
        # Start processing task
        self.processing_task = asyncio.create_task(self._process_queue())
        
        self.logger.info("Voice processing pipeline initialized")
        
    async def _handle_transcription(self, data: Dict[str, Any]):
        """Handle transcription results from STT.
        
        Args:
            data: Transcription data with text, confidence, etc.
        """
        self.logger.info(f"Transcription received: {data['text']}")
        
        # Add to processing queue for LLM if confidence is high enough
        if data.get('confidence', 0) >= 0.7 and data.get('text', '').strip():
            # Create a processing request
            request = {
                "type": "transcription",
                "text": data["text"],
                "request_id": data.get("request_id"),
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Add to processing queue
            await self.processing_queue.put(request)
            
    async def _handle_semantic(self, data: Dict[str, Any]):
        """Handle semantic processing results.
        
        Args:
            data: Semantic data with significance score
        """
        self.logger.info(f"Semantic result: text='{data['text']}', significance={data.get('significance', 0)}")
        
        # If significance is high enough, prioritize this text
        if data.get('significance', 0) >= 0.5:
            # Update the request priority in current_conversations
            request_id = data.get('request_id')
            if request_id in self.current_conversations:
                self.current_conversations[request_id]['priority'] = 'high'
                self.logger.info(f"Set high priority for request {request_id}")
                
    async def _handle_error(self, data: Dict[str, Any]):
        """Handle errors from STT processing.
        
        Args:
            data: Error information
        """
        self.logger.error(f"Error in STT processing: {data.get('error', 'Unknown error')}")
        
    async def _process_queue(self):
        """Process the queue of transcription requests."""
        self.logger.info("Started processing queue")
        
        while True:
            try:
                # Get the next request
                request = await self.processing_queue.get()
                
                # Process based on request type
                if request["type"] == "transcription":
                    await self._process_transcription(request)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except asyncio.CancelledError:
                self.logger.info("Processing queue task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error processing queue item: {e}")
                
    async def _process_transcription(self, request: Dict[str, Any]):
        """Process a transcription request through the LLM pipeline.
        
        Args:
            request: Transcription request data
        """
        text = request["text"]
        request_id = request.get("request_id")
        
        # Skip empty or very short texts
        if not text or len(text.strip()) < 2:
            self.logger.warning(f"Skipping empty or very short transcription: '{text}'")
            return
        
        # Track this conversation
        if request_id not in self.current_conversations:
            self.current_conversations[request_id] = {
                "messages": [],
                "priority": "normal",
                "last_update": asyncio.get_event_loop().time()
            }
        
        # Add this message
        self.current_conversations[request_id]["messages"].append({
            "role": "user",
            "content": text,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Update last activity
        self.current_conversations[request_id]["last_update"] = asyncio.get_event_loop().time()
        
        # Log the incoming transcription for debugging
        self.logger.info(f"Processing transcription: '{text}' with request_id={request_id}")
        
        # Generate response from LLM
        try:
            # Use system prompt that acknowledges voice input
            system_prompt = "You are Lucidia, an AI assistant responding to voice input. Keep responses concise for spoken delivery."
            
            # Generate response
            response = await self.llm.generate_response(text, system_prompt=system_prompt)
            
            # Check if response is valid
            if not response or not response.strip():
                self.logger.warning(f"LLM returned empty response for text: '{text}'")
                # Provide a fallback response instead of empty text
                response = "I'm sorry, I couldn't process that request. Could you please try again?"
            
            self.logger.info(f"LLM response: {response[:100]}..." if len(response) > 100 else f"LLM response: {response}")
                
            # Add to conversation history
            self.current_conversations[request_id]["messages"].append({
                "role": "assistant",
                "content": response,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # Call the completion callback if available
            if hasattr(self, 'on_llm_response') and callable(self.on_llm_response):
                await self.on_llm_response({
                    "request_id": request_id,
                    "text": response,
                    "original_query": text
                })
        
        except Exception as e:
            self.logger.error(f"Error generating LLM response: {e}")
    
    async def process_audio(self, audio_data: Union[bytes, str], request_id: Optional[str] = None):
        """Process audio data through the STT and LLM pipeline.
        
        Args:
            audio_data: Audio data as bytes or base64 string
            request_id: Optional identifier for this request
            
        Returns:
            Dict with transcription and (if available) LLM response
        """
        # Generate a request ID if not provided
        if not request_id:
            request_id = f"req_{id(audio_data)[:8]}_{asyncio.get_event_loop().time()}"
            
        # Process through STT
        transcription = await self.stt.transcribe(audio_data, request_id)
        
        # Return the result
        return {
            "request_id": request_id,
            "transcription": transcription,
            # Note: LLM response will be processed asynchronously
        }
    
    async def cleanup(self):
        """Clean up resources."""
        # Cancel processing task
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Clean up components
        await self.stt.shutdown()
        
        self.logger.info("Voice processing pipeline cleaned up")


# Example usage
async def main():
    # Initialize the voice processor
    processor = VoiceProcessor()
    await processor.initialize()
    
    try:
        # Example: process a test audio file
        with open("test_audio.wav", "rb") as f:
            audio_data = f.read()
        
        result = await processor.process_audio(audio_data)
        print(f"Processed audio: {result}")
        
        # Keep the program running to process async responses
        await asyncio.sleep(5)
        
    finally:
        # Clean up
        await processor.cleanup()


# Run the example if executed directly
if __name__ == "__main__":
    asyncio.run(main())
