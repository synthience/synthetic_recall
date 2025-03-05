import asyncio
from voice_core.config.config import LucidiaConfig, PHRASE_TIMEOUT, MIN_PHRASE_TIMEOUT
from voice_core.mic_utils import select_microphone
from voice_core.custom_speech_recognition import StreamingRecognizer
from voice_core.tts_utils import select_voice
from voice_core.llm_communication import get_llm_response, verify_llm_server
from voice_core.response_processor import process_response
from voice_core.audio_playback import cleanup_audio
import queue
import voice_core.shared_state as shared_state

# Create default configuration instance
config = LucidiaConfig()
conversation_manager = None

async def get_audio_input():
    """Get audio input using streaming recognition (minimal delay)."""
    print("\nStarting audio processing pipeline")
    recognizer = StreamingRecognizer(shared_state.selected_microphone)
    print("Initializing speech recognition...")
    recognizer.start()
    print("Speech recognition started")

    full_text = []
    last_text_time = asyncio.get_event_loop().time()
    collection_started = False

    try:
        while True:
            try:
                if not collection_started:
                    print("Waiting for speech input...")

                text = recognizer.text_queue.get(timeout=PHRASE_TIMEOUT)
                current_time = asyncio.get_event_loop().time()

                if not collection_started:
                    print("Speech detected, collecting input...")
                    collection_started = True

                print(f"Received text: '{text}'")
                full_text.append(text)
                last_text_time = current_time

                if text.strip().endswith((".", "!", "?")):
                    print("Sentence break detected, checking for more input...")
                    try:
                        next_text = recognizer.text_queue.get(
                            timeout=MIN_PHRASE_TIMEOUT
                        )
                        print(f"Additional text received: '{next_text}'")
                        full_text.append(next_text)
                        last_text_time = asyncio.get_event_loop().time()
                    except queue.Empty:
                        pass

                # Check if we've been silent for long enough to conclude
                current_time = asyncio.get_event_loop().time()
                silence_duration = current_time - last_text_time
                # If we've been silent long enough, conclude
                if silence_duration > MIN_PHRASE_TIMEOUT:
                    print("Speech input complete")
                    if full_text:  # Only break if we have text to process
                        break

            except queue.Empty:
                if collection_started:
                    print("No more input detected")
                    if full_text:  # Only break if we have text to process
                        break
                continue

    except Exception as e:
        print(f"Error in audio input: {e}")
        return None

    finally:
        recognizer.stop()

    return " ".join(full_text)


async def main():
    """Main entry point."""
    global conversation_manager
    
    print("Entering main function")
    print("Conversational AI with Local LLM, Edge TTS, and Speech Recognition")
    print("You can say 'stop', 'wait', 'pause', or 'quiet' to interrupt the AI's speech.\n")

    # Initialize conversation manager
    conversation_manager = voice_core.conversation_manager.ConversationManager()
    if not await conversation_manager.connect():
        print("Warning: Could not connect to memory system. Continuing without memory.")

    # Verify LLM server connection
    if not verify_llm_server():
        print("Error: Unable to connect to LLM server. Please ensure it is running.")
        return

    # Select and calibrate microphone at startup
    print("\nSetting up microphone...")
    shared_state.selected_microphone = select_microphone()
    if shared_state.selected_microphone is None:
        print("Using default microphone")
    else:
        print(f"Selected microphone index: {shared_state.selected_microphone}")

    # Initialize recognizer to perform one-time calibration
    print("\nPerforming initial microphone calibration...")
    initial_recognizer = StreamingRecognizer(shared_state.selected_microphone)
    initial_recognizer.start()
    initial_recognizer.stop()
    print("Microphone setup complete\n")

    voice = await select_voice()
    if not voice:
        print("No voice selected. Exiting.")
        return

    print(f"Selected voice: {voice}")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Or press Enter (empty) to use voice input.\n")

    while True:
        try:
            # Get user input
            user_input = input("You (type or press Enter to speak): ")
            if user_input.lower() in ["quit", "exit", "bye"]:
                break

            # Handle speech input if no text input
            if not user_input:
                print("\nStarting speech recognition...")
                try:
                    user_input = await get_audio_input()
                    if user_input:
                        print(f"\nSpeech recognized successfully: {user_input}")
                    else:
                        print("No speech input detected")
                        continue
                except Exception as e:
                    print(f"\nError during speech recognition: {e}")
                    continue

            # Process input through LLM if we have valid input
            if user_input:
                # Store user input in memory
                if conversation_manager:
                    await conversation_manager.add_to_memory(user_input, "user")
                    # Get relevant context
                    await conversation_manager.search_relevant_context(user_input)
                
                print("\nSubmitting to LLM for processing...")
                print("AI: ", end="", flush=True)

                try:
                    # Get context-enhanced prompt
                    context_prompt = conversation_manager.get_context_for_llm(user_input) if conversation_manager else user_input
                    
                    # Get LLM response
                    llm_response = get_llm_response(context_prompt, stream=True)
                    if not llm_response:
                        print("Error: Failed to get response from LLM")
                        continue

                    # Store AI response in memory
                    if conversation_manager:
                        await conversation_manager.add_to_memory(llm_response, "assistant")

                    # Process the response
                    print("Processing LLM response...")
                    await process_response(llm_response, voice, streaming=True)
                    print("Response processing complete")
                except Exception as e:
                    print(f"\nError during LLM processing: {e}")
                    continue

        except Exception as e:
            print(f"\nUnexpected error: {e}")
            continue

    print("Thank you for the conversation!")
    if conversation_manager:
        await conversation_manager.close()
    cleanup_audio()


if __name__ == "__main__":
    print("Starting main function")
    asyncio.run(main())
    print("Script completed")
