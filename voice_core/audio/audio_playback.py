import pygame
import threading
import queue
from voice_core.shared_state import should_interrupt

# Initialize pygame
print("Initializing pygame")
pygame.mixer.init()
print("Pygame initialized")


def play_audio(audio_data):
    """Play audio_data using pygame and allow interruption."""
    print("Playing audio")
    try:
        pygame.mixer.music.load(audio_data)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and not should_interrupt.is_set():
            pygame.time.Clock().tick(10)
        if should_interrupt.is_set():
            try:
                pygame.mixer.music.stop()
            except Exception as e:
                print(f"Error stopping audio playback: {e}")
            print("Audio playback interrupted")
            should_interrupt.clear()
        else:
            print("Audio playback complete")
    except Exception as e:
        print(f"Error playing audio: {e}")


def playback_worker(playback_queue):
    """Thread worker for playing back audio."""
    audio_buffer = []
    buffer_size = 3  # Adjust how many audio chunks to buffer

    while True:
        if should_interrupt.is_set():
            # Clear current playback, flush buffer
            try:
                pygame.mixer.music.stop()
            except Exception as e:
                print(f"Error stopping audio playback: {e}")
            audio_buffer.clear()
            should_interrupt.clear()
            continue

        # Attempt to fetch next audio_data
        if len(audio_buffer) < buffer_size:
            try:
                audio_data = playback_queue.get(timeout=0.05)
                if audio_data is None:
                    # End signal
                    if audio_buffer:
                        # Play remaining buffer
                        pass
                    else:
                        return
                else:
                    audio_buffer.append(audio_data)
                playback_queue.task_done()
            except queue.Empty:
                # Nothing else to enqueue, continue to process buffer
                pass

        # Process the buffer
        if audio_buffer and not should_interrupt.is_set():
            current_audio = audio_buffer.pop(0)
            play_audio(current_audio)


def cleanup_audio():
    """Clean up pygame audio resources."""
    try:
        pygame.mixer.music.stop()  # Stop any playing music
    except:
        pass  # Ignore errors if no music is playing
    try:
        pygame.mixer.quit()  # Clean up mixer
    except:
        pass  # Ignore cleanup errors
