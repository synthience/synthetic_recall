import logging
import pyaudio
import numpy as np
import time
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_microphone(device_index: Optional[int] = None, duration: float = 3.0) -> bool:
    """
    Test if the selected microphone is working by measuring audio levels.
    
    Args:
        device_index: Index of the device to test, None for default
        duration: Duration in seconds to test the microphone
    
    Returns:
        bool: True if microphone is working and detecting audio
    """
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 44100
    THRESHOLD = 0.01  # Adjust this for sensitivity

    p = pyaudio.PyAudio()
    
    try:
        logger.info("Testing microphone...")
        logger.info("Please speak into your microphone for %d seconds...", int(duration))
        
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       input_device_index=device_index,
                       frames_per_buffer=CHUNK)
        
        max_amplitude = 0.0
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
                amplitude = np.abs(data).mean()
                max_amplitude = max(max_amplitude, amplitude)
                
                # Only log significant changes in amplitude
                if amplitude > 0.01:
                    bar_length = int(amplitude * 50)
                    bar = '█' * min(bar_length, 50)
                    logger.debug(f"Level: {bar:<50} [{amplitude:.4f}]")
                
            except Exception as e:
                logger.error("Error reading from microphone: %s", e)
                return False
        
        stream.stop_stream()
        stream.close()
        
        if max_amplitude > THRESHOLD:
            logger.info("✓ Microphone test successful! Max amplitude: %.4f", max_amplitude)
            return True
        else:
            logger.warning("✗ No significant audio detected. Please check your microphone.")
            return False
            
    except Exception as e:
        logger.error("Failed to test microphone: %s", e)
        return False
    finally:
        p.terminate()

def get_device_details(p: pyaudio.PyAudio, index: int) -> Dict[str, Any]:
    """Get detailed information about an audio device."""
    try:
        info = p.get_device_info_by_index(index)
        return {
            "index": index,
            "name": info["name"],
            "channels": info["maxInputChannels"],
            "sample_rate": int(info["defaultSampleRate"]),
            "latency": info["defaultLowInputLatency"] * 1000,
            "is_default": p.get_default_input_device_info()["index"] == index
        }
    except Exception as e:
        logger.error(f"Error getting device info for index {index}: {e}")
        return {}

def list_microphones() -> List[Dict[str, Any]]:
    """
    List all available microphones and return their details.
    
    Returns:
        List[Dict[str, Any]]: List of microphone details
    """
    p = pyaudio.PyAudio()
    devices = []
    
    try:
        logger.info("\n=== Available Audio Input Devices ===")
        
        # Get default device first
        try:
            default_info = p.get_default_input_device_info()
            logger.info("\n▶ Default Input Device:")
            logger.info("  Index: %d", default_info["index"])
            logger.info("  Name: %s", default_info["name"])
            logger.info("  Channels: %d", default_info["maxInputChannels"])
            logger.info("  Sample Rate: %d Hz", int(default_info["defaultSampleRate"]))
            logger.info("  Latency: %.2f ms", default_info["defaultLowInputLatency"] * 1000)
        except Exception as e:
            logger.warning("Could not get default device info: %s", e)
        
        logger.info("\n=== Other Input Devices ===")
        for i in range(p.get_device_count()):
            device_info = get_device_details(p, i)
            if device_info and device_info["channels"] > 0:  # Only input devices
                devices.append(device_info)
                logger.info("\nDevice %d%s:", i, " (Default)" if device_info["is_default"] else "")
                logger.info("  Name: %s", device_info["name"])
                logger.info("  Channels: %d", device_info["channels"])
                logger.info("  Sample Rate: %d Hz", device_info["sample_rate"])
                logger.info("  Input Latency: %.2f ms", device_info["latency"])
    except Exception as e:
        logger.error("Error listing audio devices: %s", e)
    finally:
        p.terminate()
    
    return devices

def select_microphone(auto_select_default: bool = True) -> Optional[int]:
    """
    Select a microphone device interactively.
    Returns the device index of the selected microphone.
    """
    p = pyaudio.PyAudio()
    
    try:
        # Get input devices
        devices = []
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:
                devices.append((i, device_info))
        
        if not devices:
            logger.error("No input devices found")
            return None
            
        # Print device information
        for i, (idx, info) in enumerate(devices):
            logger.info(f"\nDevice {idx}:")
            logger.info(f"  Name: {info['name']}")
            logger.info(f"  Channels: {info['maxInputChannels']}")
            logger.info(f"  Sample Rate: {int(info['defaultSampleRate'])} Hz")
            logger.info(f"  Input Latency: {info['defaultLowInputLatency']*1000:.2f} ms")
        
        # Always use device 1 for this system
        if test_microphone(1):
            logger.info("Using device 1")
            return 1
            
        # Fallback to default device if device 1 fails
        if auto_select_default:
            default_idx = p.get_default_input_device_info()['index']
            logger.info(f"Device 1 failed, using default device: {default_idx}")
            if test_microphone(default_idx):
                return default_idx
            
        # Manual selection as last resort
        while True:
            try:
                selection = input("\nSelect input device by number (or press Enter for default): ").strip()
                if not selection:
                    default_idx = p.get_default_input_device_info()['index']
                    logger.info(f"Using default device: {default_idx}")
                    if test_microphone(default_idx):
                        return default_idx
                    continue
                    
                device_idx = int(selection)
                if device_idx < 0 or device_idx >= len(devices):
                    logger.error("Invalid device number")
                    continue
                    
                if test_microphone(device_idx):
                    return device_idx
                    
            except ValueError:
                logger.error("Please enter a valid number")
            except KeyboardInterrupt:
                logger.info("\nMicrophone selection cancelled")
                return None
                
    finally:
        p.terminate()

if __name__ == "__main__":
    selected_index = select_microphone()
    if selected_index is None:
        logger.info("Using default microphone")
    else:
        logger.info("Selected microphone index: %d", selected_index)
