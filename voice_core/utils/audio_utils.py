import numpy as np
from livekit import rtc

class AudioFrame:
    def __init__(self, data: bytes, sample_rate: int, num_channels: int, samples_per_channel: int):
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel

    def to_rtc(self) -> rtc.AudioFrame:
        return rtc.AudioFrame(
            data=self.data,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            samples_per_channel=self.samples_per_channel
        )

def normalize_audio(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        max_val = np.max(np.abs(data))
        if max_val > 1.0:
            data = data / max_val
    return data

def resample_audio(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    from scipy import signal
    if src_rate == dst_rate:
        return data
    target_length = int(len(data) * dst_rate / src_rate)
    resampled = signal.resample(data, target_length)
    return resampled

def split_audio_chunks(data: np.ndarray, chunk_size: int, overlap: int = 0) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")
    step = chunk_size - overlap
    num_chunks = (len(data) - overlap) // step
    chunks = np.zeros((num_chunks, chunk_size), dtype=data.dtype)
    for i in range(num_chunks):
        start = i * step
        end = start + chunk_size
        chunks[i] = data[start:end]
    return chunks

def convert_to_pcm16(data: np.ndarray) -> bytes:
    if data.dtype == np.float32:
        data = (data * 32767).astype(np.int16)
    elif data.dtype != np.int16:
        raise ValueError(f"Unsupported audio data type: {data.dtype}")
    return data.tobytes()

def create_audio_frame(data: np.ndarray, sample_rate: int, num_channels: int = 1) -> AudioFrame:
    if len(data.shape) == 1:
        data = data.reshape(-1, num_channels)
    samples_per_channel = data.shape[0]
    pcm_data = convert_to_pcm16(data)
    return AudioFrame(
        data=pcm_data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel
    )