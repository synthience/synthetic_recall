#!/usr/bin/env python3
"""
Enhanced Voice Training Module for fine-tuning STT (Whisper) models with extended features - Cross-platform compatible.
This version fixes the shape mismatch error by removing the incorrect transpose step.
"""

import os
import json
import torch
import torchaudio
import numpy as np
import argparse
import random
import platform
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Union
import logging
import sounddevice as sd
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)
from datasets import Dataset, Audio, DatasetDict
import pyaudio
import wave
import keyboard
import time
import signal
import threading
from tqdm import tqdm
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import (
    Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
)
from rich.live import Live
from rich.panel import Panel

# Set up enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("voice_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up rich console for better UI
console = Console()

# Detect platform
IS_WINDOWS = platform.system() == "Windows"
logger.info(f"Running on platform: {platform.system()}")

@dataclass
class TrainingConfig:
    """Configuration for voice training with enhanced options."""
    # Model configuration
    base_model: str = "openai/whisper-small"
    output_dir: str = "trained_models"
    
    # Audio configuration
    sample_rate: int = 16000
    num_channels: int = 1
    chunk_size: int = 1024
    format: int = pyaudio.paFloat32
    recording_threshold: float = 0.02
    silence_threshold: float = 0.01
    silence_duration: float = 2.0
    max_recording_duration: float = 30.0  # Increased for longer samples
    min_recording_duration: float = 1.0   # Added minimum duration
    
    # Dataset configuration
    num_samples: int = 20
    validation_split: float = 0.2
    seed: int = 42
    
    # Training configuration
    training_epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 100
    logging_steps: int = 5
    save_steps: int = 100
    eval_steps: int = 50
    gradient_clipping: float = 1.0
    max_mel_length: int = 3000  # Whisper expects up to 3000 frames in the mel spectrogram
    
    # Optimization configuration
    use_mixed_precision: bool = True
    freeze_encoder: bool = False
    freeze_layers: List[int] = field(default_factory=list)
    
    # Data augmentation
    use_augmentation: bool = True
    noise_factor: float = 0.05
    pitch_shift_range: float = 2.0
    speed_range: Tuple[float, float] = (0.9, 1.1)
    
    # Additional augmentation options
    use_reverb: bool = False
    reverb_factor: float = 0.3  # Placeholder for reverb strength
    
    # Early stopping
    patience: int = 3
    min_delta: float = 0.01

    # Experiment tracking
    use_wandb: bool = False
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

class SignalHandler:
    """Handles graceful termination of the program."""
    
    def __init__(self):
        self.terminate = False
        self.original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        """Set termination flag and restore original handler."""
        console.print("\n[yellow]Received termination signal. Cleaning up...[/yellow]")
        self.terminate = True
        signal.signal(signal.SIGINT, self.original_sigint)

class AudioVisualizer:
    """Visualizes audio levels during recording using a fixed live panel."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.audio_data = np.zeros(config.chunk_size, dtype=np.float32)
        self.is_recording = False
        self.lock = threading.Lock()
        self.prompt = ""  # Current prompt to be displayed
    
    def set_prompt(self, prompt: str):
        """Set the prompt that should be displayed in the visualizer."""
        self.prompt = prompt

    def render(self, amplitude: float = 0.0, bars: int = 0) -> Panel:
        """Render the current prompt and level as a Panel."""
        text = f"[bold yellow]Prompt:[/bold yellow] {self.prompt}\n"
        text += f"[bold green]Level:[/bold green] {'|' * bars}{' ' * (50 - bars)} {amplitude:.4f}"
        return Panel(text, title="Audio Visualizer", border_style="blue")
    
    def start_visualizing(self):
        """Start the live visualization using Rich Live."""
        self.is_recording = True
        self.live = Live(self.render(), refresh_per_second=10, console=console)
        self.live.start()
        self.thread = threading.Thread(target=self._visualize_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def _visualize_loop(self):
        """Continuously update the live panel with the latest audio level."""
        while self.is_recording:
            with self.lock:
                if len(self.audio_data) > 0:
                    amplitude = np.abs(self.audio_data).mean()
                    bars = int(amplitude / self.config.recording_threshold * 50)
                    bars = min(bars, 50)
                else:
                    amplitude, bars = 0, 0
            self.live.update(self.render(amplitude, bars))
            time.sleep(0.1)
    
    def update_audio_data(self, data):
        """Update audio data for visualization."""
        try:
            with self.lock:
                self.audio_data = np.frombuffer(data, dtype=np.float32)
        except Exception as e:
            with self.lock:
                self.audio_data = np.zeros(self.config.chunk_size, dtype=np.float32)
            logger.warning(f"Error updating visualization: {e}")
    
    def stop_visualizing(self):
        """Stop the live visualization."""
        self.is_recording = False
        if hasattr(self, 'live'):
            self.live.stop()
        if hasattr(self, 'thread') and self.thread.is_alive():
            try:
                self.thread.join(timeout=1.0)
            except Exception as e:
                logger.warning(f"Error stopping visualizer thread: {e}")

class VoiceRecorder:
    """Records voice samples with transcriptions."""
    
    def __init__(self, config: TrainingConfig, signal_handler: SignalHandler):
        self.config = config
        self.signal_handler = signal_handler
        self.audio = pyaudio.PyAudio()
        self.recordings_dir = Path("voice_samples")
        self.recordings_dir.mkdir(exist_ok=True)
        self.transcriptions: Dict[str, str] = {}
        self.visualizer = AudioVisualizer(config)
        
    def record_sample(self, prompt: Optional[str] = None) -> Tuple[str, str]:
        """Record a single voice sample with enhanced UI."""
        console.print("\n[bold green]Recording new sample...[/bold green]")
        if prompt:
            console.print(f"[bold]Please say:[/bold] {prompt}")
            self.visualizer.set_prompt(prompt)
        
        try:
            stream = self.audio.open(
                format=self.config.format,
                channels=self.config.num_channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
        except Exception as e:
            logger.error(f"Error opening audio stream: {e}")
            console.print(f"[bold red]Error opening audio stream: {e}[/bold red]")
            console.print("[yellow]Please check your microphone settings and try again.[/yellow]")
            return "", ""
        
        frames = []
        silent_chunks = 0
        is_recording = False
        start_time = time.time()
        last_space_press = 0
        
        console.print("[bold yellow]Press SPACE to start recording...[/bold yellow]")
        self.visualizer.start_visualizing()
        
        try:
            while not self.signal_handler.terminate:
                if not is_recording:
                    if keyboard.is_pressed('space'):
                        current_time = time.time()
                        if current_time - last_space_press > 0.5:  # Debounce
                            is_recording = True
                            frames = []
                            start_time = time.time()
                            console.print("\n[bold red]Recording started... (Press SPACE again to stop)[/bold red]")
                            last_space_press = current_time
                    time.sleep(0.1)
                    continue
                
                try:
                    data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                    self.visualizer.update_audio_data(data)
                    
                    if keyboard.is_pressed('space'):
                        current_time = time.time()
                        if current_time - last_space_press > 0.5:  # Debounce
                            console.print("\n[bold green]Recording stopped.[/bold green]")
                            break
                    
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    current_amplitude = np.abs(audio_data).mean()
                    if current_amplitude < self.config.silence_threshold:
                        silent_chunks += 1
                        chunk_duration = self.config.chunk_size / self.config.sample_rate
                        if silent_chunks * chunk_duration > self.config.silence_duration:
                            console.print("\n[bold green]Recording stopped (silence detected).[/bold green]")
                            break
                    else:
                        silent_chunks = 0
                    
                    current_duration = time.time() - start_time
                    if current_duration > self.config.max_recording_duration:
                        console.print("\n[bold yellow]Recording stopped (max duration reached).[/bold yellow]")
                        break
                
                except Exception as e:
                    logger.error(f"Error during recording chunk: {e}")
                    time.sleep(0.01)
            
        except Exception as e:
            logger.error(f"Error during recording: {e}")
            console.print(f"[bold red]Error during recording: {e}[/bold red]")
        finally:
            self.visualizer.stop_visualizing()
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                logger.error(f"Error closing stream: {e}")
        
        if self.signal_handler.terminate:
            return "", ""
            
        if len(frames) == 0:
            console.print("[bold red]No audio recorded. Please try again.[/bold red]")
            return self.record_sample(prompt)
        
        recording_duration = len(frames) * self.config.chunk_size / self.config.sample_rate
        if recording_duration < self.config.min_recording_duration:
            console.print(
                f"[bold yellow]Recording too short ({recording_duration:.1f}s). Please try again.[/bold yellow]"
            )
            return self.record_sample(prompt)
        
        recording_id = len(self.transcriptions)
        recording_path = self.recordings_dir / f"sample_{recording_id}.wav"
        
        try:
            wf = wave.open(str(recording_path), 'wb')
            wf.setnchannels(self.config.num_channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.config.sample_rate)
            
            try:
                audio_int16 = (np.frombuffer(b''.join(frames), dtype=np.float32) * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            except Exception as e:
                logger.error(f"Error converting audio format: {e}")
                wf.writeframes(b''.join(frames))
            
            wf.close()
            
            if not self._validate_audio_file(recording_path):
                raise Exception("Audio file validation failed")
            
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            console.print(f"[bold red]Error saving audio file: {e}[/bold red]")
            try:
                logger.info("Trying alternative save method...")
                with open(str(recording_path), 'wb') as f:
                    f.write(b''.join(frames))
                if not self._validate_audio_file(recording_path):
                    raise Exception("Audio file validation failed")
            except Exception as e2:
                logger.error(f"Alternative save method also failed: {e2}")
                console.print("[bold red]Could not save audio. Please try again.[/bold red]")
                return self.record_sample(prompt)
        
        if prompt:
            transcription = prompt
            console.print(f"[bold green]Saved with transcription:[/bold green] {transcription}")
            console.input("[bold yellow]Press Enter to continue...[/bold yellow]")
        else:
            transcription = console.input("[bold yellow]Please enter the transcription:[/bold yellow] ")
        
        self.transcriptions[str(recording_path)] = transcription
        return str(recording_path), transcription
    
    def _validate_audio_file(self, path: Path) -> bool:
        """Validate that the audio file is correctly saved and readable."""
        try:
            if not os.path.exists(path):
                logger.warning(f"Audio file {path} does not exist")
                return False
                
            if os.path.getsize(path) < 100:
                logger.warning(f"Audio file {path} is too small: {os.path.getsize(path)} bytes")
                return False
                
            try:
                waveform, sample_rate = torchaudio.load(path)
                if waveform.shape[0] == 0 or waveform.shape[1] == 0:
                    logger.warning(f"Audio file {path} has invalid dimensions: {waveform.shape}")
                    return False
            except Exception as e:
                logger.warning(f"torchaudio failed to load {path}: {e}")
                with wave.open(str(path), 'rb') as wf:
                    if wf.getnframes() == 0:
                        logger.warning(f"Audio file {path} has 0 frames")
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to validate audio file {path}: {e}")
            return False
    
    def save_transcriptions(self):
        """Save transcriptions to JSON file."""
        try:
            with open(self.recordings_dir / "transcriptions.json", "w") as f:
                json.dump(self.transcriptions, f, indent=2)
            logger.info(f"Saved {len(self.transcriptions)} transcriptions")
        except Exception as e:
            logger.error(f"Error saving transcriptions: {e}")
    
    def load_transcriptions(self):
        """Load transcriptions from JSON file."""
        try:
            transcription_file = self.recordings_dir / "transcriptions.json"
            if transcription_file.exists():
                with open(transcription_file, "r") as f:
                    self.transcriptions = json.load(f)
                logger.info(f"Loaded {len(self.transcriptions)} transcriptions")
            else:
                self.transcriptions = {}
                logger.info("No existing transcriptions found")
        except Exception as e:
            logger.error(f"Error loading transcriptions: {e}")
            self.transcriptions = {}
    
    def augment_audio(self, audio_path: str) -> List[Tuple[str, str]]:
        """Create augmented versions of the audio file - platform independent."""
        if not self.config.use_augmentation:
            return []
        
        augmented_samples = []
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            base_filename = Path(audio_path).stem
            transcription = self.transcriptions[audio_path]
            
            # 1. Noise augmentation
            if self.config.noise_factor > 0:
                try:
                    noise_tensor = torch.randn_like(waveform) * self.config.noise_factor
                    noisy_waveform = waveform + noise_tensor
                    noise_path = str(self.recordings_dir / f"{base_filename}_noise.wav")
                    torchaudio.save(noise_path, noisy_waveform, sample_rate)
                    self.transcriptions[noise_path] = transcription
                    augmented_samples.append((noise_path, transcription))
                    logger.info(f"Created noise-augmented sample: {noise_path}")
                except Exception as e:
                    logger.error(f"Error creating noise augmentation: {e}")
            
            # 2. Volume augmentation
            try:
                volume_factor = random.uniform(0.5, 1.5)
                volume_waveform = waveform * volume_factor
                volume_path = str(self.recordings_dir / f"{base_filename}_volume.wav")
                torchaudio.save(volume_path, volume_waveform, sample_rate)
                self.transcriptions[volume_path] = transcription
                augmented_samples.append((volume_path, transcription))
                logger.info(f"Created volume-augmented sample: {volume_path}")
            except Exception as e:
                logger.error(f"Error creating volume augmentation: {e}")
            
            # 3. Time stretch augmentation (if not on Windows)
            if not IS_WINDOWS and self.config.speed_range[0] < self.config.speed_range[1]:
                try:
                    speed_factor = random.uniform(self.config.speed_range[0], self.config.speed_range[1])
                    effects = [["tempo", str(speed_factor)], ["rate", str(sample_rate)]]
                    speed_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                        waveform, sample_rate, effects
                    )
                    speed_path = str(self.recordings_dir / f"{base_filename}_speed.wav")
                    torchaudio.save(speed_path, speed_waveform, sample_rate)
                    self.transcriptions[speed_path] = transcription
                    augmented_samples.append((speed_path, transcription))
                    logger.info(f"Created speed-augmented sample: {speed_path}")
                except Exception as e:
                    logger.error(f"Error creating speed augmentation: {e}")
            
            # 4. Pitch shift augmentation
            if self.config.pitch_shift_range:
                try:
                    pitch_factor = random.uniform(-self.config.pitch_shift_range, self.config.pitch_shift_range)
                    effects = [["pitch", f"{pitch_factor}"], ["rate", str(sample_rate)]]
                    pitch_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                        waveform, sample_rate, effects
                    )
                    pitch_path = str(self.recordings_dir / f"{base_filename}_pitch.wav")
                    torchaudio.save(pitch_path, pitch_waveform, sample_rate)
                    self.transcriptions[pitch_path] = transcription
                    augmented_samples.append((pitch_path, transcription))
                    logger.info(f"Created pitch-augmented sample: {pitch_path}")
                except Exception as e:
                    logger.error(f"Error creating pitch augmentation: {e}")
            
            # 5. Channel shuffle (only for stereo)
            if waveform.shape[0] > 1:
                try:
                    channel_waveform = torch.flip(waveform, [0])
                    channel_path = str(self.recordings_dir / f"{base_filename}_channel.wav")
                    torchaudio.save(channel_path, channel_waveform, sample_rate)
                    self.transcriptions[channel_path] = transcription
                    augmented_samples.append((channel_path, transcription))
                    logger.info(f"Created channel-augmented sample: {channel_path}")
                except Exception as e:
                    logger.error(f"Error creating channel augmentation: {e}")
            
            # 6. Optional reverb augmentation
            if self.config.use_reverb:
                try:
                    effects = [["reverb", "50", "50", "100"], ["rate", str(sample_rate)]]
                    reverb_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                        waveform, sample_rate, effects
                    )
                    reverb_path = str(self.recordings_dir / f"{base_filename}_reverb.wav")
                    torchaudio.save(reverb_path, reverb_waveform, sample_rate)
                    self.transcriptions[reverb_path] = transcription
                    augmented_samples.append((reverb_path, transcription))
                    logger.info(f"Created reverb-augmented sample: {reverb_path}")
                except Exception as e:
                    logger.error(f"Error creating reverb augmentation: {e}")
            
            logger.info(f"Created {len(augmented_samples)} augmented samples from {audio_path}")
            return augmented_samples
            
        except Exception as e:
            logger.error(f"Error creating augmented samples: {e}")
            return []

class EarlyStoppingCallback(TrainerCallback):
    """Callback for early stopping based on validation loss."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_loss = metrics.get("eval_loss", float('inf'))
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered! Best loss: {self.best_loss:.4f}")
                self.early_stop = True
                control.should_training_stop = True

class WhisperTrainer:
    """Trains Whisper model on voice samples with enhanced features."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        try:
            self.processor = WhisperProcessor.from_pretrained(config.base_model)
            self.model = WhisperForConditionalGeneration.from_pretrained(config.base_model)
            self._apply_freezing()
            self.model.to(self.device)
            
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": config.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise RuntimeError(f"Failed to initialize model: {e}")
    
    def _apply_freezing(self):
        """Apply layer freezing according to configuration."""
        if self.config.freeze_encoder:
            logger.info("Freezing encoder layers")
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False
        
        if self.config.freeze_layers:
            for layer_idx in self.config.freeze_layers:
                logger.info(f"Freezing decoder layer {layer_idx}")
                if hasattr(self.model.model.decoder, "layers") and layer_idx < len(self.model.model.decoder.layers):
                    for param in self.model.model.decoder.layers[layer_idx].parameters():
                        param.requires_grad = False
    
    def prepare_dataset(self, recordings_dir: Path) -> DatasetDict:
        """Load audio + transcription data from disk, create train/val splits, return DatasetDict."""
        transcription_file = recordings_dir / "transcriptions.json"
        if not transcription_file.exists():
            raise FileNotFoundError(f"Transcriptions file not found: {transcription_file}")
            
        with open(transcription_file, "r") as f:
            transcriptions = json.load(f)
            
        if not transcriptions:
            raise ValueError("No transcriptions found in the transcriptions file")
            
        audio_paths = list(transcriptions.keys())
        texts = list(transcriptions.values())
        
        # Filter out any missing audio files
        valid_indices = []
        for i, path in enumerate(audio_paths):
            if os.path.exists(path):
                valid_indices.append(i)
            else:
                logger.warning(f"Audio file not found: {path}")
        
        if not valid_indices:
            raise ValueError("No valid audio files found")
            
        audio_paths = [audio_paths[i] for i in valid_indices]
        texts = [texts[i] for i in valid_indices]
        
        dataset_dict = {
            "audio": audio_paths,
            "text": texts
        }
        
        full_dataset = Dataset.from_dict(dataset_dict)
        full_dataset = full_dataset.cast_column("audio", Audio(sampling_rate=self.config.sample_rate))
        
        train_test_split = full_dataset.train_test_split(
            test_size=self.config.validation_split, 
            seed=self.config.seed
        )
        
        dataset_splits = DatasetDict({
            'train': train_test_split['train'],
            'validation': train_test_split['test']
        })
        
        logger.info(
            f"Dataset prepared with {len(dataset_splits['train'])} training samples "
            f"and {len(dataset_splits['validation'])} validation samples"
        )
        
        return dataset_splits
    
    def preprocess_function(self, examples):
        """Preprocess audio samples for training with correct shape [batch_size, 80, n_frames]."""
        try:
            # Audio is a dictionary with 'array' and 'sampling_rate'
            audio_arrays = [x["array"] for x in examples["audio"]]
            
            # Pad/trim each audio array to 30 seconds at self.config.sample_rate = 480000 samples
            processed_arrays = []
            target_length = 30 * self.config.sample_rate
            for audio in audio_arrays:
                if len(audio) < target_length:
                    # Pad with zeros if too short
                    padding = np.zeros(target_length - len(audio))
                    audio = np.concatenate([audio, padding])
                else:
                    # Trim if too long
                    audio = audio[:target_length]
                processed_arrays.append(audio)
            
            # Use the feature extractor to get mel spectrograms
            features = self.processor.feature_extractor(
                processed_arrays,
                sampling_rate=self.config.sample_rate,
                return_tensors="pt"
            )
            
            # By default, WhisperFeatureExtractor returns shape (batch, 80, n_frames)
            input_features = features.input_features
            logger.info(f"Original mel spectrogram shape: {input_features.shape}")
            # input_features is [batch_size, 80, frames]
            
            # Ensure time dimension is exactly max_mel_length
            # (keep dim=0 as batch_size, dim=1 = 80 mel bins, dim=2 = up to 3000 frames)
            if input_features.shape[2] != self.config.max_mel_length:
                batch_size = input_features.shape[0]
                mel_bins = input_features.shape[1]  # should be 80
                padded_features = torch.zeros(
                    (batch_size, mel_bins, self.config.max_mel_length),
                    dtype=input_features.dtype,
                    device=input_features.device
                )
                
                time_to_copy = min(input_features.shape[2], self.config.max_mel_length)
                padded_features[:, :, :time_to_copy] = input_features[:, :, :time_to_copy]
                input_features = padded_features
            
            logger.info(f"Final mel spectrogram shape: {input_features.shape}")
            
            # Process text targets
            labels = self.processor.tokenizer(
                text=examples["text"],
                return_tensors="pt",
                padding="max_length",
                max_length=448
            ).input_ids
            
            return {"input_features": input_features, "labels": labels}
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise e
    
    def compute_metrics(self, pred):
        """Compute WER/CER metrics if possible (requires `pip install jiwer`)."""
        try:
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            # Replace -100 in the labels as the padding token
            label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
            
            pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)
            
            try:
                import jiwer
                wer = jiwer.wer(label_str, pred_str)
                cer = jiwer.cer(label_str, pred_str)
            except ImportError:
                logger.warning("jiwer not installed, calculating a naive WER")
                wer = self._calculate_wer(label_str, pred_str)
                cer = 0.0
            return {"wer": wer, "cer": cer}
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {"wer": 1.0, "cer": 1.0}
    
    def _calculate_wer(self, references, hypotheses):
        total_words = 0
        total_errors = 0
        for ref, hyp in zip(references, hypotheses):
            ref_words = ref.lower().split()
            hyp_words = hyp.lower().split()
            distance = self._levenshtein_distance(ref_words, hyp_words)
            total_errors += distance
            total_words += len(ref_words)
        return total_errors / max(total_words, 1)
    
    def _levenshtein_distance(self, s, t):
        rows = len(s) + 1
        cols = len(t) + 1
        dist = [[0 for _ in range(cols)] for _ in range(rows)]
        for i in range(1, rows):
            dist[i][0] = i
        for j in range(1, cols):
            dist[0][j] = j
        for j in range(1, cols):
            for i in range(1, rows):
                if s[i-1] == t[j-1]:
                    dist[i][j] = dist[i-1][j-1]
                else:
                    dist[i][j] = min(
                        dist[i-1][j] + 1,
                        dist[i][j-1] + 1,
                        dist[i-1][j-1] + 1
                    )
        return dist[rows-1][cols-1]
    
    def train(self, dataset_dict: DatasetDict):
        """Run the training loop, handle early stopping, and return the training history."""
        try:
            processed_datasets = dataset_dict.map(
                self.preprocess_function,
                batched=True,
                remove_columns=dataset_dict["train"].column_names,
                desc="Processing datasets"
            )
            
            report_to = ["tensorboard", "wandb"] if self.config.use_wandb else "tensorboard"
            
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.config.output_dir,
                evaluation_strategy="steps",
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                max_grad_norm=self.config.gradient_clipping,
                num_train_epochs=self.config.training_epochs,
                warmup_steps=self.config.warmup_steps,
                logging_dir=os.path.join(self.config.output_dir, "logs"),
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                save_total_limit=3,
                fp16=self.config.use_mixed_precision and torch.cuda.is_available(),
                report_to=report_to,
                generation_max_length=225,
                predict_with_generate=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                load_best_model_at_end=True,
            )
            
            early_stopping_callback = EarlyStoppingCallback(
                patience=self.config.patience,
                min_delta=self.config.min_delta
            )
            
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_datasets["train"],
                eval_dataset=processed_datasets["validation"],
                compute_metrics=self.compute_metrics,
                processing_class=self.processor,
                callbacks=[early_stopping_callback]
            )
            
            logger.info("Starting training")
            trainer.train()
            
            final_model_path = os.path.join(self.config.output_dir, "final-model")
            trainer.save_model(final_model_path)
            self.processor.save_pretrained(final_model_path)
            logger.info(f"Model saved to {final_model_path}")
            
            return trainer.state.log_history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise RuntimeError(f"Training failed: {e}")
    
    def test_model(self, audio_path: str) -> str:
        """Run inference (transcription) on a single audio file with the fine-tuned model."""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.config.sample_rate)
                waveform = resampler(waveform)
            # Convert multi-channel to mono by averaging
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Pad or trim to 30s
            audio_array = waveform.numpy()[0]
            target_length = 30 * self.config.sample_rate
            if len(audio_array) < target_length:
                padding = np.zeros(target_length - len(audio_array))
                audio_array = np.concatenate([audio_array, padding])
            else:
                audio_array = audio_array[:target_length]
            
            # Extract mel spectrogram
            features = self.processor.feature_extractor(
                audio_array,
                sampling_rate=self.config.sample_rate,
                return_tensors="pt"
            )
            # shape: [batch_size=1, 80, frames]
            input_features = features.input_features
            
            # Ensure the time dimension is exactly max_mel_length
            if input_features.shape[2] != self.config.max_mel_length:
                batch_size = input_features.shape[0]
                mel_bins = input_features.shape[1]
                padded_features = torch.zeros(
                    (batch_size, mel_bins, self.config.max_mel_length),
                    dtype=input_features.dtype
                )
                time_to_copy = min(input_features.shape[2], self.config.max_mel_length)
                padded_features[:, :, :time_to_copy] = input_features[:, :, :time_to_copy]
                input_features = padded_features
            
            input_features = input_features.to(self.device)
            
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription
        
        except Exception as e:
            logger.error(f"Error during model testing: {e}")
            return f"Error: {str(e)}"

def create_prompts() -> List[str]:
    """
    Return up to 100 prompts focusing on 'Lucidia' usage to fine-tune
    the Whisper model for improved recognition of the term 'Lucidia'.
    """
    return [
         "Lucidia, please activate voice controls.",
        "Good morning, Lucidia. What's on my schedule today?",
        "Could you show me how Lucidia processes commands?",
        "Testing Lucidia's ability to recognize its own name.",
        "Every time I say Lucidia, pay special attention.",
        "Hey Lucidia, I'd like to book a flight to New York.",
        "Remind me to say Lucidia in different intonations.",
        "My voice assistant's name is Lucidia.",
        "When I pronounce Lucidia, try to catch every syllable.",
        "Did Lucidia understand that last command correctly?",
        "Lucidia is here to help with everyday tasks.",
        "I need to confirm that Lucidia can parse complex sentences.",
        "Schedule a meeting using Lucidia's calendar function.",
        "Lucidia, take note of any background noise when I speak.",
        "Let's test Lucidia's ability to handle background music.",
        "Please instruct Lucidia to adjust the volume.",
        "Some users might say Lucidia quickly, or slur the syllables.",
        "Lucidia, do you support multiple languages?",
        "Let me know if Lucidia mishears the word at any point.",
        "How well does Lucidia cope with fast speech?",
        "I rely on Lucidia to keep track of my tasks and reminders.",
        "Whenever I whisper Lucidia, the system should still respond.",
        "Lucidia, clarify if the voice input is unclear.",
        "Constantly repeating Lucidia can help train the model.",
        "Speak slowly to ensure that Lucidia is captured accurately.",
        "The correct spelling is L-U-C-I-D-I-A, Lucidia.",
        "Can Lucidia interpret colloquial speech patterns?",
        "In a noisy environment, does Lucidia still understand me?",
        "I'd love to see Lucidia integrated into other apps.",
        "When traveling, I say Lucidia to start navigation.",
        "Testing Lucidia's response time under stress.",
        "Lucidia, please read the latest news headlines.",
        "I wonder if Lucidia can transcribe a tongue twister.",
        "Does Lucidia rely on advanced language models?",
        "Lucidia must be robust to handle different accents.",
        "Call the voice assistant Lucidia when addressing it.",
        "If I speak softly, can Lucidia still detect its name?",
        "Lucidia, set a timer for ten minutes.",
        "My mother asked, 'What is Lucidia used for?'",
        "I will practice saying Lucidia in a higher pitch now.",
        "Let's measure how Lucidia performs with older microphones.",
        "Does Lucidia store or learn from repeated commands?",
        "I need to confirm Lucidia's interpretation of each phrase.",
        "Sometimes people say Lucidia with an accent, like 'Lu-see-dee-ah.'",
        "In the future, Lucidia might support more advanced features.",
        "Please verify that Lucidia is spelled correctly in the transcript.",
        "If I change languages mid-sentence, will Lucidia adapt?",
        "Lucidia could become a household name someday.",
        "Let's see if Lucidia reacts to synonyms like assistant or AI.",
        "I'd like to see Lucidia's transcription of medical terms.",
        "Note how Lucidia handles unexpected words in a sentence.",
        "Lucidia, can you parse these directions accurately?",
        "Repeat after me: Lucidia is fantastic!",
        "Now let's do a volume check for Lucidia recognition.",
        "We appreciate Lucidia's advanced speech-to-text abilities.",
        "Does Lucidia support multi-user voice profiles?",
        "Let me test Lucidia with a loud background noise again.",
        "Say it again: Lucidia, Lucidia, Lucidia.",
        "Lucidia, what's on the menu tonight?",
        "I rely on Lucidia for both personal and work-related tasks.",
        "Is Lucidia reliant on external servers to interpret speech?",
        "Let's check if Lucidia transcripts match manual transcriptions.",
        "When I need help, I say: 'Lucidia, can you help me?'",
        "I'll attempt to run Lucidia offline for privacy reasons.",
        "Show me how Lucidia handles complicated statements.",
        "Lucidia should handle repeated words gracefully.",
        "How many times should we say Lucidia to be confident?",
        "Here's a short question: Lucidia?",
        "I'm going to test Lucidia from another room.",
        "Explain the difference between Lucia and Lucidia, please.",
        "Lucidia must handle words that are similar but not identical.",
        "I'll whisper: Lucidia, can you hear me?",
        "Let's see how Lucidia does with different microphone distances.",
        "If Lucidia is misheard, does the system correct itself?",
        "Ask Lucidia about the next available date in the calendar.",
        "Please prove that Lucidia can interpret a question mark (?)",
        "The brand name is definitely spelled Lucidia, not Lucida or Lucidya.",
        "Compare Lucidia's accuracy to older voice models.",
        "When I say Lucidia repeatedly, do we see improved recognition?",
        "Practice: Lucidia, read me the first line of this text.",
        "In complicated sentences, Lucidia might need extra training data.",
        "I'd like Lucidia to be integrated into wearable devices.",
        "Let me record a quiet sample: Lucidia… (very softly).",
        "Ensure that the transcript includes the exact word Lucidia.",
        "Sometimes I'll say Lucidia in the middle of a random phrase.",
        "Which tasks does Lucidia handle better than other assistants?",
        "Confirm if Lucidia can differentiate from the word 'Lucidity.'",
        "Converse with me, Lucidia, about the day's events.",
        "Lucidia, do you remember what I said earlier?",
        "In the event of a misinterpretation, say Lucidia again.",
        "Lucidia might become a standard name in voice AI soon.",
        "Read this number sequence, then say Lucidia: one, two, three...",
        "What if I shout: LUCIDIA, LUCIDIA! Does it catch that?",
        "Let's finalize our test: 'Lucidia, finalize your transcription.'",
        "Some people might incorrectly say Lucido or Lucid—train for that.",
        "Which tasks can Lucidia automate for me today?",
        "I appreciate that Lucidia can handle multilingual queries.",
        "Lucidia, please confirm the correct phonetical representation.",
        "I want to verify that after 100 prompts, Lucidia is always understood."
    ]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Voice Training Module")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--record", action="store_true", help="Record new samples")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", type=str, help="Test model on audio file")
    parser.add_argument("--samples", type=int, help="Number of samples to record")
    parser.add_argument("--model", type=str, help="Base model to use")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder layers")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB experiment tracking")
    return parser.parse_args()

def plot_training_results(history, output_dir):
    """Plot training and validation losses from the trainer's log history."""
    try:
        train_losses = []
        val_losses = []
        steps = []
        for entry in history:
            if "loss" in entry:
                train_losses.append(entry["loss"])
                steps.append(entry.get("step", len(train_losses)))
            elif "eval_loss" in entry:
                val_losses.append(entry["eval_loss"])
        
        plt.figure(figsize=(10, 6))
        if train_losses:
            plt.plot(steps[:len(train_losses)], train_losses, label="Training Loss", color="blue")
        if val_losses:
            # We place validation losses approximately by step or at the end
            eval_steps = []
            for i in range(len(val_losses)):
                idx = min(i * len(steps) // len(val_losses), len(steps) - 1) if steps else i
                eval_steps.append(steps[idx] if idx < len(steps) else i)
            plt.plot(eval_steps, val_losses, label="Validation Loss", color="red", marker="o")
        
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        
        plot_path = os.path.join(output_dir, "training_plot.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Training plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Error creating training plot: {e}")

def main():
    """Main training script with enhanced features."""
    args = parse_arguments()
    config = TrainingConfig()
    
    if args.config and os.path.exists(args.config):
        config = TrainingConfig.load(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    
    if args.samples:
        config.num_samples = args.samples
    if args.model:
        config.base_model = args.model
    if args.epochs:
        config.training_epochs = args.epochs
    if args.augment:
        config.use_augmentation = True
    if args.no_augment:
        config.use_augmentation = False
    if args.freeze_encoder:
        config.freeze_encoder = True
    if args.wandb:
        config.use_wandb = True
    
    os.makedirs(config.output_dir, exist_ok=True)
    config.save(os.path.join(config.output_dir, "config.json"))
    
    signal_handler = SignalHandler()
    recorder = VoiceRecorder(config, signal_handler)
    recorder.load_transcriptions()
    
    # If user wants to record new samples OR if neither train/test is specified, record by default
    if args.record or (not args.train and not args.test):
        sample_prompts = create_prompts()
        num_existing = len(recorder.transcriptions)
        num_needed = config.num_samples - num_existing
        
        if num_needed > 0:
            console.print(f"\n[bold green]We'll record {num_needed} voice samples.[/bold green]")
            console.print("[bold]For each sample:[/bold]")
            console.print("1. You'll see a prompt to read")
            console.print("2. Press SPACE to start recording")
            console.print("3. Read the prompt naturally")
            console.print("4. Press SPACE again to stop recording")
            console.print("5. Press Enter to continue to the next sample")
            console.input("\n[bold yellow]Press Enter when ready to begin...[/bold yellow]")
            
            try:
                with Progress(
                    TextColumn("[bold blue]{task.description}[/bold blue]"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeRemainingColumn()
                ) as progress:
                    task = progress.add_task("[green]Recording samples...", total=num_needed)
                    for i in range(num_needed):
                        if signal_handler.terminate:
                            console.print("[bold red]Recording interrupted by user.[/bold red]")
                            break
                        console.print(f"\n[bold]Sample {i+1} of {num_needed}[/bold]")
                        prompt = sample_prompts[i % len(sample_prompts)]
                        path, transcription = recorder.record_sample(prompt)
                        if signal_handler.terminate or not path:
                            break
                        if config.use_augmentation:
                            augmented = recorder.augment_audio(path)
                            if augmented:
                                console.print(f"[green]Created {len(augmented)} augmented versions[/green]")
                        recorder.save_transcriptions()
                        progress.update(task, advance=1)
                
                console.print("[bold green]Sample recording completed![/bold green]")
            except Exception as e:
                logger.error(f"Error during recording session: {e}")
                console.print(f"[bold red]Error during recording session: {e}[/bold red]")
        else:
            console.print(
                f"[bold yellow]Already have {num_existing} samples. No new recordings needed.[/bold yellow]"
            )
    
    # If user wants to train OR if neither record/test is specified, train by default
    if args.train or (not args.record and not args.test):
        console.print("\n[bold green]Starting model training...[/bold green]")
        try:
            if len(recorder.transcriptions) < 2:
                console.print("[bold red]Not enough samples for training. Please record more samples.[/bold red]")
                return
            trainer = WhisperTrainer(config)
            dataset = trainer.prepare_dataset(recorder.recordings_dir)
            training_history = trainer.train(dataset)
            plot_training_results(training_history, config.output_dir)
            console.print(f"\n[bold green]Training complete! Model saved to {config.output_dir}[/bold green]")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            console.print(f"[bold red]Error during training: {e}[/bold red]")
    
    # If user wants to test the model on a given audio file
    if args.test:
        if not os.path.exists(args.test):
            console.print(f"[bold red]Test file not found: {args.test}[/bold red]")
            return
        console.print(f"\n[bold green]Testing model on {args.test}...[/bold green]")
        try:
            trainer = WhisperTrainer(config)
            transcription = trainer.test_model(args.test)
            console.print(f"[bold]Transcription:[/bold] {transcription}")
        except Exception as e:
            logger.error(f"Error during testing: {e}")
            console.print(f"[bold red]Error during testing: {e}[/bold red]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Program terminated by user[/bold red]")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        console.print(f"\n[bold red]Error: {e}[/bold red]")
