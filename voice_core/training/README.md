# Enhanced Voice Training Module

This module provides advanced tools for recording voice samples and fine-tuning speech-to-text models tailored to your voice.

## Features

- Interactive voice recording with visual feedback
- Automated speech-to-text model training
- Data augmentation for improved model robustness
- Transfer learning optimizations
- Comprehensive evaluation metrics
- Training visualization
- Command-line interface

## Installation

1. Clone this repository or download the script files

2. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. For PyAudio on some systems:
   - Ubuntu/Debian: `sudo apt-get install portaudio19-dev`
   - macOS: `brew install portaudio`
   - Windows: PyAudio is included in the requirements

## Usage

### Basic Usage

```bash
# Record samples and train model
python voice_training.py

# Only record new samples
python voice_training.py --record

# Only train with existing samples
python voice_training.py --train

# Test model on an audio file
python voice_training.py --test path/to/audio.wav
```

### Advanced Options

```bash
# Specify a custom configuration file
python voice_training.py --config my_config.json

# Override number of samples to record
python voice_trainer.py --record --samples 20

# Specify a different base model
python voice_training.py --model "openai/whisper-medium"

# Set custom training epochs
python voice_training.py --train --epochs 10

# Enable data augmentation
python voice_training.py --train --augment

# Freeze encoder layers for faster training
python voice_training.py --train --freeze-encoder
```

## Configuration

You can modify the default configuration by editing the `TrainingConfig` class in the script or by creating a JSON configuration file:

```json
{
  "base_model": "openai/whisper-small",
  "output_dir": "trained_models",
  "sample_rate": 16000,
  "num_samples": 20,
  "training_epochs": 5,
  "learning_rate": 2e-5,
  "batch_size": 4,
  "use_augmentation": true
}
```

## Recording Process

1. When prompted, press SPACE to start recording
2. Speak the prompt text clearly
3. Press SPACE again to stop recording (or wait for automatic stop)
4. Continue until all samples are recorded

## Model Training

The script will:
1. Process all recorded samples
2. Split data into training and validation sets
3. Apply data augmentation (if enabled)
4. Train the model with early stopping
5. Save the trained model
6. Generate training visualization

## Using the Trained Model

After training, your model will be saved in the specified output directory. You can:

1. Test it on new audio files:
```bash
python voice_training.py --test my_audio.wav
```

2. Use it with the Hugging Face transformers library in your applications:
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load your fine-tuned model
processor = WhisperProcessor.from_pretrained("trained_models/final-model")
model = WhisperForConditionalGeneration.from_pretrained("trained_models/final-model")

# Use it for transcription
# ...
```

## Troubleshooting

- **Audio recording issues**: Check your microphone settings and try adjusting the recording thresholds in the configuration.
- **CUDA/GPU errors**: If you encounter GPU-related errors, try setting `use_mixed_precision: false` in your configuration.
- **Memory errors**: Reduce batch size or gradient accumulation steps.
- **Training instability**: Try lowering the learning rate or increasing gradient clipping.

## License

MIT