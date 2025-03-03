# Audio Transcription & Summarization Tool

A Python application that automatically transcribes audio files and summarizes the content, optimized for Apple Silicon (M1/M2) Macs.

## Features

- **Audio Transcription**: Convert spoken words from audio files to text
- **Text Summarization**: Automatically create concise summaries of transcribed content
- **M1/M2 Optimization**: Specifically tuned for Apple Silicon Mac performance
- **Multiple Audio Processing Options**: Support for ffmpeg, libav, and SoX
- **Fallback Systems**: Multiple layers of fallbacks to ensure reliability
- **Easy-to-Use Interface**: Simple Gradio web interface for uploading and processing audio

## Requirements

- Python 3.7+
- macOS (optimized for Apple Silicon M1/M2)
- An audio processing tool (one of the following):
  - ffmpeg (recommended)
  - libav
  - SoX

## Installation

1. Clone or download this repository:

```bash
git clone <repository-url>
cd audio-transcription-tool
```

2. Install required dependencies:

```bash
pip install gradio whisper transformers torch sentencepiece accelerate
```

3. For better performance (recommended):

```bash
pip install faster-whisper
```

4. Install an audio processing tool (if not already installed):

```bash
# Using Homebrew on macOS
brew install ffmpeg
# OR
brew install libav
# OR
brew install sox
```

## Usage

1. Run the application:

```bash
python whisper_app_m1.py
```

2. Open the provided URL in your web browser (typically http://127.0.0.1:7860)

3. Use the interface to:
   - Record audio directly from your microphone
   - Upload existing audio files (mp3, wav, m4a, etc.)
   - Click "Transcribe & Summarize" to process the audio
   - View the transcript and summary results

## Troubleshooting

### Common Issues

- **"No audio processing tools found"**: Install ffmpeg, libav, or SoX using the commands in the Installation section.
- **Memory errors**: Try using the smaller "tiny" Whisper model by changing the model size in the code.
- **GPU-related errors**: The application should automatically fall back to CPU mode, but you can force CPU-only mode by setting the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1`.

### M1/M2 Mac Specific Issues

This application is designed to work around common issues on Apple Silicon:

- PyTorch MPS backend limitations with sparse tensors
- Compatibility issues with certain model operations
- GPU memory management quirks on Apple Silicon

The app automatically detects these issues and applies appropriate fallbacks to ensure functionality.

## Performance Optimization

For best performance on M1/M2 Macs:

1. Install faster-whisper: `pip install faster-whisper`
2. Ensure you have ffmpeg installed: `brew install ffmpeg`
3. For processing longer audio files, ensure adequate memory is available
4. Close resource-intensive applications when processing large files

## How It Works

1. **Audio Processing**: The uploaded audio is processed using ffmpeg, libav, or SoX
2. **Transcription**: The processed audio is transcribed using OpenAI's Whisper model
   - On M1/M2, this optimally uses faster-whisper with int8 quantization on CPU
3. **Summarization**: The transcript is processed through a BART-CNN model
4. **Results Display**: Both the full transcript and summary are presented in the interface

## License

[Specify your license here]

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for optimized Whisper implementation
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the summarization model
- [Gradio](https://github.com/gradio-app/gradio) for the web interface