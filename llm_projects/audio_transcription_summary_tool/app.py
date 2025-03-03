import gradio as gr
import time
import os
import tempfile
import subprocess
import sys
import platform

# Check for audio processing tools
def check_audio_tools():
    # Try to find ffmpeg first (most common)
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpeg is installed! Using ffmpeg for audio processing.")
        return "ffmpeg"
    except FileNotFoundError:
        # Try libav (avconv) as an alternative
        try:
            subprocess.run(['avconv', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("avconv (libav) is installed! Using libav for audio processing.")
            return "libav"
        except FileNotFoundError:
            # Try SoX as another alternative
            try:
                subprocess.run(['sox', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print("SoX is installed! Using SoX for audio processing.")
                return "sox"
            except FileNotFoundError:
                print("No audio processing tools found. Please install one of the following:")
                print("- ffmpeg (recommended): brew install ffmpeg")
                print("- libav: brew install libav")
                print("- SoX: brew install sox")
                print("\nWould you like to continue without audio tools? Some functionality may be limited.")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    sys.exit(1)
                return "none"

# Check for required packages and install if missing
def check_and_install_packages():
    required_packages = ['whisper', 'transformers', 'torch', 'sentencepiece', 'accelerate']
    
    # For M1 Macs, ensure PyTorch is installed with MPS support
    if platform.processor() == 'arm' and sys.platform == 'darwin':
        print("M1/M2 Mac detected - ensuring PyTorch with MPS support is installed...")
        try:
            import torch
            if not torch.backends.mps.is_available():
                print("PyTorch MPS is not available. Installing PyTorch with MPS support...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'torch'])
        except (ImportError, AttributeError):
            print("Installing PyTorch with MPS support...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'torch'])
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"{package} not found. Installing...")
            if package == 'torch' and platform.processor() == 'arm' and sys.platform == 'darwin':
                # Skip if we already handled torch for M1
                continue
            subprocess.run([sys.executable, '-m', 'pip', 'install', package])

# Run checks
audio_tool = check_audio_tools()
check_and_install_packages()

# Now import after checks
import torch
import whisper
from transformers import pipeline

# Force CPU-only mode to avoid MPS sparse tensor issues
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback for operations that support it
device = "cpu"  # Force CPU for all models
print("Using CPU mode for all models to ensure compatibility on Apple Silicon")

# Check for PyTorch installation
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("PyTorch not found. Installing...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch'])


# Try to use faster-whisper which works better on M1 Macs with CPU
try:
    from faster_whisper import WhisperModel
    print("Using faster-whisper for improved performance on CPU!")
    
    # Always use CPU with int8 quantization (most reliable on M1)
    model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)
    use_faster_whisper = True
    print("Loaded faster-whisper model with optimized CPU settings")
        
except (ImportError, ValueError) as e:
    if isinstance(e, ValueError):
        try:
            # If int8 failed, try with standard float32
            print(f"Could not use int8 precision: {str(e)}")
            print("Trying with standard float32...")
            model = WhisperModel("small", device="cpu", compute_type="float32")
            use_faster_whisper = True
            print("Loaded faster-whisper model with float32 precision")
        except Exception as e2:
            print(f"Failed to initialize faster-whisper: {str(e2)}")
            use_faster_whisper = False
    else:
        print("faster-whisper not found. Using standard whisper.")
        print("For better performance on M1/M2 Mac, consider:")
        print("pip install faster-whisper")
        use_faster_whisper = False

# Load standard whisper as fallback
if not use_faster_whisper:
    try:
        # Ensure we use tiny model which has fewer parameters and is less likely 
        # to hit compatibility issues
        print("Loading Whisper tiny model on CPU...")
        model = whisper.load_model("tiny", device="cpu")
        print("Whisper tiny model loaded successfully!")
    except Exception as e:
        print(f"Error loading standard whisper model: {str(e)}")
        print("Trying with environment variable workarounds...")
        # Last resort - use environment variables and minimal model
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model = whisper.load_model("tiny", device="cpu", download_root=os.path.expanduser("~/.cache/whisper"))
        print("Whisper tiny model loaded with fallback configuration!")

# Load a lightweight summarization model
print("Loading summarization model...")
try:
    # Use a smaller model for better compatibility
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    print("Summarization model loaded!")
except Exception as e:
    print(f"Error loading standard summarization model: {str(e)}")
    print("Trying with a smaller model...")
    try:
        # If the large model fails, try with a smaller one
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=-1)
        print("Smaller summarization model loaded as fallback!")
    except Exception as e2:
        print(f"Error loading fallback summarization model: {str(e2)}")
        # Define a simple function that just returns the input as a last resort
        def dummy_summarize(text, **kwargs):
            words = text.split()
            if len(words) > 50:
                return [{'summary_text': ' '.join(words[:50]) + '...'}]
            return [{'summary_text': text}]
        
        summarizer = dummy_summarize
        print("Using basic text truncation as summarization fallback!")

def transcribe_and_summarize(audio_file):
    start_time = time.time()
    global model  # Use global instead of nonlocal
    
    # Step 1: Check if audio file was uploaded
    if audio_file is None:
        return "Please upload an audio file.", "", 0
    
    # Step 2: Convert audio if necessary based on the available tool
    processed_audio = audio_file
    if audio_tool != "none" and audio_tool != "ffmpeg":
        try:
            # Create temp file for processed audio
            temp_dir = tempfile.gettempdir()
            processed_audio = os.path.join(temp_dir, "processed_audio.wav")
            
            if audio_tool == "libav":
                # Use avconv (libav) to convert
                subprocess.run(['avconv', '-i', audio_file, '-ar', '16000', '-ac', '1', processed_audio], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elif audio_tool == "sox":
                # Use SoX to convert
                subprocess.run(['sox', audio_file, '-r', '16000', '-c', '1', processed_audio], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"Audio processed with {audio_tool}")
        except Exception as e:
            print(f"Warning: Could not process audio with {audio_tool}: {e}")
            # Fall back to original file
            processed_audio = audio_file
    
    # Step 3: Transcribe the audio using Whisper with error handling
    transcribe_start = time.time()
    try:
        if use_faster_whisper:
            # Using faster-whisper
            segments, info = model.transcribe(processed_audio, beam_size=5)
            transcript = " ".join([segment.text for segment in segments])
        else:
            # Using standard whisper with additional error handling
            try:
                result = model.transcribe(processed_audio)
                transcript = result["text"]
            except RuntimeError as e:
                if "cuda" in str(e).lower() or "mps" in str(e).lower():
                    print("GPU error detected. Forcing CPU mode and retrying...")
                    # Force CPU mode if GPU fails
                    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                    # Reload model on CPU if needed
                    model = whisper.load_model("tiny", device="cpu")
                    result = model.transcribe(processed_audio)
                    transcript = result["text"]
                else:
                    raise
        
        if not transcript.strip():
            return "No speech detected in the audio file.", "", 0
            
        transcribe_time = time.time() - transcribe_start
        print(f"Transcription completed in {transcribe_time:.2f} seconds")
    except Exception as e:
        return f"Error transcribing audio: {str(e)}", "", 0
        
    # Clean up temp file if created
    if processed_audio != audio_file and os.path.exists(processed_audio):
        try:
            os.remove(processed_audio)
        except:
            pass
    
    # Step 3: Summarize the transcript
    summarize_start = time.time()
    
    # Function to chunk text for summarization
    def chunk_text(text, max_length=1000):
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(" ".join(current_chunk)) > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    # Process chunks and summarize
    chunks = chunk_text(transcript)
    chunk_summaries = []
    
    for chunk in chunks:
        # Skip empty chunks
        if not chunk.strip():
            continue
            
        try:
            # Summarize the chunk
            summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
            chunk_summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Error summarizing chunk: {str(e)}")
            # If summarization fails, include original chunk
            chunk_summaries.append(chunk[:100] + "...")
    
    # Combine chunk summaries
    summary = " ".join(chunk_summaries)
    
    # If we have multiple chunks, summarize again for coherence
    if len(chunk_summaries) > 1:
        try:
            final_summary = summarizer(" ".join(chunk_summaries), max_length=100, min_length=30, do_sample=False)
            summary = final_summary[0]['summary_text']
        except Exception as e:
            print(f"Error in final summarization: {str(e)}")
            # Keep existing summary if final summarization fails
        
    summarize_time = time.time() - summarize_start
    
    # Calculate total processing time
    total_time = time.time() - start_time
    
    return transcript, summary, round(total_time, 2)

# Create Gradio interface
with gr.Blocks(title="Audio Transcription & Summarization") as demo:
    gr.Markdown("# Audio Transcription & Summarization Tool")
    gr.Markdown("Upload an audio file to transcribe and summarize its content.")
    
    # Add device and audio tool info
    if use_faster_whisper:
        gr.Markdown(f"**Running on:** {platform.processor()} with faster-whisper on CPU")
    else:
        gr.Markdown(f"**Running on:** {platform.processor()} using standard whisper on CPU")
    gr.Markdown(f"**Audio processing:** Using {audio_tool}")
    
    # Add error message display
    error_output = gr.Textbox(label="Status", visible=True)
    
    with gr.Row():
        with gr.Column():
            # Changed to allow both microphone recording and file upload
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or Upload Audio")
            submit_btn = gr.Button("Transcribe & Summarize", variant="primary")
            processing_time = gr.Number(label="Processing Time (seconds)", precision=2)
        
        with gr.Column():
            transcript_output = gr.Textbox(label="Transcript", lines=10)
            summary_output = gr.Textbox(label="Summary", lines=5)
    
    # Function to handle errors
    def process_with_error_handling(audio_file):
        try:
            transcript, summary, time_taken = transcribe_and_summarize(audio_file)
            return "", transcript, summary, time_taken
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if "ffmpeg" in str(e).lower() or "audio" in str(e).lower():
                error_msg += "\n\nPlease install an audio processing tool:\n"
                error_msg += "- ffmpeg: brew install ffmpeg\n"
                error_msg += "- libav: brew install libav\n" 
                error_msg += "- SoX: brew install sox"
            return error_msg, "", "", 0
    
    submit_btn.click(
        fn=process_with_error_handling,
        inputs=audio_input,
        outputs=[error_output, transcript_output, summary_output, processing_time]
    )
    
    gr.Markdown("### How to Use")
    gr.Markdown("""
    1. Upload an audio file (mp3, wav, m4a, etc.)
    2. Click the 'Transcribe & Summarize' button
    3. Wait for processing (time depends on audio length)
    4. View the transcript and summary
    
    This app uses OpenAI's Whisper (small model) for transcription and BART-large-CNN for summarization.
    """)

# Setup function
def setup():
    """Install necessary dependencies before launching the app"""
    # Check for audio processing tools
    global audio_tool
    
    print("\nChecking for audio processing tools...")
    if audio_tool == "none":
        print("\nNO AUDIO PROCESSING TOOLS FOUND")
        print("This application works best with one of these tools:")
        print("- ffmpeg (recommended): brew install ffmpeg")
        print("- libav: brew install libav")
        print("- SoX: brew install sox")
        
        # Ask if user wants to continue anyway
        response = input("\nContinue anyway? Some functionality may be limited. (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print(f"Found {audio_tool} for audio processing.")

# Launch the app
if __name__ == "__main__":
    setup()
    print("\ntarting Voice-to-Text Application...")
    print(f"Running on {platform.processor()} with {device.upper()} acceleration")
    print("Loading models (this may take a moment)...")
    print("Once loaded, the application will be available at http://127.0.0.1:7860")
    demo.launch(share=False)  # Set share=True if you want a public link
