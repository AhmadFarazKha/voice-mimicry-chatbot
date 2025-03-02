from fastapi import FastAPI, BackgroundTasks
import gradio as gr
from pydub import AudioSegment
import tempfile
import os
import subprocess
import torch
import torchaudio
import time
import gc
from tortoise.api import TextToSpeech, MODELS_DIR
from tortoise.utils.audio import load_audio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Mimicry Chatbot")

# Cache for storing processed results
results_cache = {}

# Global settings for optimization
MAX_TEXT_LENGTH = 100  # Severely limit text for faster processing
MAX_AUDIO_SECONDS = 5  # Use only first 5 seconds of reference audio
SAMPLE_RATE = 22050
USE_AUTOCAST = True  # Use mixed precision for faster computation

# Force half precision for ALL models to reduce memory usage
os.environ["TORTOISE_MODELS_DIR"] = MODELS_DIR
os.environ["TORTOISE_HALF_PRECISION"] = "1"

# Status tracking
current_status = {"processing": False, "message": "Ready", "progress": 0}

def get_tts():
    """Optimized TTS model loader with minimal components"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Force garbage collection before loading model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Load with minimal config
        logger.info(f"Initializing TTS model on {device} with low resource settings")
        tts = TextToSpeech(
            device=device,
            autoregressive_batch_size=1,  # Process one token at a time (slower but less memory)
            low_vram=True,  # Enable low VRAM mode
            kv_cache=True   # Use KV caching for optimization
        )
        return tts
    except Exception as e:
        logger.error(f"Failed to initialize TTS: {e}")
        return None

def process_audio_file(file_path):
    """Process input audio with strict optimization"""
    try:
        # Handle different input formats
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            logger.info("Extracting audio from video file")
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            # Extract at lowest usable quality
            subprocess.run([
                "ffmpeg", "-i", file_path, "-vn", "-acodec", "pcm_s16le", 
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-t", str(MAX_AUDIO_SECONDS), temp_audio
            ], check=True, capture_output=True)
            audio_path = temp_audio
        elif ext in [".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"]:
            # Convert audio to minimal format
            logger.info("Processing audio file")
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
            # Limit length
            if len(audio) > MAX_AUDIO_SECONDS * 1000:
                audio = audio[:MAX_AUDIO_SECONDS * 1000]
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            audio.export(temp_wav, format="wav")
            audio_path = temp_wav
        else:
            logger.error(f"Unsupported file format: {ext}")
            return None
            
        return audio_path
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        return None

def voice_clone_async(file_path, text, background_tasks):
    """Non-blocking voice cloning function"""
    result_id = f"{time.time()}"
    results_cache[result_id] = {"status": "processing", "output": None}
    
    background_tasks.add_task(
        process_voice_clone, file_path, text, result_id
    )
    
    return {"result_id": result_id, "message": "Processing started"}

def process_voice_clone(file_path, text, result_id):
    """Actual processing function that runs in background"""
    try:
        output_path = None
        current_status["processing"] = True
        current_status["message"] = "Starting processing"
        current_status["progress"] = 10
        
        # 1. Input validation and limitation
        if not file_path or not text:
            results_cache[result_id] = {"status": "error", "output": "Missing input file or text"}
            return
        
        # 2. Limit text length strictly
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
            logger.info(f"Text truncated to {MAX_TEXT_LENGTH} characters")
        
        current_status["progress"] = 20
        current_status["message"] = "Processing audio file"
        
        # 3. Process audio file
        audio_path = process_audio_file(file_path)
        if not audio_path:
            results_cache[result_id] = {"status": "error", "output": "Failed to process audio file"}
            return
            
        current_status["progress"] = 30
        current_status["message"] = "Loading TTS model"
        
        # 4. Get TTS model
        tts = get_tts()
        if tts is None:
            results_cache[result_id] = {"status": "error", "output": "Failed to initialize TTS model"}
            return
        
        current_status["progress"] = 40
        current_status["message"] = "Loading reference audio"
        
        # 5. Load reference audio
        try:
            reference_audio = load_audio(audio_path, SAMPLE_RATE)
            # Further limit reference audio length in case the previous step didn't
            if reference_audio.shape[0] > MAX_AUDIO_SECONDS * SAMPLE_RATE:
                reference_audio = reference_audio[:MAX_AUDIO_SECONDS * SAMPLE_RATE]
                logger.info(f"Reference audio trimmed to {MAX_AUDIO_SECONDS} seconds")
        except Exception as e:
            logger.error(f"Error loading reference audio: {e}")
            results_cache[result_id] = {"status": "error", "output": f"Error loading reference audio: {str(e)}"}
            return
        
        current_status["progress"] = 50
        current_status["message"] = "Generating speech"
        
        # 6. Generate speech
        try:
            # Use autocast for mixed precision if available (faster)
            context = torch.cuda.amp.autocast() if USE_AUTOCAST and torch.cuda.is_available() else nullcontext()
            
            with torch.inference_mode(), context:
                gen_audio = tts.tts_with_preset(
                    text=text,
                    voice_samples=[reference_audio],
                    preset="ultra_fast",  # Use fastest preset
                    conditioning_latents=None,  # Skip computing new conditioning
                    use_deterministic_seed=42,  # Fixed seed
                    num_autoregressive_samples=1,  # Minimum samples
                    diffusion_iterations=10,  # Minimum diffusion steps (lower quality but faster)
                    cond_free=False  # Disable classifier-free guidance for speed
                )
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            results_cache[result_id] = {"status": "error", "output": f"Error generating speech: {str(e)}"}
            return
            
        current_status["progress"] = 80
        current_status["message"] = "Finalizing output"
        
        # 7. Save output
        try:
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            torchaudio.save(output_path, gen_audio.squeeze(0).cpu(), SAMPLE_RATE)
        except Exception as e:
            logger.error(f"Error saving output audio: {e}")
            results_cache[result_id] = {"status": "error", "output": f"Error saving output: {str(e)}"}
            return
            
        current_status["progress"] = 90
        current_status["message"] = "Cleaning up"
        
        # 8. Cleanup
        try:
            if audio_path != file_path:
                os.remove(audio_path)
        except Exception as e:
            logger.warning(f"Failed to remove temp file: {e}")
            
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 9. Set result
        results_cache[result_id] = {"status": "complete", "output": output_path}
        current_status["progress"] = 100
        current_status["message"] = "Complete"
    except Exception as e:
        logger.error(f"Error in voice cloning process: {e}")
        results_cache[result_id] = {"status": "error", "output": f"Processing error: {str(e)}"}
    finally:
        current_status["processing"] = False
        gc.collect()  # Force garbage collection

# Null context manager for when autocast isn't used
class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): pass

def check_status(result_id):
    """Check the status of a processing job"""
    if result_id not in results_cache:
        return {"status": "not_found"}
    
    result = results_cache[result_id]
    if result["status"] == "complete":
        # Return the actual audio file
        return result["output"]
    elif result["status"] == "error":
        return result["output"]  # Error message
    else:
        # Still processing
        return {"status": "processing", "progress": current_status["progress"], "message": current_status["message"]}

def get_current_status():
    """Get the current status for the progress indicator"""
    return current_status

# Gradio Interface with two-step process
with gr.Blocks(title="🎭 Voice Mimicry Chatbot") as iface:
    gr.Markdown("# 🎭 Voice Mimicry Chatbot")
    gr.Markdown("## Upload an audio/video file, enter text, and get the same voice speaking the text")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="🎤 Upload Audio/Video File (5 sec max recommended)")
            text_input = gr.Textbox(label="📝 Enter Text to Convert", 
                                   placeholder=f"Enter text (max {MAX_TEXT_LENGTH} chars for faster processing)")
            start_btn = gr.Button("Start Processing")
            
            status_display = gr.Textbox(label="Status", value="Ready", interactive=False)
            progress = gr.Slider(minimum=0, maximum=100, value=0, label="Progress", interactive=False)
            
            result_id_output = gr.Textbox(label="Result ID", visible=False)
            check_btn = gr.Button("Check Status")
        
        with gr.Column():
            audio_output = gr.Audio(label="🔊 Generated Speech")
    
    # Event handlers
    def start_processing(audio, text, background_tasks=BackgroundTasks()):
        """Start the processing and return a job ID"""
        if not audio or not text:
            return "Missing audio file or text", "error", None, 0
        
        result = voice_clone_async(audio, text, background_tasks)
        return (f"Processing started with ID: {result['result_id']}", 
                "processing", 
                result["result_id"],
                10)
    
    def update_status(result_id):
        """Update the status display"""
        if not result_id:
            return "No job ID provided", None, 0
            
        result = check_status(result_id)
        if isinstance(result, dict):
            if result.get("status") == "not_found":
                return "Job not found", None, 0
            elif result.get("status") == "processing":
                return f"Processing: {result.get('message', '')}", None, result.get("progress", 0)
            else:
                return f"Error: {result}", None, 0
        else:
            # Result is a file path
            return "Processing complete", result, 100
    
    start_btn.click(
        start_processing, 
        inputs=[audio_input, text_input],
        outputs=[status_display, result_id_output, progress]
    )
    
    check_btn.click(
        update_status,
        inputs=[result_id_output],
        outputs=[status_display, audio_output, progress]
    )
    
    # Add periodic refresh for progress updates
    status_timer = gr.Timer(2, lambda: get_current_status())
    status_timer.tick(lambda s: (s["message"], s["progress"]), 
                     inputs=[status_timer], 
                     outputs=[status_display, progress])

# Mount Gradio app into FastAPI with increased timeout
app = gr.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    import uvicorn
    # Increase timeout settings
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)