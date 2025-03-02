import os
import subprocess
import librosa

# Set FFmpeg path explicitly
FFMPEG_PATH = "C:\\ffmpeg\\bin"  # Adjust this to your actual FFmpeg path

def initialize_ffmpeg():
    """Ensure FFmpeg is available in the PATH"""
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ["PATH"]
    
    # Verify installation
    try:
        subprocess.run([os.path.join(FFMPEG_PATH, "ffmpeg"), "-version"], 
                       capture_output=True, check=True)
        subprocess.run([os.path.join(FFMPEG_PATH, "ffprobe"), "-version"], 
                       capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def load_audio(file_path, sr=22050):
    """Load audio with explicit FFmpeg path"""
    initialize_ffmpeg()
    return librosa.load(file_path, sr=sr)