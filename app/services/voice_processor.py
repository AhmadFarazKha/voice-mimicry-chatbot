import librosa
import torch
import numpy as np

class VoiceProcessor:
    def __init__(self):
        self.sample_rate = 22050
        
    def load_audio(self, file_path):
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        return audio
        
    def extract_features(self, audio):
        # Extract mel spectrogram features
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_db