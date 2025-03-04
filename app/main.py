from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from gtts import gTTS
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Voice Mimicry Chatbot! Visit /gradio to use the app."}

def generate_voice(audio, text):
    if not text:
        return "Please enter some text.", None, None

    # For now, the uploaded audio is not used (we are NOT doing voice cloning here)
    # In a real voice cloning system, you'd process `audio` to extract the speaker's characteristics
    tts = gTTS(text=text, lang="en")
    
    # Save TTS output to MP3 file
    temp_dir = tempfile.gettempdir()
    output_file = os.path.join(temp_dir, "output_voice.mp3")
    tts.save(output_file)

    # Return success message, playable audio, and download link
    return "Voice generated successfully!", output_file, output_file

with gr.Blocks() as demo:
    gr.Markdown("## Voice Mimicry - (Basic Version - No Real Cloning Yet)")
    with gr.Row():
        audio_input = gr.Audio(sources=["upload"], type="filepath", label="Upload Sample Voice")
        text_input = gr.Textbox(label="Enter Text to Speak")
    with gr.Row():
        generate_button = gr.Button("Generate Voice")
    output_message = gr.Textbox(label="Status", interactive=False)
    audio_player = gr.Audio(label="Generated Voice", interactive=False)
    download_link = gr.File(label="Download MP3 File")

    generate_button.click(generate_voice, inputs=[audio_input, text_input], outputs=[output_message, audio_player, download_link])

app = gr.mount_gradio_app(app, demo, path="/gradio")
