from utils.audio_processor import process_input
import os

from dotenv import load_dotenv
load_dotenv()

# Add ffmpeg to the system PATH so Whisper can find it
os.environ["PATH"] += os.pathsep + r"D:\ffmpeg\ffmpeg-8.1.1-full_build\bin"

import whisper

WHISPER_MODEL = os.getenv("WHISPER_MODEL","small")

_model = None

def load_model():
    global _model

    if _model == None:
        print("Loading model\n\n")
        # Save whisper downloads to D drive explicitly
        _model = whisper.load_model(WHISPER_MODEL, download_root=r"D:\whisper_models")
        print("Model loaded\n\n")

    return _model

def transcribe_chunk(model, chunk_path : str, translate : bool = False) -> str:
 
    task = "translate" if translate else "transcribe"

    result = model.transcribe(chunk_path, task=task)

    return result['text']

def transcribe_all(chunks : list, translate : bool = False) -> str:

    model = load_model()

    full_transcript = ""

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}\n\n")
        text = transcribe_chunk(model, chunk, translate=translate)
        full_transcript += text + " "

    print("Transcription completed")
    return full_transcript.strip()

def get_transcript(source : str) -> str:
    chunks, video_id = process_input(source=source)

    transcript = transcribe_all(chunks,True)

    return transcript, video_id