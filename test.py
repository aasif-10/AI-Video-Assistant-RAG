from utils.audio_processor import process_input
from core.transcriber import transcribe_all

source = "https://youtu.be/vJ09OdAqDTk?si=hBiBqs92UXZWXIRV"

chunks = process_input(source=source)

transcript = transcribe_all(chunks,translate=True)