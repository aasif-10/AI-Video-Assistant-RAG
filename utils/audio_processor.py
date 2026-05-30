import yt_dlp
import os
import librosa
import soundfile as sf

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR,exist_ok=True)

def download_youtube_video(url : str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR,"%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "ffmpeg_location": r"D:\ffmpeg\ffmpeg-8.1.1-full_build\bin",

        "outtmpl": output_path,

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],

        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url=url,download=True)

        filename = ydl.prepare_filename(info).replace(".webm",".wav").replace(".m4a",".wav")

        return filename


def convert_to_wav(input_path : str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio, sr = librosa.load(input_path, sr=16000, mono=True)

    sf.write(output_path,audio,sr)

    return output_path


def chunk_audio(wav_path : str, chunk_minutes : int = 10) -> list:
    audio, sr = librosa.load(wav_path, sr=16000, mono=True)

    chunk_ms = int(chunk_minutes * 60 * sr)

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        sf.write(chunk_path,chunk,sr)

        chunks.append(chunk_path)

    return chunks


def process_input(source : str):
    if source.startswith("https://www.youtube.com") or source.startswith("https://youtu.be"):
        
        print("\n\n\nDetected youtube URL, downloading audio!\n\n")
        
        wav_path = download_youtube_video(source)

        print("\n\nVideo downloaded!!\n\n")

        video_id = source.split("/")[-1].split("?")[0]
        
    else:
        print("Detected local file, converting to WAV!\n\n\n")
        wav_path = convert_to_wav(source)

        video_id = os.path.basename(source)

    print("Chunking audio\n\n")    
    chunks = chunk_audio(wav_path)
    print(f"Audio is ready, {len(chunks)} chunks created\n\n")

    return chunks, video_id
