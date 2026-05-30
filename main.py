from core.assist_pipeline import video_assist_pipeline
from core.transcriber import get_transcript
from utils.rag_utils import split_transcript

if __name__ == "__main__":
    print("\n\nAI Video Assistant\n")
    
    text_chunks = None
    transcript = None
    video_id = None

    while True:
        source = input("Enter the source or upload local file of audio/video: ")
        query = input("Enter query: ")

        if(query.lower() == 'exit'):
            print("\n\nExiting...!")
            break

        if(source):
            transcript, video_id = get_transcript(source)

            text_chunks = split_transcript(transcript)

            content = video_assist_pipeline(text_chunks, query, video_id)
        
            print(content)

        else:
            content = video_assist_pipeline(text_chunks, query, video_id)

            print(content)

