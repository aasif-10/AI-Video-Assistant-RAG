from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
import os

def split_transcript(transcript : str) -> list :
    print("\n\nSplitting transcript\n\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, 
        chunk_overlap=200
    )
    text_chunks = text_splitter.split_text(transcript)
    return text_chunks

def get_vector_store(text_chunks : str, embedding_model, video_id : str):
    

    if not os.path.exists("chroma_db"):
        vector_store = Chroma.from_texts(
            texts=text_chunks,
            collection_name=video_id,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

    else:
        print("\n\nCreating and storing transcript chunks in vector\n\n")

        vector_store = Chroma(
            collection_name=video_id,
            persist_directory="chroma_db",
            embedding_function=embedding_model
        )

        if vector_store._collection.count()==0:
            print("\n\nStoring transcript chunks in vector\n\n")
            vector_store.add_texts(text_chunks)
    
    return vector_store

def get_retriever(vector_store):
    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {
            "k": 4,
            "fetch_k": 10,
            "lambda_mult":0.5 # diversity 0 - 1
        }
    )

    return retriever

def get_relevant_chunks(text_chunks : list, query : str, video_id : str) -> str:

    embedding_model = MistralAIEmbeddings(model="mistral-embed")

    vector_store = get_vector_store(text_chunks, embedding_model, video_id )

    retriever = get_retriever(vector_store)

    print("\n\nRetrieving relevant chunks\n\n")

    chunks = retriever.invoke(query)
    
    content = "\n\n".join([chunk.page_content for chunk in chunks])

    return content

