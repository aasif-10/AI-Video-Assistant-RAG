from langchain_core.prompts import ChatPromptTemplate
from utils.model import get_model
from langchain_core.output_parsers import StrOutputParser
from utils.rag_utils import get_relevant_chunks

from dotenv import load_dotenv
load_dotenv()

def video_assist_pipeline(text_chunks : str, query : str, video_id : str) -> str:
    model = get_model()

    prompt_template = ChatPromptTemplate.from_messages([
        (
        "system",
        """
You are an expert AI assistant. Your task is to accurately answer the user's query based strictly on the provided transcript.

Instructions:
- Use only the provided transcript to answer the query.
- Do not invent information that is not present in the transcript.
- If the transcript does not contain the answer, state that clearly.
- Do not mention the answer was refered from transcript - but mention it was refered from provided video
"""
    ),
        (
        "human",
        """
Transcript Context:
{content}

User Query:
{query}
"""
    )
    ])

    output_parser = StrOutputParser()

    chain = prompt_template | model | output_parser

    content = get_relevant_chunks(text_chunks, query, video_id)

    response = chain.invoke({
        "content" : content,
        "query" : query
    })
    
    return response