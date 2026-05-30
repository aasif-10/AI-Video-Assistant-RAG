from langchain_mistralai import ChatMistralAI

def get_model():
    model = ChatMistralAI(
        model_name="mistral-small-2506",
        temperature=0
    )
    return model