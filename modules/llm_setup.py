
from langchain_groq import ChatGroq

def initialize_llm(api_key: str, model_name: str = "gemma-7b-it"):  
    return ChatGroq(
        groq_api_key=api_key,
        model=model_name
    )
