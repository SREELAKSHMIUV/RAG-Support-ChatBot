import os
from .groq_llm import GroqLLM

def get_llm():

    provider = os.getenv("LLM_PROVIDER")

    if provider == "groq":
        return GroqLLM()

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")