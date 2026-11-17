"""
load the model out from the .env and create the model instance
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os


class ModelError(Exception):
    """Raised when the model cannot be created from the environment."""


def model_from_env() -> ChatOpenAI:
    load_dotenv()

    model_name = os.environ.get("OPENAI_MODEL") or "gpt-6-luna"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ModelError("OPENAI_API_KEY is not set")
    url = os.environ.get("OPENAI_URL")
    model = ChatOpenAI(model=model_name, api_key=api_key, base_url=url, timeout=120)
    return model


