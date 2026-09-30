import os

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()
QDRANT_API_KEY = str(os.environ["QDRANT_API_KEY"])
QDRANT_URL = str(os.environ["QDRANT_URL"])
GEMINI_API_KEY = SecretStr(os.environ["GEMINI_API_KEY"])
OPENAI_API_KEY = SecretStr(os.environ["OPENAI_API_KEY"])
GROQ_API_KEY = SecretStr(os.environ["GROQ_API_KEY"])
TAVILY_API_KEY = SecretStr(os.environ["TAVILY_API_KEY"])
DEEPSEEK_API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])
ANTHROPIC_API_KEY = SecretStr(os.environ["ANTHROPIC_API_KEY"])
NEONDB_URL = str(os.environ["NEONDB_URL"])
