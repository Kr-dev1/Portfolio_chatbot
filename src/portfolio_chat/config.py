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
CALENDAR_API_KEY= SecretStr(os.environ["CALENDAR_API_KEY"])
GOOGLE_REFRESH_TOKEN= str(os.environ["GOOGLE_REFRESH_TOKEN"])
GOOGLE_CLIENT_ID= str(os.environ["GOOGLE_CLIENT_ID"])
GOOGLE_CLIENT_SECRET=str(os.environ["GOOGLE_CLIENT_SECRET"])
