from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from portfolio_chat.config import GEMINI_API_KEY, QDRANT_API_KEY, QDRANT_URL

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    api_key=GEMINI_API_KEY,
)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="resume",
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
