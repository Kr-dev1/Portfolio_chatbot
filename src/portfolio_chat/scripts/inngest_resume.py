from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter

from ..config import GEMINI_API_KEY, QDRANT_API_KEY, QDRANT_URL


def inngest_resume(file_path: Path | str):
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError("No file found in path")

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "section"),
            ("####", "role"),
            ("#####", "company"),
            ("######", "project_name"),
        ]
    )

    markdown = file_path.read_text(encoding="utf-8")
    docs = splitter.split_text(markdown)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview", api_key=GEMINI_API_KEY
    )

    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name="resume",
    )


current_dir = Path(__file__).parent
inngest_resume(current_dir / "../data/resume.md")
