from langchain.chat_models import init_chat_model
from langchain_core.messages import RemoveMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from portfolio_chat.api.chat.state import State
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

retriever = vector_store.as_retriever(search_kwargs={"k": 5})


def conversation_summariser(messages: list, existing_summary: str | None):
    llm = init_chat_model(model="gemini-3.1-flash-lite", api_key=GEMINI_API_KEY)

    prompt = f"""Summarize the conversation into compact memory for a future LLM.

    Preserve only information that could be useful later, including:
    - Important facts the user provided
    - Preferences and constraints
    - Decisions and conclusions
    - Ongoing tasks or plans
    - Dates, times, events, and commitments
    - Important technical context, errors, and solutions
    - Unresolved questions or pending decisions
    - Previous questions or answers that may matter later

    Rules:
    - Never invent or infer information.
    - Preserve exact names, dates, times, numbers, and technical details when relevant.
    - Preserve uncertainty or conflicting information.
    - Remove greetings, small talk, repetition, and irrelevant details.
    - Do not preserve exact wording unless it is important.
    - Prefer concise facts over explanations.

    Return only the summary. If there is nothing useful to retain, return an empty summary.

    Messages:
    {messages[:-4]}
    """

    result = llm.invoke(prompt)
    return result.content


MAX_MESSAGES = 10
KEEP_MESSAGES = 4


async def compact_conversation(state: State) -> dict:
    if len(state.messages) < MAX_MESSAGES:
        return {}

    messages_to_summarize = state.messages[:-KEEP_MESSAGES]
    recent_messages = state.messages[-KEEP_MESSAGES:]

    summary = conversation_summariser(
        messages=messages_to_summarize,
        existing_summary=state.conversation_summary,
    )

    return {
        "conversation_summary": summary,
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *recent_messages,
        ],
    }
