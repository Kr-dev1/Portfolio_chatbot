from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.messages import RemoveMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from portfolio_chat.api.chat.state import State
from portfolio_chat.config import (
    GEMINI_API_KEY,
    GROQ_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
)
from portfolio_chat.scripts.google_calendar import service
from portfolio_chat.tools.requests.request import get_events_time

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
    llm = init_chat_model(model="groq:openai/gpt-oss-120b", api_key=GROQ_API_KEY)

    prompt = f"""Summarize the conversation into compact memory for a future LLM.

    Preserve only information that could be useful later, including:
    - Original Intention
    - Emails any time and date mentioned
    - Details that can be usefull

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


MAX_MESSAGES = 40
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


def check_events(start_time: datetime, end_time: datetime):
    pending_events = []
    events = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=start_time.isoformat(),
            timeMax=end_time.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    if start_time and end_time:
        pending_events_check = get_events_time(start_time, end_time)
        if pending_events_check:
            for event in pending_events_check:
                pending_events.append({"start_time":event.start_date_time, "end_time": event.end_date_time})
    return {"google_calendar": events["items"], "pending_requests": pending_events}
