from datetime import datetime
from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class State(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    conversation_summary: str | None = None
    context: list[Document] = Field(default_factory=list)

    response: str = ""
    query: str = ""
    user_input: str = ""
    intent: str = ""
    calendar_action: str | None = None

    is_on_topic: bool = False
    reason: str = ""

    name: str | None = None
    email: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    attendees: list[str] = Field(default_factory=list)
    calendar_events: list | None = None
    calendar_available: bool | None = None
