from datetime import datetime
from typing import Annotated, Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    thread_id: str | None = None
    req_id: str | None = None

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
    update_email: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    attendees: list[str] = Field(default_factory=list)
    calendar_events: dict | None = None
    calendar_available: bool | None = None
    event_description: str | None = None
    booking_approved: bool | None = None
    event_exists: bool | None = None
    selected_event_id: str | None = None
    retrived_events: list[Any] | None = None
    selected_event: Any | None = None
    update_values: list[dict] | None = None
    current_calendar_flow: str | None = None
