from datetime import datetime
from typing import Annotated, Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from portfolio_chat.tools.requests.model import ApprovalRequests


class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    thread_id: str | None = None
    req_id: str | None = None
    event_id: str | None = None

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
    calendar_events: dict | list[Any] | None = None
    calendar_available: bool | None = None
    event_description: str | None = None
    booking_approved: bool | None = None
    event_exists: bool | None = None
    selected_event_id: str | None = None
    retrived_events: list[Any] = Field(default_factory=list)
    selected_event: ApprovalRequests | None = None
    update_values: list[dict] | None = None
    current_calendar_flow: str | None = None
    existing_request_declined: bool | None = None
    done_editing: bool = False
    cancel_edit: bool = False
