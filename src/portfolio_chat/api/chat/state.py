from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class State(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(
        default_factory=list
    )
    conversation_summary: str | None = ""
    context: list[Document] = []

    response: str = ""
    query: str = ""
    user_input: str = ""
    intent: str = ""
    calendar_action: str =""

    is_on_topic: bool = False
    reason: str = " "
