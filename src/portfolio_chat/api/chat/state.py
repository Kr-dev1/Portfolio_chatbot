from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class State(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = []
    context: list[Document] = []

    response: str = ""
    query: str = ""
    user_input: str = ""

    is_on_topic: bool = False
    reason: str = " "
