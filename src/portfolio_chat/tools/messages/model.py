import uuid
from enum import Enum

from sqlmodel import Field, SQLModel


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Messages(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    thread_id: str
    role: Role
    message: str
