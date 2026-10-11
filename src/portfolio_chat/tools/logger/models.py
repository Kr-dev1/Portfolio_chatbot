import uuid
from datetime import datetime
from enum import Enum
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlmodel import Field, SQLModel


class AIusage(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    guard_rail_model: str
    chat_model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int
    guardrail_triggered: bool
    node: str


class GuardrailTrigger(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_message: str
    reason: str


def get_ist_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Kolkata"))


class status(str, Enum):
    APPROVED = "approved"
    PENDING = "pending"
    DECLINED = "declined"


class ApprovalRequests(SQLModel, table=True):
    req_id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    thread_id: str
    name: str
    start_date_time: datetime
    end_date_time: datetime
    created_at: datetime = Field(default_factory=get_ist_now)
    email: str
    event_description: str
    status: status
