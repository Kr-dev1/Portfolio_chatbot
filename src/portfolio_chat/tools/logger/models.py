import uuid

from sqlmodel import Field, SQLModel


class AIusage(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    guard_rail_model: str
    chat_model: str
    time_to_first_token: float
    total_time: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: float | None
    guardrail_triggered: bool

class GuardrailTrigger(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_message: str
    reason: str
