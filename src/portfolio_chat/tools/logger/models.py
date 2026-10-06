import uuid

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

class GuardrailTrigger(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_message: str
    reason: str
