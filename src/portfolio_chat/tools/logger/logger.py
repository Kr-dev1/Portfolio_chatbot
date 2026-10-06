from sqlmodel import Session

from ..db.db import engine
from .models import AIusage


def usage_logger(
    guard_rail_model: str,
    chat_model: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    reasoning_tokens: int,
    guardrail_triggered: bool,
):
    logs = AIusage(
        chat_model=chat_model,
        guard_rail_model=guard_rail_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        guardrail_triggered=guardrail_triggered,
    )

    with Session(engine) as session:
        session.add(logs)
        session.commit()
