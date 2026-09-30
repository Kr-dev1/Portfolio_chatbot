from sqlmodel import Session

from ..db.db import engine
from .models import AIusage


def usage_logger(
    guard_rail_model: str,
    chat_model: str,
    time_to_first_token: float,
    total_time: float,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    reasoning_tokens: float,
    guardrail_triggered: bool,
):
    logs = AIusage(
        chat_model=chat_model,
        guard_rail_model=guard_rail_model,
        time_to_first_token=time_to_first_token,
        total_time=total_time,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        guardrail_triggered=guardrail_triggered
    )

    with Session(engine) as session:
        session.add(logs)
        session.commit()
