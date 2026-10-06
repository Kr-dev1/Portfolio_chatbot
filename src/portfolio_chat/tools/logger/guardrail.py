from sqlmodel import Session

from ..db.db import engine
from .models import GuardrailTrigger


def guardrail_logger(user_message: str, reason: str):
    guardrail_logs = GuardrailTrigger(reason=reason, user_message=user_message)
    with Session(engine) as session:
        session.add(guardrail_logs)
        session.commit()
