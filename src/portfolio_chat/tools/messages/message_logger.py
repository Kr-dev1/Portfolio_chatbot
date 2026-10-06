from sqlmodel import Session

from portfolio_chat.tools.db.db import engine, init_db

from .model import Messages, Role


async def message_logger(thread_id: str, message: str, role: Role):
    init_db()
    message_logs = Messages(role=role, message=message, thread_id=thread_id)
    with Session(engine) as session:
        session.add(message_logs)
        session.commit()
