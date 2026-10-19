from sqlmodel import Session
from starlette.concurrency import run_in_threadpool

from portfolio_chat.tools.db.db import engine, init_db

from .model import Messages, Role


def _write_message(thread_id: str, message: str, role: Role, node: str):
    message_logs = Messages(role=role, message=message, thread_id=thread_id, node=node)

    with Session(engine) as session:
        session.add(message_logs)
        session.commit()


async def message_logger(thread_id: str, message: str, role: Role, node: str):
    init_db()
    await run_in_threadpool(_write_message, thread_id, message, role, node)
