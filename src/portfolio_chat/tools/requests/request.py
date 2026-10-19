from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlmodel import Session, col, select

from portfolio_chat.tools.db.db import engine, init_db
from portfolio_chat.tools.requests.model import ApprovalRequests, status


def create_req(**data) -> ApprovalRequests:
    init_db()
    with Session(engine) as session:
        existing = session.get(ApprovalRequests, data["req_id"])

        if existing:
            return existing

        request = ApprovalRequests(**data)
        session.add(request)
        session.commit()
        session.refresh(request)

        return request


def update_req(req_id: str, approval: status):
    init_db()
    if not id or not req_id:
        raise ValueError("Fields Missing")

    with Session(engine) as session:
        statement = select(ApprovalRequests).where(ApprovalRequests.req_id == req_id)
        request = session.exec(statement).first()

        if request is None:
            raise ValueError(f"Request {req_id} not found")

        request.status = approval

        session.add(request)
        session.commit()
        session.refresh(request)

        return request


def check_db_for_event(
    email: str | None,
    start_time: datetime | None,
) -> ApprovalRequests | None:
    if not email or not start_time:
        raise ValueError("Email and start time is required")

    tz = ZoneInfo("Asia/Kolkata")
    day_start = start_time.astimezone(tz).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    day_end = day_start + timedelta(days=1)

    with Session(engine) as session:
        statement = (
            select(ApprovalRequests)
            .where(
                ApprovalRequests.email == email,
                col(ApprovalRequests.status).in_([status.APPROVED, status.PENDING]),
                col(ApprovalRequests.start_date_time) >= day_start,
                col(ApprovalRequests.start_date_time) < day_end,
            )
            .order_by(col(ApprovalRequests.start_date_time))
        )

        return session.exec(statement).first()


def get_events_email(email: str | None):
    with Session(engine) as session:
        statement = (
            select(ApprovalRequests)
            .where(
                ApprovalRequests.email == email,
                col(ApprovalRequests.status).in_([status.APPROVED, status.PENDING]),
            )
            .order_by(col(ApprovalRequests.start_date_time))
        )
        return session.exec(statement).all()


def get_events_time(start_time: datetime, end_time: datetime):
    with Session(engine) as session:
        statement = (
            select(ApprovalRequests)
            .where(
                (ApprovalRequests.start_date_time > start_time),
                (ApprovalRequests.end_date_time < end_time),
                col(ApprovalRequests.status).in_([status.APPROVED, status.PENDING]),
            )
            .order_by(col(ApprovalRequests.start_date_time))
        )

        return session.exec(statement).all()
