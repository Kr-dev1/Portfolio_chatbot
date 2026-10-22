from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlmodel import Session, col, select

from portfolio_chat.tools.db.db import engine, init_db
from portfolio_chat.tools.requests.model import (
    ApprovalRequests,
    UpdateRequestResult,
    status,
)


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


def update_exisitng_req(req_id: str, approval: status, gcal_id: str | None):
    if not req_id or not approval:
        raise ValueError("Fields Missing")

    with Session(engine) as session:
        statement = select(ApprovalRequests).where(ApprovalRequests.req_id == req_id)
        request = session.exec(statement).first()

        if request is None:
            raise ValueError(f"Request {req_id} not found")

        request.status = approval
        request.gca_id = gcal_id

        session.add(request)
        session.commit()
        session.refresh(request)

        return request


def check_db_for_event(
    email: str | None,
    start_time: datetime | None,
) -> ApprovalRequests | None:
    init_db()
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
    init_db()
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
                (ApprovalRequests.start_date_time < end_time),
                (ApprovalRequests.end_date_time > start_time),
                col(ApprovalRequests.status).in_([status.APPROVED, status.PENDING]),
            )
            .order_by(col(ApprovalRequests.start_date_time))
        )

        return session.exec(statement).all()


def create_update_req(req_id: str) -> UpdateRequestResult:
    with Session(engine) as session:
        statement = select(ApprovalRequests).where(ApprovalRequests.req_id == req_id)
        request = session.exec(statement).first()

        if request and request.status == status.PENDING:
            request.status = status.DECLINED
            session.add(request)
            session.commit()
            return UpdateRequestResult(event_declined=True, request=None)

        if request and request.status == status.APPROVED:
            return UpdateRequestResult(event_declined=False, request=request)

        raise ValueError(f"Request {req_id} not found or has an unsupported status")


def get_event_status(req_id: str, thread_id: str):
    with Session(engine) as session:
        statement = (
            select(ApprovalRequests)
            .where(ApprovalRequests.req_id == req_id)
            .where(ApprovalRequests.thread_id == thread_id)
            .where(col(ApprovalRequests.status).in_([status.PENDING]))
        )
        request = session.exec(statement).first()

        if not request:
            return {"found_event": False}
        return {"found_event": True}


def get_event_by_req_id(req_id: str):
    with Session(engine) as session:
        statement = select(ApprovalRequests).where(ApprovalRequests.req_id == req_id)
        request = session.exec(statement).first()

        if not request:
            return None
        return request
