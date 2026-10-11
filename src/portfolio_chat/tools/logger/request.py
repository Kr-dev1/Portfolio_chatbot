from sqlmodel import Session, select

from portfolio_chat.tools.db.db import engine
from portfolio_chat.tools.logger.models import ApprovalRequests, status


def create_req(**data) -> ApprovalRequests:
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
