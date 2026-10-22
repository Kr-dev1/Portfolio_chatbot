from googleapiclient.errors import HttpError
from sqlmodel import Session, select

from portfolio_chat.api.chat.schema.schema import calendar_id
from portfolio_chat.scripts.google_calendar import service
from portfolio_chat.tools.db.db import engine
from portfolio_chat.tools.requests.model import ApprovalRequests, status


def delete_event(req_id, gcal_id):
    if gcal_id:
        try:
            service.events().delete(
                calendarId=calendar_id,
                eventId=gcal_id,
                sendUpdates="all",
            ).execute()
        except HttpError as e:
            if e.resp.status not in (404, 410):
                raise ValueError("Failed to delete Google event") from e

    with Session(engine) as session:
        request = session.exec(
            select(ApprovalRequests).where(ApprovalRequests.req_id == req_id)
        ).first()
        if not request:
            raise ValueError(f"No request found for {req_id}")

        request.status = status.DECLINED
        session.add(request)
        session.commit()
