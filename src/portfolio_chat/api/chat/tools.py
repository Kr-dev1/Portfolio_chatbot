from datetime import datetime

from langchain_core.tools import tool

from portfolio_chat.scripts.google_calendar import service

time_zone: str = "Asia/Kolkata"
order_by: str = "startTime"
calendar_id: str = "primary"
single_events: bool = True


@tool
def create_event(
    summary: str,
    description: str | None,
    start_date_time: datetime,
    end_date_time: datetime,
    attendees: list,
):
    """Create event on google calendar this could be for both freelance or work"""
    event = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_date_time.isoformat(), "timeZone": time_zone},
        "end": {"dateTime": end_date_time.isoformat(), "timeZone": time_zone},
        "reminders": {
            "useDefault": True,
            "overrides": [
                {"method": "email", "minutes": 60},
                {"method": "popup", "minutes": 10},
            ],
        },
        "attendees": [{"email": email} for email in attendees],
    }
    created_event = service.events().insert(calendarId=calendar_id, body=event).execute()
    return created_event


@tool
def get_events(
    start_date_time: datetime,
    end_date_time: datetime,
):
    """Check availavility and to get event info to edit them if required"""
    event_results = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=start_date_time.isoformat(),
            timeMax=end_date_time.isoformat(),
            singleEvents=single_events,
            orderBy=order_by,
        )
        .execute()
    )
    return event_results


@tool
def update_event(
    event_id: str,
    updated_fields: dict,
):
    """Partially update an event by its event_id."""
    patched_event = (
        service.events()
        .patch(calendarId=calendar_id, eventId=event_id, body=updated_fields)
        .execute()
    )
    return patched_event


@tool
def delete_event(event_id: str):
    """
    Delete an event completely using its event_id.
    """
    service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
    return f"Event {event_id} successfully deleted."

tools = [create_event, update_event, delete_event, get_events]
