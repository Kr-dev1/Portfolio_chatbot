import uuid
from datetime import datetime, timedelta
from typing import cast
from zoneinfo import ZoneInfo

from googleapiclient.errors import HttpError
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from portfolio_chat.api.chat.schema.schema import (
    CalendarEnum,
    GetUserInfo,
    calendar_id,
    chat_model,
    classifier_model,
)
from portfolio_chat.api.chat.state import State
from portfolio_chat.config import DEEPSEEK_API_KEY, GROQ_API_KEY
from portfolio_chat.scripts.google_calendar import service
from portfolio_chat.tools.helpers.helper import check_events
from portfolio_chat.tools.requests.model import status
from portfolio_chat.tools.requests.request import (
    check_db_for_event,
    create_req,
    update_exisitng_req,
)


def extract_user_info(state: State):
    user_input = state.user_input
    llm = init_chat_model(
        model=classifier_model, model_provider="deepseek", api_key=DEEPSEEK_API_KEY
    ).with_structured_output(GetUserInfo)

    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    prompt = f"""
     Extract calendar information from the user's message.

     Current date and time:
     {now.isoformat()}

     Timezone:
     Asia/Kolkata

     User message:
     {user_input, state.messages, state.conversation_summary}

     Rules:
     - The calendar owner is Kasturi.
     - The person making the request is the requester.
     - If the user introduces themselves, extract their name into `name`.
     - Meeting with kasturi cannot be a meeting name unless specified by the user.
       Examples:
         "I am Pranav" -> name = "Pranav"
         "Hey, I'm John" -> name = "John"
         "This is Sarah" -> name = "Sarah"
     - If the user says they want to schedule a meeting "with Kasturi",
       Kasturi is the calendar owner and must NOT be extracted as `name`.
     - If there is one singular email it is always the persons email.
     - Do not put names into `attendees`.
     - If the exact time and date is not present, ex:- next week at 2pm donot extract.
     - `attendees` must contain only explicitly provided attendee email addresses.
     - Extract the requester's email when explicitly provided.
     - Resolve relative dates such as "today", "tomorrow", and "next Monday"
       using the current date above.
     - Extract explicit times such as "5pm", "5 PM", "17:00", etc.
     - If only a start time is provided, leave `end_time` empty.
     - Only extract information explicitly stated or directly implied by
       relative date/time expressions.
     """

    result = cast(GetUserInfo, llm.invoke(prompt))
    start_time = result.start_time or state.start_time

    if result.end_time:
        end_time = result.end_time
    elif result.start_time:
        end_time = result.start_time + timedelta(hours=1)
    else:
        end_time = state.end_time

    updates = {
        "name": result.name or state.name,
        "email": result.email or state.email,
        "attendees": result.attendees or state.attendees,
        "event_description": result.event_description or state.event_description,
        "current_calendar_flow": state.current_calendar_flow,
    }

    if start_time is not None:
        updates["start_time"] = start_time.isoformat()

    if end_time is not None:
        updates["end_time"] = end_time.isoformat()

    return updates


def get_availability_info(state: State):
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    today = now.strftime("%A, %d %B %Y %I:%M %p")
    if state.start_time:
        start_time = state.start_time

        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        else:
            start_time = start_time.astimezone(ZoneInfo("Asia/Kolkata"))

        if start_time <= now:
            response = (
                f"It is {today} IST, and the time you selected has already passed. "
                "Could you share another date and time that works for you? "
                "I'll check availability for that slot."
            )
            return {
                "messages": [AIMessage(content=response)],
                "response": response,
                "calendar_available": None,
            }
    missing = []
    if not state.start_time:
        missing.append("start_time_date")
    if not state.name and state.start_time:
        missing.append("name")
    if not state.event_description and state.start_time:
        missing.append("meeting_summary")
    if not state.email and state.start_time:
        missing.append("email")

    llm = init_chat_model(
        model=chat_model,
        api_key=GROQ_API_KEY,
    )
    prompt = ""
    if state.calendar_available is False:
        prompt = f"""
         The user wants to schedule a meeting, but the meeting slot is already booked during the same interval.
         Here are all the events from that day {state.calendar_events}.
         Ask the user for the date and time they would like to re-schedule the
         meeting since the original time they picked is already booked.

         Keep the response concise and conversational.
         Return only the message shown to the user."""
    elif not state.start_time:
        prompt = f"""
         Ask the user for their preferred date and time for a meeting with Kasturi.

         Mention that all meetings are scheduled in IST (UTC+5:30).

         Keep it to one short sentence or two brief sentences.
         Avoid filler like "Sure!" or "Could you let me know".
         Return only the message shown to the user.
         date and tiem for today is {today}

         <sample response>
         When would you like to meet with Kasturi? All meetings are scheduled in IST (UTC+5:30). The current time in India is {today} IST.
         </sampleresponse>
         """
    else:
        prompt = f"""
         This means there is a meeting slot available respond based on the user input {state.user_input}
         The user wants to schedule a meeting, but the {missing} is not available.
         Ask them for missing fields and also include are there any atendees.

         Do not ask for any additional information yet.
         Return only the message shown to the user.
         Request only the fields that are missing
         """

    response = llm.invoke(prompt)
    return {"messages": [response], "response": response.content}


def check_availability(state: State):
    if not state.start_time or not state.end_time:
        raise ValueError("Start time and end time are required")

    # inital check for events
    check_event = check_events(state.start_time, state.end_time)

    if not check_event["google_calendar"] and not check_event["pending_requests"]:
        return {"calendar_events": [], "calendar_available": True}

    start_window = state.start_time.replace(hour=10, minute=0, second=0, microsecond=0)
    end_window = state.end_time.replace(hour=18, minute=0, second=0, microsecond=0)
    check_event = check_events(start_window, end_window)
    return {"calendar_events": check_event, "calendar_available": False}


async def save_approval_request(state: State):
    req_id = str(uuid.uuid4())
    llm = init_chat_model(model=chat_model, api_key=GROQ_API_KEY)
    event_exists = check_db_for_event(state.email, state.start_time)

    if event_exists:
        response = await llm.ainvoke(
            f"""
             The user wants to create a meeting, but they already have
             an meeting that is either approved or pending on the same day.

             Existing meeting:
             - Date and time: {event_exists.start_date_time}
             - End time: {event_exists.end_date_time}
             - Description: {event_exists.event_description}

             User's current request:
             - Requested date and time: {state.start_time}
             - Description: {state.event_description}

             Explain that they cannot create another meeting under the current
             booking rules. Suggest editing their existing meeting if needed.

             Keep the response concise and conversational.
             Do not claim that any changes have been made.
             """
        )

        return {
            "messages": [response],
            "response": response.content,
            "event_exists": True,
        }

    data = {
        "thread_id": state.thread_id,
        "req_id": req_id,
        "name": state.name,
        "email": state.email,
        "start_date_time": state.start_time,
        "end_date_time": state.end_time,
        "event_description": state.event_description,
        "status": status.PENDING,
    }

    create_req(**data)

    response = await llm.ainvoke(
        f"""
         The user's meeting request has been saved and is pending verification.
         Meeting: {state.event_description}
         Start time: {state.start_time}

         Tell the user you have sent a request provisionally to kasturi to confirm availability outside the calendar
         and that they will receive an email from assistant@krangan.dev
         once Kasturi confirms it.

         Keep the response concise and conversational.
         Do not imply that the calendar event has already been created.
         """
    )

    return {
        "req_id": req_id,
        "messages": [response],
        "response": response.content,
        "event_exists": False,
    }


def availability_confirmation(state: State):
    data = {
        "req_id": state.req_id,
        "thread_id": state.thread_id,
        "name": state.name,
        "email": state.email,
        "start_date_time": state.start_time,
        "end_date_time": state.end_time,
        "event_description": state.event_description,
    }

    decision = interrupt(data)

    return {"req_id": state.req_id, "booking_approved": decision.get("approved", False)}


def create_event(state: State):
    if not state.start_time or not state.end_time:
        raise ValueError("Start time and end time are required")

    if not state.req_id or state.booking_approved is not True:
        raise ValueError("Request ID missing or request not approved")
    if state.calendar_action == CalendarEnum.CREATE:
        event = {
            "summary": state.event_description,
            "description": f"Meeting requested by {state.name}",
            "start": {
                "dateTime": state.start_time.isoformat(),
                "timeZone": "Asia/Kolkata",
            },
            "end": {
                "dateTime": state.end_time.isoformat(),
                "timeZone": "Asia/Kolkata",
            },
            "attendees": [
                {"email": state.email},
                *[{"email": email} for email in state.attendees],
            ],
        }
        try:
            created_event = (
                service.events()
                .insert(
                    calendarId=calendar_id,
                    body=event,
                    sendUpdates="all",
                )
                .execute()
            )
        except HttpError as exc:
            raise RuntimeError("Google Calendar event creation failed") from exc

        approval = status.APPROVED

        update_exisitng_req(
            req_id=state.req_id, approval=approval, gcal_id=created_event.get("id")
        )

        return {
            "calendar_event_id": created_event["id"],
            "current_calendar_flow": None,
        }

    tz = ZoneInfo("Asia/Kolkata")

    if state.calendar_action == CalendarEnum.UPDATE:
        selected_event = state.selected_event
        if selected_event is None or selected_event.gca_id is None:
            raise ValueError("Selected event or Google Calendar event ID missing")

        updated_fields = {}

        if state.update_values:
            for item in state.update_values:
                field = item["field"]
                value = item["value"]

                if field == "name":
                    updated_fields["summary"] = value

                elif field == "event_description":
                    updated_fields["description"] = value

                elif field in ("start_date_time", "end_date_time"):
                    key = "start" if field == "start_date_time" else "end"
                    dt = datetime.fromisoformat(value).replace(tzinfo=tz)

                    updated_fields[key] = {
                        "dateTime": dt.isoformat(),
                        "timeZone": str(tz),
                    }

        if updated_fields:
            try:
                created_event = (
                    service.events()
                    .patch(
                        calendarId=calendar_id,
                        eventId=selected_event.gca_id,
                        body=updated_fields,
                    )
                    .execute()
                )
            except HttpError as exc:
                raise RuntimeError("Google Calendar event updation failed") from exc

            update_exisitng_req(
                req_id=state.req_id,
                approval=status.APPROVED,
                gcal_id=created_event.get("id"),
            )


def denied_request(state: State):
    if not state.req_id or state.booking_approved is not False:
        raise ValueError("Request ID missing or request not approved")

    approval = status.DECLINED

    update_exisitng_req(req_id=state.req_id, approval=approval, gcal_id=None)

    return {
        "current_calendar_flow": None,
    }
