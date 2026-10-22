from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from googleapiclient.http import HttpError
from langchain_core.messages import AIMessage

from portfolio_chat.api.chat.state import State
from portfolio_chat.config import PRIMARY_SPACE_MAIL
from portfolio_chat.tools.delete.delete import delete_event


def delete_node(state: State):

    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    if state.selected_event:
        start_time = state.selected_event.start_date_time
        is_positive_24_hrs = False
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=ZoneInfo("Asia/Kolkata"))

        is_positive_24_hrs = (
            timedelta(hours=0) <= (start_time - now) >= timedelta(hours=24)
        )
        if not is_positive_24_hrs:
            response = f"""I can only cancel meetings with at least 24 hours' notice.
            For anything sooner, please email Kasturi at {PRIMARY_SPACE_MAIL}."""
            return {
                "messages": [AIMessage(content=response)],
                "response": response,
                "calendar_available": None,
                "selected_event": None,
                "selected_event_id": None,
                "update_email": None,
                "update_values": None,
                "current_calendar_flow": None,
            }
        else:
            if state.selected_event:
                try:
                    delete_event(
                        state.selected_event.req_id, state.selected_event.gca_id
                    )

                    response = "The event has been deleted successfully."

                    return {
                        "messages": [AIMessage(content=response)],
                        "response": response,
                        "calendar_available": None,
                    }
                except (HttpError, ValueError) as e:
                    response = (
                        "There was an error deleting the event"
                        "You can contact Kasturi to get it removed at me@krangan.dev"
                        "Or try again in some time"
                    )
                    print(e)
                    return {
                        "messages": [AIMessage(content=response)],
                        "response": response,
                        "calendar_available": None,
                        "selected_event": None,
                        "selected_event_id": None,
                        "update_email": None,
                        "update_values": None,
                        "current_calendar_flow": None,
                    }
