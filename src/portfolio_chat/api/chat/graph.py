from langgraph.graph import END, START, StateGraph

from portfolio_chat.api.chat.nodes.booking import (
    availability_confirmation,
    check_availability,
    create_event,
    denied_request,
    extract_user_info,
    get_availability_info,
    save_approval_request,
)
from portfolio_chat.api.chat.nodes.chat import chat_node
from portfolio_chat.api.chat.nodes.delete import delete_node
from portfolio_chat.api.chat.nodes.email import send_contact_email, send_email
from portfolio_chat.api.chat.nodes.guardrail import (
    guard_rail_and_classiier_node,
    guardrail_fail_node,
)
from portfolio_chat.api.chat.nodes.update import (
    cancel_edit_node,
    check_event_info,
    check_update_event_status,
    confirm_change_event,
    create_update_request,
    create_update_response,
    edit_event_extractor,
    update_get_info,
)
from portfolio_chat.api.chat.schema.schema import CalendarEnum
from portfolio_chat.api.chat.state import State


def guardrail_router(state: State):
    if not state.is_on_topic:
        return "guardrail_fail"
    if state.calendar_action == "create":
        return "calendar_info_extractor"
    if state.calendar_action == "update" or state.calendar_action == "delete":
        return "edit_event_extractor"
    else:
        return "chat"


def calendar_router(state: State):
    if not state.start_time:
        return "get_more_info"
    else:
        return "check_availability"


def clendar_action_router(state: State):
    if state.calendar_available is False:
        return "get_more_info"

    if not state.email or not state.name or not state.event_description:
        return "get_more_info"

    return "save_approval_request"


def route_after_event_check(state: State):
    if state.event_exists:
        return END
    return "availability_confirmation"


def approval_router(state: State):
    if state.booking_approved:
        return "create_event"
    return "denied_request"


def update_router(state: State):
    if not state.update_email:
        return "update_get_info"

    if state.calendar_action == CalendarEnum.UPDATE:
        if state.cancel_edit:
            return "cancel_edit_node"
        if state.done_editing:
            return "check_update_event_status"
        if state.selected_event_id and not state.update_values:
            return "update_get_info"
        if state.selected_event and state.update_values:
            return "confirm_change_event"

    if state.calendar_action == CalendarEnum.DELETE and state.selected_event:
        return "delete_node"

    return "check_event_info"


def update_confirmation_route(state: State):
    if state.existing_request_declined:
        return "create_update_response"
    return "send_contact_email"


def build_graph(checkpointer):
    graph_handler = StateGraph(State)

    # Nodes
    graph_handler.add_node("guardrail", guard_rail_and_classiier_node)
    graph_handler.add_node("chat", chat_node)
    graph_handler.add_node("guardrail_fail", guardrail_fail_node)
    graph_handler.add_node("calendar_info_extractor", extract_user_info)
    graph_handler.add_node("check_availability", check_availability)
    graph_handler.add_node("get_more_info", get_availability_info)
    graph_handler.add_node("save_approval_request", save_approval_request)
    graph_handler.add_node("availability_confirmation", availability_confirmation)
    graph_handler.add_node("create_event", create_event)
    graph_handler.add_node("denied_request", denied_request)
    graph_handler.add_node("send_email", send_email)
    graph_handler.add_node("edit_event_extractor", edit_event_extractor)
    graph_handler.add_node("update_get_info", update_get_info)
    graph_handler.add_node("check_event_info", check_event_info)
    graph_handler.add_node("confirm_change_event", confirm_change_event)
    graph_handler.add_node("cancel_edit_node", cancel_edit_node)
    graph_handler.add_node("check_update_event_status", check_update_event_status)
    graph_handler.add_node("send_contact_email", send_contact_email)
    graph_handler.add_node("create_update_response", create_update_response)
    graph_handler.add_node("create_update_request", create_update_request)
    graph_handler.add_node("delete_node", delete_node)

    # Edges
    graph_handler.add_edge(START, "guardrail")
    graph_handler.add_edge("create_event", "send_email")
    graph_handler.add_edge("create_update_response", "create_update_request")
    graph_handler.add_edge("chat", END)
    graph_handler.add_edge("guardrail_fail", END)

    # Conditionals
    graph_handler.add_conditional_edges(
        "guardrail",
        guardrail_router,
        {
            "guardrail_fail": "guardrail_fail",
            "calendar_info_extractor": "calendar_info_extractor",
            "edit_event_extractor": "edit_event_extractor",
            "chat": "chat",
        },
    )

    graph_handler.add_conditional_edges(
        "calendar_info_extractor",
        calendar_router,
        {"get_more_info": "get_more_info", "check_availability": "check_availability"},
    )

    graph_handler.add_conditional_edges(
        "check_availability",
        clendar_action_router,
        {
            "get_more_info": "get_more_info",
            "save_approval_request": "save_approval_request",
        },
    )

    graph_handler.add_conditional_edges(
        "availability_confirmation",
        approval_router,
        {
            "create_event": "create_event",
            "denied_request": "denied_request",
        },
    )

    graph_handler.add_conditional_edges(
        "create_update_request",
        approval_router,
        {
            "create_event": "create_event",
            "denied_request": "denied_request",
        },
    )

    graph_handler.add_conditional_edges(
        "save_approval_request",
        route_after_event_check,
        {END: END, "availability_confirmation": "availability_confirmation"},
    )

    graph_handler.add_conditional_edges(
        "edit_event_extractor",
        update_router,
        {
            "check_event_info": "check_event_info",
            "update_get_info": "update_get_info",
            "confirm_change_event": "confirm_change_event",
            "cancel_edit_node": "cancel_edit_node",
            "check_update_event_status": "check_update_event_status",
            "delete_node": "delete_node",
        },
    )

    graph_handler.add_conditional_edges(
        "check_update_event_status",
        update_confirmation_route,
        {
            "create_update_response": "create_update_response",
            "send_contact_email": "send_contact_email",
        },
    )

    graph = graph_handler.compile(checkpointer=checkpointer)

    return graph


def studio_graph():
    return build_graph(None)
