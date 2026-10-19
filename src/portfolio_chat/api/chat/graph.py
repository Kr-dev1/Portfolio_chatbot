from langgraph.graph import END, START, StateGraph

from portfolio_chat.api.chat.nodes import (
    availability_confirmation,
    chat_node,
    check_availability,
    check_event_info,
    create_event,
    denied_request,
    edit_event_extractor,
    extract_user_info,
    get_availability_info,
    guard_rail_and_classiier_node,
    guardrail_fail_node,
    save_approval_request,
    send_email,
    update_get_info,
)
from portfolio_chat.api.chat.state import State


def guardrail_router(state: State):
    if not state.is_on_topic:
        return "guardrail_fail"
    if state.calendar_action == "create":
        return "calendar_info_extractor"
    if state.calendar_action == "update":
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
    if state.selected_event_id and not state.update_values:
        return "update_get_info"
    else:
        return "check_event_info"


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

    # Edges
    graph_handler.add_edge(START, "guardrail")
    graph_handler.add_edge("create_event", "send_email")
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
        "save_approval_request",
        route_after_event_check,
        {END: END, "availability_confirmation": "availability_confirmation"},
    )

    graph_handler.add_conditional_edges(
        "edit_event_extractor",
        update_router,
        {"check_event_info": "check_event_info", "update_get_info": "update_get_info"},
    )

    graph = graph_handler.compile(checkpointer=checkpointer)

    return graph


def studio_graph():
    return build_graph(None)
