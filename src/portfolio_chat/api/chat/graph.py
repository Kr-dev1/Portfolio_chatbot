from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt.tool_node import ToolNode

from portfolio_chat.api.chat.nodes import (
    availability_confirmation,
    chat_node,
    check_availability,
    extract_user_info,
    get_availability_info,
    guard_rail_and_classiier_node,
    guardrail_fail_node,
)
from portfolio_chat.api.chat.state import State

memory = MemorySaver()

graph_handler = StateGraph(State)


def guardrail_router(state: State):
    if not state.is_on_topic:
        return "guardrail_fail"
    if state.calendar_action:
        return "calendar_info_extractor"
    else:
        return "chat"


def calendar_router(state: State):
    if not state.start_time:
        return "get_more_info"
    else:
        return "check_availability"


def clendar_action_router(state: State):
    if not state.calendar_available or not state.email or not state.name:
        return "get_more_info"
    else:
        return "availability_confirmation"


# Nodes
graph_handler.add_node("guardrail", guard_rail_and_classiier_node)
graph_handler.add_node("chat", chat_node)
graph_handler.add_node("guardrail_fail", guardrail_fail_node)
graph_handler.add_node("calendar_info_extractor", extract_user_info)
graph_handler.add_node("check_availability", check_availability)
graph_handler.add_node("get_more_info", get_availability_info)
graph_handler.add_node("availability_confirmation", availability_confirmation)


# Edges
graph_handler.add_edge(START, "guardrail")
graph_handler.add_edge("chat", END)
graph_handler.add_edge("guardrail_fail", END)

# Conditionals
graph_handler.add_conditional_edges(
    "guardrail",
    guardrail_router,
    {
        "guardrail_fail": "guardrail_fail",
        "calendar_info_extractor": "calendar_info_extractor",
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
        "availability_confirmation": "availability_confirmation",
    },
)

graph = graph_handler.compile(checkpointer=memory)
