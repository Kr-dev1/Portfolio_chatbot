from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from portfolio_chat.api.chat.nodes import (
    chat_node,
    guard_rail_and_classiier_node,
    guardrail_fail_node,
)
from portfolio_chat.api.chat.state import State
from portfolio_chat.api.chat.tools import tools

memory = MemorySaver()

graph_handler = StateGraph(State)


def guardrail_router(state: State):
    if not state.is_on_topic:
        return "guardrail_fail"
    else:
        return "chat"

def calendar_router(state: State):
    if state.calendar_action:
        return "tools"
    else:
        return "chat"

tool_node = ToolNode(tools)

# Nodes
graph_handler.add_node("guardrail", guard_rail_and_classiier_node)
graph_handler.add_node("chat", chat_node)
graph_handler.add_node("guardrail_fail", guardrail_fail_node)
graph_handler.add_node("tools", tool_node)


# Edges
graph_handler.add_edge(START, "guardrail")
graph_handler.add_edge("chat", END)
graph_handler.add_edge("guardrail_fail", END)

# Conditionals
graph_handler.add_conditional_edges(
    "guardrail", guardrail_router, {"guardrail_fail": "guardrail_fail", "chat": "chat"}
)

graph = graph_handler.compile(checkpointer=memory)
