from langgraph.graph import END, START, StateGraph

from portfolio_chat.api.chat.nodes import (
    chat_node,
    guard_rail_node,
    guardrail_fail_node,
)
from portfolio_chat.api.chat.state import State

graph_handler = StateGraph(State)


def guardrail_router(state: State):
    if not state.is_on_topic:
        return "guardrail_fail"
    else:
        return "chat"


# Nodes
graph_handler.add_node("guardrail", guard_rail_node)
graph_handler.add_node("chat", chat_node)
graph_handler.add_node("guardrail_fail", guardrail_fail_node)

# Edges
graph_handler.add_edge(START, "guardrail")
graph_handler.add_edge("chat", END)
graph_handler.add_edge("guardrail_fail", END)

# Conditionals
graph_handler.add_conditional_edges("guardrail", guardrail_router, {
    "guardrail_fail": "guardrail_fail",
    "chat": "chat"
})

graph =  graph_handler.compile()
