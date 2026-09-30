from fastapi import APIRouter, status

from portfolio_chat.api.chat.graph import graph
from portfolio_chat.api.chat.nodes import State

router = APIRouter(
    prefix="/chat",
    tags=["Health"],
)


@router.get("/chat")
def chat(user_message: str):
    result = graph.invoke(State(user_input=user_message))

    return {
        "response": result["response"],
    }
