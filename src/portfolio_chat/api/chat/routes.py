from fastapi import APIRouter

from portfolio_chat.api.chat.graph import graph
from portfolio_chat.api.chat.state import State

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.get("")
def chat(user_message: str):
    result = graph.invoke(State(user_input=user_message))
    return {
        "response": result["response"],
    }
