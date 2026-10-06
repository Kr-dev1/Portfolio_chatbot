import time
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from portfolio_chat.api.chat.graph import graph
from portfolio_chat.api.chat.state import State
from portfolio_chat.tools.helpers.helper import compact_conversation
from portfolio_chat.tools.messages.message_logger import message_logger
from portfolio_chat.tools.messages.model import Role

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


class ChatRequest(BaseModel):
    user_message: str
    thread_id: str | None = None


@router.post("")
async def chat(request: ChatRequest):
    thread_id = request.thread_id or str(uuid.uuid4())

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "recursion_limit": 4}
    }

    async def generate():
        full_response = []
        async for message, metadata in graph.astream(
            State(
                user_input=request.user_message,
                messages=[HumanMessage(content=request.user_message)],
            ),
            config,
            stream_mode="messages",
        ):
            if message and hasattr(message, "content") and message.content:
                full_response.append(message.content)
                yield message.content.encode("utf-8")

        await message_logger(
            thread_id=thread_id, message=request.user_message, role=Role.USER
        )

        await message_logger(
            thread_id=thread_id,
            message="".join(full_response),
            role=Role.ASSISTANT,
        )

        state_snapshot = await graph.aget_state(config)

        state_update = await compact_conversation(State(**state_snapshot.values))

        if state_update:
            await graph.aupdate_state(
                config,
                state_update,
            )

    return StreamingResponse(
        generate(), media_type="text/plain", headers={"X_Thread_ID": thread_id}
    )
