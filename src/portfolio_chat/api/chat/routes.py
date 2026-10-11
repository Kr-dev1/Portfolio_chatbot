import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

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
async def chat(chat_req: ChatRequest, request: Request):
    thread_id = chat_req.thread_id or str(uuid.uuid4())
    graph = request.app.state.graph

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "recursion_limit": 4}
    }

    async def generate():
        first_token_time = None
        full_response = []
        node = ""

        async for message, metadata in graph.astream(
            State(
                thread_id=thread_id,
                user_input=chat_req.user_message,
                messages=[HumanMessage(content=chat_req.user_message)],
            ),
            config,
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node")

            if node not in {
                "chat",
                "guardrail_fail",
                "get_more_info",
                "availability_confirmation",
            }:
                continue

            if message and hasattr(message, "content") and message.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_response.append(message.content)
                yield message.content

        await message_logger(
            thread_id=thread_id,
            message=chat_req.user_message,
            role=Role.USER,
            node=node,
        )

        await message_logger(
            thread_id=thread_id,
            message="".join(full_response),
            role=Role.ASSISTANT,
            node=node,
        )

        state_snapshot = await graph.aget_state(config)

        state_update = await compact_conversation(State(**state_snapshot.values))

        if state_update:
            await graph.aupdate_state(
                config,
                state_update,
            )

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"X_Thread_ID": thread_id},
    )
