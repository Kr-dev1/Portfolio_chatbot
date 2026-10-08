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
    request_start = time.perf_counter()

    thread_id = request.thread_id or str(uuid.uuid4())

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "recursion_limit": 4}
    }

    async def generate():
        graph_start = time.perf_counter()
        first_token_time = None
        full_response = []

        async for message, metadata in graph.astream(
            State(
                user_input=request.user_message,
                messages=[HumanMessage(content=request.user_message)],
            ),
            config,
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node")

            if node not in {
                "chat",
                "guardrail_fail",
                "get_more_info",
            }:
                continue

            if message and hasattr(message, "content") and message.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                    print(
                        f"[TIMING] First token: "
                        f"{first_token_time - graph_start:.3f}s"
                    )

                full_response.append(message.content)
                yield message.content

        stream_end = time.perf_counter()

        print(
            f"[TIMING] Graph stream complete: "
            f"{stream_end - graph_start:.3f}s"
        )

        print(
            f"[TIMING] Request → stream complete: "
            f"{stream_end - request_start:.3f}s"
        )

        await message_logger(
            thread_id=thread_id,
            message=request.user_message,
            role=Role.USER,
        )

        await message_logger(
            thread_id=thread_id,
            message="".join(full_response),
            role=Role.ASSISTANT,
        )

        state_snapshot = await graph.aget_state(config)

        state_update = await compact_conversation(
            State(**state_snapshot.values)
        )

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
