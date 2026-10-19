import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from groq import RateLimitError
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

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    user_message: str
    thread_id: str | None = None


def sse_event(event: str, data: dict[str, Any]) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("")
async def chat(chat_req: ChatRequest, request: Request):
    thread_id = chat_req.thread_id or str(uuid.uuid4())
    graph = request.app.state.graph

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
        },
    }

    async def generate():
        full_response: list[str] = []
        node = ""
        graph_completed = False

        try:
            async for message, metadata in graph.astream(
                State(
                    thread_id=thread_id,
                    user_input=chat_req.user_message,
                    messages=[HumanMessage(content=chat_req.user_message)],
                ),
                config,
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node", "")

                content = getattr(message, "content", None)

                if isinstance(content, str) and content:
                    full_response.append(content)

                    yield f"data: {json.dumps({'content': content})}\n\n"

            graph_completed = True
            yield sse_event("done", {"thread_id": thread_id})

        except RateLimitError:
            logger.warning("Groq rate limit reached for thread %s", thread_id)

            yield sse_event(
                "error",
                {
                    "code": "RATE_LIMITED",
                    "message": ("The AI service is busy. Please try again shortly."),
                },
            )

        except Exception:
            logger.exception(
                "Chat stream failed for thread %s",
                thread_id,
            )

            yield sse_event(
                "error",
                {
                    "code": "INTERNAL_ERROR",
                    "message": ("Something went wrong while generating your response."),
                },
            )

        finally:
            try:
                await message_logger(
                    thread_id=thread_id,
                    message=chat_req.user_message,
                    role=Role.USER,
                    node=node,
                )

                if full_response:
                    await message_logger(
                        thread_id=thread_id,
                        message="".join(full_response),
                        role=Role.ASSISTANT,
                        node=node,
                    )

            except Exception:
                logger.exception(
                    "Failed to log chat messages for thread %s",
                    thread_id,
                )

            if graph_completed:
                try:
                    state_snapshot = await graph.aget_state(config)

                    state_update = await compact_conversation(
                        State(**state_snapshot.values)
                    )

                    if state_update:
                        await graph.aupdate_state(
                            config,
                            state_update,
                        )

                except Exception:
                    logger.exception(
                        "Conversation compaction failed for thread %s",
                        thread_id,
                    )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Thread-ID": thread_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
