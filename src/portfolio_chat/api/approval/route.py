from fastapi import APIRouter, HTTPException, Request
from langgraph.types import Command
from pydantic import BaseModel

from portfolio_chat.tools.requests.request import get_event_status


class ApprovalRequest(BaseModel):
    req_id: str
    thread_id: str
    approved: bool


router = APIRouter(
    prefix="/approval",
    tags=["approval"],
)


@router.post("")
async def handle_approval(
    body: ApprovalRequest,
    request: Request,
):
    graph = request.app.state.graph

    config = {
        "configurable": {
            "thread_id": body.thread_id,
        }
    }

    snapshot = await graph.aget_state(config)

    target_interrupt_id = None

    event_exisits = get_event_status(body.req_id, body.thread_id)

    if not event_exisits.get("found_event"):
        raise HTTPException(
            status_code=404,
            detail=f"No pending approval found for req_id: {body.req_id} on this thread.",
        )

    if snapshot.tasks:
        for task in snapshot.tasks:
            for interrupt in task.interrupts:
                payload = interrupt.value
                if isinstance(payload, dict) and payload.get("req_id") == body.req_id:
                    target_interrupt_id = interrupt.id
                    break
            if target_interrupt_id:
                break

    if not target_interrupt_id:
        raise HTTPException(
            status_code=409,
            detail=f"No pending approval found for req_id: {body.req_id} on this thread.",
        )

    if not snapshot.tasks or not any(task.interrupts for task in snapshot.tasks):
        raise HTTPException(
            status_code=409,
            detail="No pending approval found for this thread",
        )

    result = await graph.ainvoke(
        Command(resume={"approved": body.approved}),
        config=config,
    )

    return {
        "req_id": body.req_id,
        "status": "approved" if body.approved else "denied",
        "result": result.get("response"),
    }
