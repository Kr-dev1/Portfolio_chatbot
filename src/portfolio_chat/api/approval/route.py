from fastapi import APIRouter, HTTPException, Request
from langgraph.types import Command
from pydantic import BaseModel


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
