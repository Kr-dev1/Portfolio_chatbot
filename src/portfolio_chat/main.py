from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from portfolio_chat.api.chat.graph import build_graph
from portfolio_chat.config import NEONDB_URL

from .api.approval.route import router as approval_router
from .api.chat.routes import router as chat_router
from .api.health.routes import router as health_router

load_dotenv()

serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("portfolio_chat.tools.requests.model", "status"),
        ("portfolio_chat.tools.requests.model", "ApprovalRequests"),
    ]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPostgresSaver.from_conn_string(
        NEONDB_URL, serde=serde
    ) as checkpointer:
        await checkpointer.setup()

        app.state.checkpointer = checkpointer
        app.state.graph = build_graph(checkpointer=checkpointer)

        yield


app = FastAPI(lifespan=lifespan)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(approval_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
