from dotenv import load_dotenv
from fastapi import FastAPI

from .api.chat.routes import router as chat_router
from .api.health.routes import router as health_router

load_dotenv()

app = FastAPI()

app.include_router(health_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
