from dotenv import load_dotenv
from fastapi import FastAPI

from .health.routes import router as health_router

load_dotenv()

app = FastAPI()

app.include_router(health_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
