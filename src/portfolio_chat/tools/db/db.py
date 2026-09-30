from sqlmodel import SQLModel, create_engine

from ...config import NEONDB_URL

engine = create_engine(NEONDB_URL, pool_pre_ping=True)

def init_db():
    SQLModel.metadata.create_all(engine)
