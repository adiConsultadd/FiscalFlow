from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Create the Async Engine
engine = create_async_engine(DATABASE_URL, echo=False)

# Create the Session Factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

# Dependency to get DB session in endpoints
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session