import asyncio
from database import engine, Base
from sqlalchemy import text

async def init_models():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        
        await conn.run_sync(Base.metadata.drop_all) 
        await conn.run_sync(Base.metadata.create_all)
        print("✅ Database Tables Created!")

if __name__ == "__main__":
    asyncio.run(init_models())