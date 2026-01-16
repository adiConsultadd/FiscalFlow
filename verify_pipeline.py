
import asyncio
import os
import sqlalchemy
from reportlab.pdfgen import canvas
from database import engine, Base
from models import Node # Import to ensure models are registered
from document_processor import process_document

async def create_tables():
    async with engine.begin() as conn:
        await conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created.")

async def main():
    await create_tables()
    
    try:
        await process_document("documents/one97/concalls/PAYTM_04112025215154_Reg30_Earnings_Release_FY26_Q2_OCL_sd.pdf")
    except Exception as e:
        print(f"Processing failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
