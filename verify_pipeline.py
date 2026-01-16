
import asyncio
import os
import sqlalchemy
from reportlab.pdfgen import canvas
from database import engine, Base
from models import Node # Import to ensure models are registered
from document_processor import process_document

async def create_tables():
    async with engine.begin() as conn:
        await conn.execute(sqlalchemy.text("DROP TABLE IF EXISTS nodes CASCADE"))
        await conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created (clean slate).")

async def main():
    await create_tables()
    
    try:
        await process_document("documents/one97/concalls/PAYTM_04112025215154_Reg30_Earnings_Release_FY26_Q2_OCL_sd.pdf")
        
        # Verify Hierarchy
        async with engine.connect() as conn:
            # Check levels
            result = await conn.execute(sqlalchemy.text("SELECT level_depth, count(*) FROM nodes GROUP BY level_depth ORDER BY level_depth"))
            print("\nNode Counts by Level:")
            for row in result:
                print(f"Level {row[0]}: {row[1]}")
                
            # Check linking
            result = await conn.execute(sqlalchemy.text("SELECT count(*) FROM nodes WHERE parent_node_id IS NOT NULL"))
            print(f"Nodes with parents: {result.scalar()}")
            
            # Show a sample topic
            result = await conn.execute(sqlalchemy.text("SELECT text_content, topic, node_metadata, embedding FROM nodes WHERE level_depth = 1 LIMIT 1"))
            row = result.fetchone()
            if row:
                print(f"\nSample Topic Node:")
                print(f"Content (Summary): {row[0]}")
                print(f"Topic: {row[1]}")
                print(f"Metadata Keys: {list(row[2].keys())}")
                print(f"Has Embedding: {row[3] is not None}")

    except Exception as e:
        print(f"Processing failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
