
import asyncio
import sqlalchemy
from database import engine

async def check_db():
    async with engine.connect() as conn:
        print("Checking Database Content...")
        
        # Check levels
        result = await conn.execute(sqlalchemy.text("SELECT level_depth, count(*) FROM nodes GROUP BY level_depth ORDER BY level_depth"))
        print("\nNode Counts by Level:")
        rows = result.fetchall()
        for row in rows:
            print(f"Level {row[0]}: {row[1]}")
            
        # Check linking
        result = await conn.execute(sqlalchemy.text("SELECT count(*) FROM nodes WHERE parent_node_id IS NOT NULL"))
        print(f"\nNodes with parents: {result.scalar()}")
        
        # Sample Topic
        result = await conn.execute(sqlalchemy.text("SELECT text_content, node_metadata FROM nodes WHERE level_depth = 1 LIMIT 1"))
        row = result.fetchone()
        if row:
            print(f"\nSample Topic Node:\nContent: {row[0]}")
            print(f"Metadata: {row[1]}")

if __name__ == "__main__":
    asyncio.run(check_db())
