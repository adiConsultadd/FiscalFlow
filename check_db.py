
import asyncio
import sqlalchemy
from database import engine

async def check_db():
    async with engine.connect() as conn:
        print("Checking Database Content...")
        
        # Check levels
        result = await conn.execute(sqlalchemy.text("SELECT level_depth, count(*) FROM nodes GROUP BY level_depth ORDER BY level_depth"))
        print("\nNode Counts by Level (Expect 0-4):")
        rows = result.fetchall()
        for row in rows:
            print(f"Level {row[0]}: {row[1]}")
            
        # Check linking
        result = await conn.execute(sqlalchemy.text("SELECT count(*) FROM nodes WHERE parent_node_id IS NOT NULL"))
        print(f"\nNodes with parents: {result.scalar()}")
        
        # Sample Quarter (L2)
        result = await conn.execute(sqlalchemy.text("SELECT text_content, topic, node_metadata FROM nodes WHERE level_depth = 2 LIMIT 1"))
        row = result.fetchone()
        if row:
            print(f"\nSample Quarter Node (L2):\nTopic: {row[1]}\nContent: {row[0][:200]}...")
            
        # Sample Year (L3)
        result = await conn.execute(sqlalchemy.text("SELECT text_content, topic, node_metadata FROM nodes WHERE level_depth = 3 LIMIT 1"))
        row = result.fetchone()
        if row:
            print(f"\nSample Year Node (L3):\nTopic: {row[1]}\nContent: {row[0][:200]}...")
            
        # Sample Root (L4)
        result = await conn.execute(sqlalchemy.text("SELECT text_content, topic, node_metadata FROM nodes WHERE level_depth = 4 LIMIT 1"))
        row = result.fetchone()
        if row:
            print(f"\nSample Root Node (L4):\nTopic: {row[1]}\nContent: {row[0][:200]}...")

if __name__ == "__main__":
    asyncio.run(check_db())
