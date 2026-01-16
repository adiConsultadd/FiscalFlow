from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from database import Base
import uuid

class Node(Base):
    __tablename__ = "nodes" 

    node_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    parent_node_id = Column(UUID(as_uuid=True), ForeignKey("nodes.node_id"), nullable=True)
    
    level_depth = Column(Integer, nullable=False) 

    text_content = Column(Text, nullable=False)
    
    embedding = Column(Vector(1536))

    company_ticker = Column(String(10), nullable=False) 
    fiscal_year = Column(String(10), nullable=False)   
    fiscal_quarter = Column(String(10), nullable=True)  
    
    # Topic Name (Level 1)
    topic = Column(String, nullable=True)

    node_metadata = Column(JSONB, default={}, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # OPTIMIZATION INDEXES
    __table_args__ = (
        # 1. HNSW Index for Vector Search (Critical for semantic speed)
        Index(
            "idx_nodes_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
        # 2. Composite B-Tree Index for Time-Slice Filtering
        Index(
            "idx_nodes_filters",
            "company_ticker", 
            "fiscal_year", 
            "fiscal_quarter", 
            "level_depth"
        ),
        # 3. GIN Index for Metadata (If you search inside the JSON)
        Index(
            "idx_nodes_metadata", 
            "node_metadata", 
            postgresql_using="gin"
        ),
    )