from sqlalchemy import Column, Integer, String, ForeignKey, JSON, Text, Float, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from database import Base
import uuid

class Node(Base):
    __tablename__ = "nodes"

    # IDENTITY
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    level = Column(Integer, nullable=False)  # 0=Raw, 1=Qtr, 2=Year, 3=Arc
    
    # TEMPORAL
    year = Column(Integer, nullable=True)
    quarter = Column(Integer, nullable=True)
    
    # GRAPH (The Skeleton)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("nodes.id"), nullable=True)
    
    # CONTENT (The Knowledge)
    content = Column(Text, nullable=False)
    
    # SEARCH (The Vector) - 1536 dim for OpenAI
    embedding = Column(Vector(1536))
    
    # FORENSICS (The Evidence: bbox, page number, section name)
    metadata_ = Column("metadata", JSONB, nullable=True)


