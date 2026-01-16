from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class QueryRequest(BaseModel):
    """Request model for the query endpoint."""
    username: str
    query: str
    company_ticker: Optional[str] = None


class ProvenanceInfo(BaseModel):
    """Provenance metadata for a source chunk."""
    page_index: Optional[int] = None
    page_label: Optional[str] = None
    file_name: Optional[str] = None
    bbox: Optional[List[float]] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class SourceChunk(BaseModel):
    """A source chunk with provenance for frontend display."""
    evidence_id: int
    text_content: str
    similarity: float
    fiscal_year: Optional[str] = None
    fiscal_quarter: Optional[str] = None
    company_ticker: Optional[str] = None
    provenance: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    """Full response model for non-streaming queries."""
    query: str
    answer: str
    target_years: List[str]
    chunks_retrieved: int
    sources: List[SourceChunk]
    traversal_log: List[str]