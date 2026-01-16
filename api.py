"""
FiscalFlow API - FastAPI Backend with Server-Sent Events (SSE) Streaming

Provides endpoints for:
1. Document upload and processing
2. Query execution with SSE streaming for real-time progress
3. Returns provenance metadata for frontend citation display
"""

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional, List, Dict, Any
import os
import shutil
import json
import asyncio
from datetime import datetime

from schema import QueryRequest, ProvenanceInfo, SourceChunk, QueryResponse
from document_processor import process_document
from graph import run_query, AgentState

app = FastAPI(
    title="FiscalFlow API",
    description="Forensic Financial Analysis with Time-Aware RAG",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCUMENTS_DIR = "documents"

# =============================================================================
# SSE STREAMING ENDPOINT
# =============================================================================

async def generate_sse_events(username: str, query: str, company_ticker: Optional[str] = None):
    """
    Generator that yields Server-Sent Events for streaming query progress.
    
    Event Types:
    - status: Progress updates (e.g., "Identifying relevant years...")
    - sources: Retrieved chunks with provenance
    - answer: The final synthesized answer (streamed token by token if possible)
    - complete: Final event with full results
    - error: Error information if something fails
    """
    
    def format_sse(event_type: str, data: Any) -> str:
        """Format data as SSE event."""
        json_data = json.dumps(data, ensure_ascii=False)
        return f"event: {event_type}\ndata: {json_data}\n\n"
    
    try:
        # Send initial status
        yield format_sse("status", {
            "stage": "started",
            "message": f"Processing query for {username}...",
            "timestamp": datetime.now().isoformat()
        })
        
        await asyncio.sleep(0.1) 
        
        # Stage 1: Time scoping
        yield format_sse("status", {
            "stage": "time_scoping",
            "message": "Identifying relevant fiscal years...",
            "timestamp": datetime.now().isoformat()
        })
        
        # Run the full query through the graph
        result = await run_query(query, company_ticker=company_ticker)
        
        # Stage 2: Send target years found
        yield format_sse("status", {
            "stage": "years_identified",
            "message": f"Found relevant years: {', '.join(result['target_years'])}",
            "target_years": result["target_years"],
            "timestamp": datetime.now().isoformat()
        })
        
        await asyncio.sleep(0.1)
        
        # Stage 3: Send sources with provenance
        yield format_sse("status", {
            "stage": "retrieving_sources",
            "message": f"Retrieved {result['chunks_retrieved']} relevant sources...",
            "timestamp": datetime.now().isoformat()
        })
        
        # Send each source as a separate event for progressive display
        for source in result["sources"]:
            yield format_sse("source", source)
            await asyncio.sleep(0.05)  # Slight stagger for visual effect
        
        # Stage 4: Send the answer
        yield format_sse("status", {
            "stage": "synthesizing",
            "message": "Generating forensic analysis...",
            "timestamp": datetime.now().isoformat()
        })
        
        await asyncio.sleep(0.1)
        
        # Send the full answer
        yield format_sse("answer", {
            "content": result["answer"],
            "timestamp": datetime.now().isoformat()
        })
        
        # Stage 5: Complete with full results
        yield format_sse("complete", {
            "query": query,
            "username": username,
            "target_years": result["target_years"],
            "chunks_retrieved": result["chunks_retrieved"],
            "sources_count": len(result["sources"]),
            "traversal_log": result["traversal_log"],
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        yield format_sse("error", {
            "message": str(e),
            "stage": "processing",
            "timestamp": datetime.now().isoformat()
        })


@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    """
    SSE streaming endpoint for query processing.
    
    Streams events as the query is processed:
    - status: Progress updates
    - source: Individual source chunks with provenance
    - answer: The final synthesized answer
    - complete: Processing complete with metadata
    - error: Error details if processing fails
    
    Frontend should use EventSource or fetch with ReadableStream to consume.
    """
    return StreamingResponse(
        generate_sse_events(
            username=request.username,
            query=request.query,
            company_ticker=request.company_ticker
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no" 
        }
    )


@app.post("/query", response_model=QueryResponse)
async def query_sync(request: QueryRequest):
    """
    Synchronous query endpoint (non-streaming).
    Returns the full result after processing completes.
    """
    try:
        result = await run_query(
            query=request.query,
            company_ticker=request.company_ticker
        )
        
        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            target_years=result["target_years"],
            chunks_retrieved=result["chunks_retrieved"],
            sources=[SourceChunk(**s) for s in result["sources"]],
            traversal_log=result["traversal_log"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DOCUMENT UPLOAD ENDPOINT
# =============================================================================

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    company: str = Form(...),
    year: str = Form(...),
    quarter: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload a financial document for processing.
    Processing happens in the background.
    """
    # Create directory structure
    save_dir = os.path.join(DOCUMENTS_DIR, company, year, quarter)
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, file.filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Trigger processing in background
    background_tasks.add_task(
        process_document, 
        file_path, 
        company_ticker=company, 
        fiscal_year=year, 
        fiscal_quarter=quarter
    )
    
    return {
        "message": "File uploaded and processing started.",
        "file_path": file_path,
        "metadata": {
            "company": company,
            "year": year,
            "quarter": quarter
        }
    }


@app.get("/")
def read_root():
    """API health check and info."""
    return {
        "message": "Welcome to FiscalFlow API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query/stream": "SSE streaming query with provenance",
            "POST /query": "Synchronous query",
            "POST /upload": "Upload financial documents"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

