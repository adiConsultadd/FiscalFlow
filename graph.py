"""
AlphaPulse: Time-Aware LangGraph RAG System for Forensic Financial Analysis

This module implements a StateGraph with 3 main nodes:
1. time_scoper - Identifies relevant fiscal years from user query
2. tree_traverser - BFS pruning tree traversal for intelligent retrieval
3. synthesizer - Generates forensic financial answers with citations

Architecture: Year (L3) → Quarter (L2) → Topic (L1) → Chunks (L0)
"""

import os
import uuid
import asyncio
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Optional imports - only needed when using real database
try:
    from sqlalchemy import select, text
    from sqlalchemy.ext.asyncio import AsyncSession
    from database import AsyncSessionLocal
    from models import Node
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    AsyncSession = None  # type: ignore

load_dotenv()

# =============================================================================
# AGENT STATE DEFINITION
# =============================================================================

class ChunkWithMetadata(TypedDict):
    """
    A retrieved chunk with its metadata for provenance tracking.
    """
    text_content: str
    similarity: float
    node_id: str
    fiscal_year: str
    fiscal_quarter: Optional[str]
    company_ticker: str
    provenance: Dict[str, Any]  # Contains page_index, page_label, file_name, bbox, etc.


class AgentState(TypedDict):
    """
    Shared state passed between all nodes in the LangGraph.
    
    Attributes:
        query: The original user query
        target_years: List of fiscal years to scope (e.g., ['FY2021', 'FY2022'])
        target_quarters: List of quarters after pruning (e.g., ['Q1', 'Q2'])
        target_topics: List of topic node IDs after pruning
        retrieved_chunks: List of chunk dicts with text and metadata for provenance
        final_answer: The synthesized forensic answer
        traversal_log: Debug info tracking pruning decisions at each level
        is_direct_query: Flag for simple fact lookups (bypass tree walk)
        company_ticker: Target company for filtering (optional)
    """
    query: str
    target_years: List[str]
    target_quarters: List[str]
    target_topics: List[Dict[str, Any]]
    retrieved_chunks: List[ChunkWithMetadata]  # Now includes metadata!
    final_answer: str
    traversal_log: List[str]
    is_direct_query: bool
    company_ticker: Optional[str]


# =============================================================================
# LLM INITIALIZATION
# =============================================================================

# GPT-4o-mini for fast pruning decisions (cost-effective)
pruning_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

# GPT-4o for high-quality synthesis
synthesis_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)


# =============================================================================
# REAL DATABASE FUNCTIONS
# These use SQLAlchemy async sessions with pgvector for semantic search
# =============================================================================

# OpenAI client for generating query embeddings
from openai import AsyncOpenAI
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def generate_query_embedding(query: str) -> List[float]:
    """
    Generates an embedding vector for the user query using OpenAI.
    
    Args:
        query: The user's natural language query
        
    Returns:
        List of floats representing the 1536-dimension embedding
    """
    try:
        response = await openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []


async def get_nodes_by_level_and_years(
    level_depth: int,
    fiscal_years: List[str],
    company_ticker: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetches nodes at a specific hierarchy level filtered by fiscal years.
    
    Executes SQL Query:
    ```sql
    SELECT node_id, text_content, topic, fiscal_year, fiscal_quarter, node_metadata
    FROM nodes
    WHERE level_depth = :level_depth
      AND fiscal_year IN (:fiscal_years)
      AND (:company_ticker IS NULL OR company_ticker = :company_ticker)
    ORDER BY fiscal_year, fiscal_quarter;
    ```
    
    Args:
        level_depth: The hierarchy level (3=Year, 2=Quarter, 1=Topic, 0=Chunk)
        fiscal_years: List of fiscal years to filter by
        company_ticker: Optional company filter
        
    Returns:
        List of node dictionaries with id, content, and metadata
    """
    if not DB_AVAILABLE:
        print("[WARNING] Database not available, returning empty list")
        return []
    
    if not fiscal_years:
        return []
    
    async with AsyncSessionLocal() as session:
        try:
            # Build the query dynamically based on filters
            # Using parameterized query for safety
            years_placeholder = ', '.join([f":year_{i}" for i in range(len(fiscal_years))])
            
            query_sql = f"""
                SELECT node_id, text_content, topic, fiscal_year, fiscal_quarter, node_metadata
                FROM nodes
                WHERE level_depth = :level_depth
                  AND fiscal_year IN ({years_placeholder})
            """
            
            # Add company ticker filter if provided
            if company_ticker:
                query_sql += " AND company_ticker = :company_ticker"
            
            query_sql += " ORDER BY fiscal_year, fiscal_quarter"
            
            # Build parameters
            params = {"level_depth": level_depth}
            for i, year in enumerate(fiscal_years):
                params[f"year_{i}"] = year
            if company_ticker:
                params["company_ticker"] = company_ticker
            
            result = await session.execute(text(query_sql), params)
            rows = result.fetchall()
            
            nodes = []
            for row in rows:
                nodes.append({
                    "node_id": str(row[0]),
                    "text_content": row[1],
                    "topic": row[2],
                    "fiscal_year": row[3],
                    "fiscal_quarter": row[4],
                    "node_metadata": row[5] or {}
                })
            
            return nodes
            
        except Exception as e:
            print(f"Error fetching nodes by level and years: {e}")
            return []


async def get_children_nodes(parent_node_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetches all child nodes for given parent node IDs.
    
    Executes SQL Query:
    ```sql
    SELECT node_id, parent_node_id, text_content, topic, fiscal_year, 
           fiscal_quarter, level_depth, node_metadata
    FROM nodes
    WHERE parent_node_id IN (:parent_node_ids)
    ORDER BY fiscal_quarter, topic;
    ```
    
    Args:
        parent_node_ids: List of parent node UUIDs (as strings)
        
    Returns:
        List of child node dictionaries
    """
    if not DB_AVAILABLE:
        print("[WARNING] Database not available, returning empty list")
        return []
    
    if not parent_node_ids:
        return []
    
    async with AsyncSessionLocal() as session:
        try:
            # Build parameterized query for parent IDs
            ids_placeholder = ', '.join([f":id_{i}" for i in range(len(parent_node_ids))])
            
            query_sql = f"""
                SELECT node_id, parent_node_id, text_content, topic, fiscal_year, 
                       fiscal_quarter, level_depth, node_metadata
                FROM nodes
                WHERE parent_node_id IN ({ids_placeholder})
                ORDER BY fiscal_quarter, topic
            """
            
            # Build parameters - convert string UUIDs to proper format
            params = {}
            for i, pid in enumerate(parent_node_ids):
                params[f"id_{i}"] = pid
            
            result = await session.execute(text(query_sql), params)
            rows = result.fetchall()
            
            nodes = []
            for row in rows:
                nodes.append({
                    "node_id": str(row[0]),
                    "parent_node_id": str(row[1]) if row[1] else None,
                    "text_content": row[2],
                    "topic": row[3],
                    "fiscal_year": row[4],
                    "fiscal_quarter": row[5],
                    "level_depth": row[6],
                    "node_metadata": row[7] or {}
                })
            
            return nodes
            
        except Exception as e:
            print(f"Error fetching children nodes: {e}")
            return []


async def vector_search_in_scope(
    query: str,
    parent_topic_ids: List[str],
    chunks_per_topic: int = 5,
    final_top_k: int = 15
) -> List[ChunkWithMetadata]:
    """
    Performs semantic vector search with BALANCED TOPIC COVERAGE.
    
    NEW APPROACH (Per-Topic + Re-rank):
    1. For EACH topic, retrieve top `chunks_per_topic` most similar chunks
    2. Combine all candidates into a single pool
    3. Re-rank the combined pool by similarity score
    4. Return the final `final_top_k` chunks
    
    This ensures no topic is completely excluded due to another topic's
    chunks being more similar to the query.
    
    Example:
        10 topics × 5 chunks/topic = 50 candidates
        Re-rank → return top 15
    
    Args:
        query: The original user query for embedding
        parent_topic_ids: Topic node IDs to scope the search
        chunks_per_topic: How many chunks to retrieve from each topic
        final_top_k: Final number of chunks to return after re-ranking
        
    Returns:
        List of ChunkWithMetadata dicts containing text, similarity, and provenance
    """
    if not DB_AVAILABLE:
        print("[WARNING] Database not available, returning empty list")
        return []
    
    if not parent_topic_ids:
        return []
    
    # Generate embedding for the query
    query_embedding = await generate_query_embedding(query)
    if not query_embedding:
        print("[WARNING] Failed to generate query embedding")
        return []
    
    all_candidates = []
    
    async with AsyncSessionLocal() as session:
        try:
            # Fetch chunks from EACH topic individually
            for topic_id in parent_topic_ids:
                query_sql = """
                    SELECT node_id, text_content, fiscal_year, fiscal_quarter, 
                           company_ticker, node_metadata,
                           1 - (embedding <=> :query_embedding) as similarity
                    FROM nodes
                    WHERE parent_node_id = :topic_id
                      AND level_depth = 0
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> :query_embedding
                    LIMIT :chunks_per_topic
                """
                
                params = {
                    "query_embedding": str(query_embedding),
                    "topic_id": topic_id,
                    "chunks_per_topic": chunks_per_topic
                }
                
                result = await session.execute(text(query_sql), params)
                rows = result.fetchall()
                
                # Add chunks from this topic to the candidate pool
                for row in rows:
                    metadata = row[5] or {}
                    all_candidates.append({
                        "node_id": str(row[0]),
                        "text_content": row[1],
                        "fiscal_year": row[2],
                        "fiscal_quarter": row[3],
                        "company_ticker": row[4],
                        "provenance": metadata.get("provenance", {}),
                        "similarity": float(row[6]) if row[6] else 0.0,
                        "_topic_id": topic_id  # Track source topic for debugging
                    })
            
            # Re-rank all candidates by similarity and take top final_top_k
            all_candidates.sort(key=lambda x: x["similarity"], reverse=True)
            final_chunks = all_candidates[:final_top_k]
            
            # Remove internal tracking field before returning
            for chunk in final_chunks:
                chunk.pop("_topic_id", None)
            
            print(f"[VECTOR_SEARCH] Retrieved {len(all_candidates)} candidates from {len(parent_topic_ids)} topics, returning top {len(final_chunks)}")
            
            return final_chunks
            
        except Exception as e:
            print(f"Error in scoped vector search: {e}")
            return []



async def global_vector_search(
    query: str, 
    top_k: int = 10,
    fiscal_years: Optional[List[str]] = None
) -> List[ChunkWithMetadata]:
    """
    Performs a global semantic search across all chunk nodes (level_depth = 0).
    Used as fallback for direct/simple queries when tree traversal is not needed.
    Returns chunks WITH metadata for provenance tracking.
    
    Now supports OPTIONAL fiscal_years filter to scope results to specific years.
    
    Args:
        query: User query for embedding
        top_k: Number of results
        fiscal_years: Optional list of fiscal years to filter by (e.g., ['FY25'])
        
    Returns:
        List of ChunkWithMetadata dicts with provenance info
    """
    if not DB_AVAILABLE:
        print("[WARNING] Database not available, returning empty list")
        return []
    
    # Generate embedding for the query
    query_embedding = await generate_query_embedding(query)
    if not query_embedding:
        print("[WARNING] Failed to generate query embedding")
        return []
    
    async with AsyncSessionLocal() as session:
        try:
            # Build base query
            if fiscal_years and len(fiscal_years) > 0:
                # Add fiscal year filter
                years_placeholder = ', '.join([f":year_{i}" for i in range(len(fiscal_years))])
                query_sql = f"""
                    SELECT node_id, text_content, fiscal_year, fiscal_quarter,
                           company_ticker, node_metadata,
                           1 - (embedding <=> :query_embedding) as similarity
                    FROM nodes
                    WHERE level_depth = 0
                      AND embedding IS NOT NULL
                      AND fiscal_year IN ({years_placeholder})
                    ORDER BY embedding <=> :query_embedding
                    LIMIT :top_k
                """
                params = {
                    "query_embedding": str(query_embedding),
                    "top_k": top_k
                }
                for i, year in enumerate(fiscal_years):
                    params[f"year_{i}"] = year
            else:
                # No year filter - search all
                query_sql = """
                    SELECT node_id, text_content, fiscal_year, fiscal_quarter,
                           company_ticker, node_metadata,
                           1 - (embedding <=> :query_embedding) as similarity
                    FROM nodes
                    WHERE level_depth = 0
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> :query_embedding
                    LIMIT :top_k
                """
                params = {
                    "query_embedding": str(query_embedding),
                    "top_k": top_k
                }
            
            result = await session.execute(text(query_sql), params)
            rows = result.fetchall()
            
            # Build ChunkWithMetadata for each result
            chunks = []
            for row in rows:
                metadata = row[5] or {}
                chunks.append({
                    "node_id": str(row[0]),
                    "text_content": row[1],
                    "fiscal_year": row[2],
                    "fiscal_quarter": row[3],
                    "company_ticker": row[4],
                    "provenance": metadata.get("provenance", {}),
                    "similarity": float(row[6]) if row[6] else 0.0
                })
            
            return chunks
            
        except Exception as e:
            print(f"Error in global vector search: {e}")
            return []


# =============================================================================
# NODE 1: TIME SCOPER (The Orchestrator)
# =============================================================================

async def time_scoper(state: AgentState) -> AgentState:
    """
    Identifies which Fiscal Years are relevant to the user's query.
    
    Logic:
    - If explicit time reference (e.g., "in FY2023"), restrict to that year
    - If implicit/vague (e.g., "recent changes"), select broad range (last 3-4 FYs)
    - Does NOT extract topics - that's the tree_traverser's job
    
    Returns:
        Updated state with target_years populated
    """
    query = state["query"]
    current_year = datetime.now().year
    
    # System prompt for time extraction
    system_prompt = """You are a financial time period analyzer. 
Your task is to identify which fiscal years a query is asking about.

Rules:
1. If the query mentions specific years (FY24, 2024, fiscal year 2024), return those exact years
2. If the query says "recent", "latest", "current", return the last 2-3 fiscal years
3. If the query says "historical", "over time", "evolution", return the last 4-5 fiscal years
4. If the query is vague about time, default to the last 3 fiscal years
5. Also determine if this is a "direct" simple fact query (single data point) or "complex" (requires analysis)

IMPORTANT: Use TWO-DIGIT year format like FY24, FY25, FY26 (NOT FY2024, FY2025).

Respond in this exact format:
YEARS: FY24, FY25, FY26
QUERY_TYPE: DIRECT or COMPLEX"""

    user_prompt = f"""Query: "{query}"

Current calendar year is {current_year}. 
What fiscal years should we search? Use FY24/FY25/FY26 format (two digits).
Is this a direct or complex query?"""

    try:
        response = await pruning_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        response_text = response.content
        
        # Parse response
        years = []
        is_direct = False
        
        for line in response_text.split('\n'):
            if line.startswith('YEARS:'):
                years_str = line.replace('YEARS:', '').strip()
                years = [y.strip() for y in years_str.split(',')]
                # Normalize any 4-digit years to 2-digit format
                normalized_years = []
                for y in years:
                    # Convert FY2024 -> FY24, FY2025 -> FY25, etc.
                    if len(y) == 6 and y.startswith('FY20'):
                        y = 'FY' + y[4:]  # FY2024 -> FY24
                    normalized_years.append(y)
                years = normalized_years
            elif line.startswith('QUERY_TYPE:'):
                query_type = line.replace('QUERY_TYPE:', '').strip()
                is_direct = query_type.upper() == 'DIRECT'
        
        # Fallback if parsing fails - use two-digit format
        if not years:
            years = [f"FY{str(current_year)[2:]}", f"FY{str(current_year-1)[2:]}", f"FY{str(current_year-2)[2:]}"]
            
        state["target_years"] = years
        state["is_direct_query"] = is_direct
        state["traversal_log"].append(f"[TIME_SCOPER] Identified years: {years}, Direct query: {is_direct}")
        
    except Exception as e:
        # Fallback to last 3 years with two-digit format
        state["target_years"] = [f"FY{str(current_year)[2:]}", f"FY{str(current_year-1)[2:]}", f"FY{str(current_year-2)[2:]}"]
        state["is_direct_query"] = False
        state["traversal_log"].append(f"[TIME_SCOPER] Error: {e}, using fallback years")
    
    return state


# =============================================================================
# NODE 2: TREE TRAVERSER (The BFS Pruning Engine)
# =============================================================================

async def tree_traverser(state: AgentState) -> AgentState:
    """
    Performs Breadth-First Search (BFS) tree traversal with LLM-guided pruning.
    
    The Algorithm:
    1. Level 3 (Year): Fetch year nodes, LLM prunes irrelevant years
    2. Level 2 (Quarter): Fetch children of surviving years, LLM prunes quarters
    3. Level 1 (Topic): Fetch children of surviving quarters, LLM prunes topics
    4. Level 0 (Chunks): Vector search ONLY within valid topic children
    
    Fallback: For direct queries, skip tree walk and do global vector search
    
    Returns:
        Updated state with retrieved_chunks populated
    """
    query = state["query"]
    target_years = state["target_years"]
    company_ticker = state.get("company_ticker")
    
    # -------------------------------------------------------------------------
    # FALLBACK: Direct query bypass
    # -------------------------------------------------------------------------
    if state.get("is_direct_query", False):
        state["traversal_log"].append(f"[TREE_TRAVERSER] Direct query detected, using global vector search (filtered to {target_years})")
        chunks = await global_vector_search(query, top_k=15, fiscal_years=target_years)
        state["retrieved_chunks"] = chunks
        return state
    
    # -------------------------------------------------------------------------
    # LEVEL 3: YEAR PRUNING
    # -------------------------------------------------------------------------
    state["traversal_log"].append(f"[TREE_TRAVERSER] Starting BFS from Year level with targets: {target_years}")
    
    year_nodes = await get_nodes_by_level_and_years(
        level_depth=3, 
        fiscal_years=target_years,
        company_ticker=company_ticker
    )
    
    if not year_nodes:
        state["traversal_log"].append(f"[TREE_TRAVERSER] No year nodes found, falling back to global search (filtered to {target_years})")
        chunks = await global_vector_search(query, top_k=15, fiscal_years=target_years)
        state["retrieved_chunks"] = chunks
        return state
    
    # LLM pruning at Year level
    surviving_year_ids = await _prune_nodes_with_llm(
        query=query,
        nodes=year_nodes,
        level_name="Year",
        state=state
    )
    
    # -------------------------------------------------------------------------
    # LEVEL 2: QUARTER PRUNING  
    # -------------------------------------------------------------------------
    if not surviving_year_ids:
        state["traversal_log"].append(f"[TREE_TRAVERSER] All years pruned, using global search (filtered to {target_years})")
        chunks = await global_vector_search(query, top_k=15, fiscal_years=target_years)
        state["retrieved_chunks"] = chunks
        return state
    
    quarter_nodes = await get_children_nodes(surviving_year_ids)
    
    surviving_quarter_ids = await _prune_nodes_with_llm(
        query=query,
        nodes=quarter_nodes,
        level_name="Quarter",
        state=state
    )
    
    # -------------------------------------------------------------------------
    # LEVEL 1: TOPIC PRUNING
    # -------------------------------------------------------------------------
    if not surviving_quarter_ids:
        state["traversal_log"].append(f"[TREE_TRAVERSER] All quarters pruned, using global search (filtered to {target_years})")
        chunks = await global_vector_search(query, top_k=15, fiscal_years=target_years)
        state["retrieved_chunks"] = chunks
        return state
    
    topic_nodes = await get_children_nodes(surviving_quarter_ids)
    
    surviving_topic_ids = await _prune_nodes_with_llm(
        query=query,
        nodes=topic_nodes,
        level_name="Topic",
        state=state
    )
    
    state["target_topics"] = [{"id": tid} for tid in surviving_topic_ids]
    
    # -------------------------------------------------------------------------
    # LEVEL 0: VECTOR SEARCH (Scoped to surviving topics)
    # -------------------------------------------------------------------------
    if not surviving_topic_ids:
        state["traversal_log"].append(f"[TREE_TRAVERSER] All topics pruned, using global search (filtered to {target_years})")
        chunks = await global_vector_search(query, top_k=15, fiscal_years=target_years)
    else:
        state["traversal_log"].append(f"[TREE_TRAVERSER] Vector search scoped to {len(surviving_topic_ids)} topic(s)")
        # Fetch 5 chunks per topic, then re-rank and return top 15
        chunks = await vector_search_in_scope(query, surviving_topic_ids, chunks_per_topic=5, final_top_k=15)
    
    state["retrieved_chunks"] = chunks
    return state


async def _prune_nodes_with_llm(
    query: str,
    nodes: List[Dict[str, Any]],
    level_name: str,
    state: AgentState
) -> List[str]:
    """
    Helper function: Uses LLM to determine which nodes are relevant.
    
    This is where the "PRUNING" happens:
    - If LLM says a node is irrelevant, we DO NOT fetch its children
    - This dramatically reduces search space for complex queries
    
    Args:
        query: User query
        nodes: Nodes at current level to evaluate
        level_name: Level name for logging
        state: AgentState for logging
        
    Returns:
        List of node_ids that survived pruning
    """
    if not nodes:
        return []
    
    # Build context for LLM
    node_summaries = "\n".join([
        f"[{i+1}] {node.get('topic', 'N/A')} ({node['fiscal_year']}{' ' + node.get('fiscal_quarter', '') if node.get('fiscal_quarter') else ''}): {node['text_content'][:200]}..."
        for i, node in enumerate(nodes)
    ])
    
    system_prompt = f"""You are a financial document retrieval expert.
Your task is to determine which {level_name} nodes are RELEVANT to the user's query.

Be INCLUSIVE - when in doubt, keep the node. It's better to include marginally relevant 
content than to miss important information.

Only exclude nodes that are CLEARLY irrelevant to what the user is asking about."""

    user_prompt = f"""Query: "{query}"

Available {level_name} nodes:
{node_summaries}

Which nodes are relevant? Return ONLY the numbers of relevant nodes as comma-separated values.
Example: 1, 3, 4

If ALL nodes are relevant, return: ALL
If NO nodes are relevant, return: NONE

Relevant nodes:"""

    try:
        response = await pruning_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        result = response.content.strip().upper()
        
        if result == "ALL":
            surviving_ids = [n["node_id"] for n in nodes]
        elif result == "NONE":
            surviving_ids = []
        else:
            # Parse numbers
            try:
                indices = [int(x.strip()) - 1 for x in result.split(',')]
                surviving_ids = [nodes[i]["node_id"] for i in indices if 0 <= i < len(nodes)]
            except:
                # Fallback: keep all
                surviving_ids = [n["node_id"] for n in nodes]
        
        state["traversal_log"].append(
            f"[PRUNE_{level_name.upper()}] {len(nodes)} candidates → {len(surviving_ids)} surviving"
        )
        
        return surviving_ids
        
    except Exception as e:
        state["traversal_log"].append(f"[PRUNE_{level_name.upper()}] Error: {e}, keeping all nodes")
        return [n["node_id"] for n in nodes]


# =============================================================================
# NODE 3: SYNTHESIZER (The Analyst)
# =============================================================================

async def synthesizer(state: AgentState) -> AgentState:
    """
    Generates the final forensic financial answer using retrieved evidence.
    
    Takes the retrieved_chunks and original query to produce a detailed,
    citation-backed analysis using GPT-4o.
    
    Returns:
        Updated state with final_answer populated
    """
    query = state["query"]
    chunks = state["retrieved_chunks"]
    traversal_log = state["traversal_log"]
    
    if not chunks:
        state["final_answer"] = "I could not find relevant information to answer your query. Please try rephrasing or providing more specific time periods."
        return state
    
    # Format evidence with citations - now chunks are dicts with metadata
    evidence_text = "\n\n".join([
        f"[Evidence {i+1}] ({chunk.get('fiscal_year', 'N/A')} {chunk.get('fiscal_quarter', '') or ''}, Page {chunk.get('provenance', {}).get('page_label', 'N/A')}): {chunk['text_content']}"
        for i, chunk in enumerate(chunks)
    ])
    
    system_prompt = """You are a senior financial analyst providing forensic analysis of company documents.

Your response must:
1. Directly answer the user's question with specific facts and figures
2. Cite evidence using [Evidence N] format
3. Identify trends, changes, or patterns across time periods when relevant
4. Note any inconsistencies or areas requiring further investigation
5. Be concise but comprehensive

Do not make up information. Only use what is provided in the evidence."""

    user_prompt = f"""User Question: {query}

Retrieved Evidence:
{evidence_text}

Provide a detailed forensic analysis answering the user's question."""

    try:
        response = await synthesis_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        state["final_answer"] = response.content
        state["traversal_log"].append("[SYNTHESIZER] Generated final answer")
        
    except Exception as e:
        state["final_answer"] = f"Error generating analysis: {str(e)}"
        state["traversal_log"].append(f"[SYNTHESIZER] Error: {e}")
    
    return state


# =============================================================================
# GRAPH COMPILATION
# =============================================================================

def create_graph() -> StateGraph:
    """
    Creates and compiles the AlphaPulse LangGraph.
    
    Flow: time_scoper → tree_traverser → synthesizer → END
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("time_scoper", time_scoper)
    workflow.add_node("tree_traverser", tree_traverser)
    workflow.add_node("synthesizer", synthesizer)
    
    # Define edges (linear flow for this implementation)
    workflow.set_entry_point("time_scoper")
    workflow.add_edge("time_scoper", "tree_traverser")
    workflow.add_edge("tree_traverser", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    # Compile the graph
    return workflow.compile()


# Create the compiled graph instance
app = create_graph()


# =============================================================================
# QUERY INTERFACE
# =============================================================================

async def run_query(
    query: str,
    company_ticker: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for running queries through AlphaPulse.
    
    Args:
        query: User's natural language question
        company_ticker: Optional company filter
        
    Returns:
        Dictionary containing the answer and debug information
    """
    # Initialize state
    initial_state: AgentState = {
        "query": query,
        "target_years": [],
        "target_quarters": [],
        "target_topics": [],
        "retrieved_chunks": [],
        "final_answer": "",
        "traversal_log": [],
        "is_direct_query": False,
        "company_ticker": company_ticker
    }
    
    # Run the graph
    final_state = await app.ainvoke(initial_state)
    
    # Build sources list with provenance for frontend
    sources = []
    for i, chunk in enumerate(final_state["retrieved_chunks"]):
        sources.append({
            "evidence_id": i + 1,
            "text_content": chunk["text_content"],
            "similarity": chunk.get("similarity", 0.0),
            "fiscal_year": chunk.get("fiscal_year"),
            "fiscal_quarter": chunk.get("fiscal_quarter"),
            "company_ticker": chunk.get("company_ticker"),
            "provenance": chunk.get("provenance", {})
        })
    
    return {
        "query": query,
        "answer": final_state["final_answer"],
        "target_years": final_state["target_years"],
        "chunks_retrieved": len(final_state["retrieved_chunks"]),
        "sources": sources,  # Full provenance data for frontend!
        "traversal_log": final_state["traversal_log"]
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import sys
    
    async def main():
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
        else:
            query = "How did the revenue strategy evolve in FY2023?"
        
        print(f"\n{'='*60}")
        print(f"FiscalFlow Query: {query}")
        print(f"{'='*60}\n")
        
        result = await run_query(query)
        
        print("TARGET YEARS:", result["target_years"])
        print(f"CHUNKS RETRIEVED: {result['chunks_retrieved']}")
        print("\nTRAVERSAL LOG:")
        for log in result["traversal_log"]:
            print(f"  {log}")
        print(f"\n{'='*60}")
        print("ANSWER:")
        print(f"{'='*60}")
        print(result["answer"])
    
    asyncio.run(main())
