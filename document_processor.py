import os
import asyncio
import uuid
import numpy as np
import pdfplumber
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from openai import AsyncOpenAI
from sklearn.cluster import KMeans
from dotenv import load_dotenv

from database import AsyncSessionLocal
from models import Node

load_dotenv()

# Initialize OpenAI Client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------------------------
# 1. PARSER FUNCTION (Support PDF)
# -------------------------------------------------------------------------
def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Parses a PDF file and extracts text with metadata (page number, bbox, etc.)
    """
    extracted_data = []
    file_name = os.path.basename(file_path)

    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                words = page.extract_words()
                text = page.extract_text()
                if not text:
                    continue
                
                extracted_data.append({
                    "page_index": i,
                    "page_label": str(page.page_number),
                    "file_name": file_name,
                    "text": text,
                    "words": words, 
                    "width": page.width,
                    "height": page.height
                })
    except Exception as e:
        print(f"Error parsing PDF {file_path}: {e}")
    
    return extracted_data

# -------------------------------------------------------------------------
# 2. SEMANTIC CHUNKING
# -------------------------------------------------------------------------
def semantic_chunking(parsed_data: List[Dict[str, Any]], 
                      chunk_size: int = 500, 
                      overlap: int = 50,
                      company_ticker: str = "UNKNOWN",
                      fiscal_year: str = "UNKNOWN",
                      fiscal_quarter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Breaks parsed text into semantic chunks with metadata.
    """
    chunks = []
    
    for page_data in parsed_data:
        text = page_data["text"]
        page_index = page_data["page_index"]
        page_label = page_data["page_label"]
        file_name = page_data["file_name"]
        
        if not text:
            continue
            
        i = 0
        while i < len(text):
            end = min(i + chunk_size, len(text))
            chunk_text = text[i:end]
            
            provenance = {
                "page_index": page_index,
                "page_label": page_label,
                "file_name": file_name,
                "bbox": [0, 0, float(page_data["width"]), float(page_data["height"])], 
                "char_start": i,
                "char_end": end
            }
            
            chunks.append({
                "text_content": chunk_text,
                "provenance": provenance,
                "company_ticker": company_ticker,
                "fiscal_year": fiscal_year,
                "fiscal_quarter": fiscal_quarter
            })
            
            i += (chunk_size - overlap)
            
    return chunks

# -------------------------------------------------------------------------
# 3. EMBEDDING GENERATION
# -------------------------------------------------------------------------
async def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates embeddings for a list of text chunks using OpenAI.
    """
    texts = [c["text_content"] for c in chunks]
    if not texts:
        return chunks

    try:
        # Batch this if you have many chunks
        response = await client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        
        for i, data in enumerate(response.data):
            chunks[i]["embedding"] = data.embedding
            
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        
    return chunks

# -------------------------------------------------------------------------
# 4. K-MEANS CLUSTERING
# -------------------------------------------------------------------------
def cluster_chunks_semantically(nodes: List[Dict[str, Any]], num_clusters: Optional[int] = None) -> Dict[int, List[Dict[str, Any]]]:
    """
    Clusters chunks based on their embeddings.
    """
    if not nodes:
        return {}
        
    embeddings = np.array([n["embedding"] for n in nodes])
    
    if len(embeddings) < 2:
        return {0: nodes}

    # Determine Cluster Count
    if not num_clusters:
        num_clusters = max(1, len(nodes) // 5)
        num_clusters = min(num_clusters, 10)

    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        labels = kmeans.labels_

        clustered_nodes = {}
        for i, label in enumerate(labels):
            if label not in clustered_nodes:
                clustered_nodes[label] = []
            clustered_nodes[label].append(nodes[i])
            
        return clustered_nodes
    except Exception as e:
        print(f"Clustering error: {e}")
        return {0: nodes} # Fallback

# -------------------------------------------------------------------------
# 5. SUMMARIZATION ENGINE
# -------------------------------------------------------------------------
async def summarize_cluster(cluster_nodes: List[Dict[str, Any]], fiscal_meta: str = "Fiscal Data") -> tuple[str, str]:
    """
    Summarizes a list of nodes into a single 'Topic Node'.
    """
    # Combine text from all chunks in this cluster
    context_text = "\n\n".join([n["text_content"] for n in cluster_nodes])
    
    # PROMPT: The 'Secret Sauce' for specific summaries
    prompt = f"""
    You are analyzing specific segments of a company's financial transcript for {fiscal_meta}.
    Below are several raw text chunks that the system has identified as semantically similar.
    
    RAW DATA:
    {context_text}
    
    TASK:
    1. Identify the single specific topic binding these chunks (e.g., "Cloud Revenue", "Employee Attrition", "Supply Chain Issues").
    2. Write a detailed summary (3-5 sentences) capturing the specific numbers, facts, and sentiment found in these chunks.
    3. Do not be vague. Use actual figures if present.
    
    OUTPUT FORMAT:
    Topic: [Topic Name]
    Summary: [Detailed Summary]
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini", # Or slightly larger if needed, but mini is good for this
            messages=[
                {"role": "system", "content": "You are a helpful financial analyst assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        response_text = response.choices[0].message.content
        
        # Simple parsing
        topic_title = "General Topic"
        summary_body = response_text
        
        lines = response_text.split('\n')
        topic_line = next((line for line in lines if "Topic:" in line), None)
        summary_start = next((i for i, line in enumerate(lines) if "Summary:" in line), None)
        
        if topic_line:
            topic_title = topic_line.replace("Topic:", "").strip()
            
        if summary_start is not None:
            summary_body = "\n".join(lines[summary_start:]).replace("Summary:", "").strip()
            
        return topic_title, summary_body
    except Exception as e:
        print(f"Summarization error: {e}")
        return "Error Topic", "Error creating summary."


# -------------------------------------------------------------------------
# 6. DB SAVING
# -------------------------------------------------------------------------
async def save_hierarchy_to_db(clusters: Dict[int, List[Dict[str, Any]]], session: AsyncSession):
    """
    Saves hierarchical Node objects (Level 1 Topics -> Level 0 Chunks) to the database.
    """
    
    for cluster_id, nodes_data in clusters.items():
        # 0. Pre-generate IDs for children to link and store in parent metadata
        child_ids = [uuid.uuid4() for _ in nodes_data]
        
        # 1. Summarize Cluster to get Topic Node info
        fiscal_meta = f"{nodes_data[0].get('company_ticker', '')} {nodes_data[0].get('fiscal_year', '')}"
        topic, summary = await summarize_cluster(nodes_data, fiscal_meta=fiscal_meta)
        
        # 2. Key Step: Generate Embedding for the Summary (Level 1)
        summary_node_data = [{"text_content": summary}]
        await generate_embeddings(summary_node_data)
        summary_embedding = summary_node_data[0].get("embedding")

        # 3. Create Topic Node (Level 1)
        topic_node = Node(
            node_id=uuid.uuid4(),
            text_content=summary, 
            topic=topic,          
            embedding=summary_embedding, 
            company_ticker=nodes_data[0].get("company_ticker", "UNKNOWN"),
            fiscal_year=nodes_data[0].get("fiscal_year", "UNKNOWN"),
            fiscal_quarter=nodes_data[0].get("fiscal_quarter"),
            level_depth=1, 
            node_metadata={
                "cluster_id": int(cluster_id),
                "child_node_ids": [str(uid) for uid in child_ids], 
                "child_count": len(nodes_data)
            }
        )
        
        session.add(topic_node)
        await session.flush()
        
        # 4. Create Chunk Nodes (Level 0) linked to Topic
        for i, data in enumerate(nodes_data):
            chunk_node = Node(
                node_id=child_ids[i], 
                parent_node_id=topic_node.node_id,
                text_content=data["text_content"],
                embedding=data["embedding"],
                company_ticker=data.get("company_ticker", "UNKNOWN"),
                fiscal_year=data.get("fiscal_year", "UNKNOWN"),
                fiscal_quarter=data.get("fiscal_quarter"),
                level_depth=0, 
                node_metadata={"provenance": data["provenance"]}
            )
            session.add(chunk_node)
            
    await session.commit()
    print("Saved hierarchy to database.")

# -------------------------------------------------------------------------
# ORCHESTRATOR
# -------------------------------------------------------------------------
async def process_document(file_path: str, 
                           company_ticker: str = "UNKNOWN", 
                           fiscal_year: str = "UNKNOWN", 
                           fiscal_quarter: Optional[str] = None):
    print(f"Processing {file_path} for {company_ticker} {fiscal_year} {fiscal_quarter}...")
    
    # 1. Parse
    parsed_data = parse_pdf(file_path)
    print(f"Parsed {len(parsed_data)} pages.")
    
    # 2. Chunk
    chunks = semantic_chunking(parsed_data, 
                               company_ticker=company_ticker, 
                               fiscal_year=fiscal_year, 
                               fiscal_quarter=fiscal_quarter)
    print(f"Created {len(chunks)} chunks.")
    
    # 3. Embed
    chunks_with_embeddings = await generate_embeddings(chunks)
    print("Generated embeddings.")
    
    # 4. Cluster
    clusters = cluster_chunks_semantically(chunks_with_embeddings)
    print(f"Formed {len(clusters)} clusters.")
    
    # 5. Summarize & Save Hierarchy
    async with AsyncSessionLocal() as session:
        await save_hierarchy_to_db(clusters, session)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        asyncio.run(process_document(file_path))
    else:
        print("Please provide a file path as argument.")
