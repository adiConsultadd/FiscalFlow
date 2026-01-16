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
                # We can group words or just take the full text and reconstruct mappings
                # For semantic chunking, getting the full text with a simple map is often easiest.
                # However, for detailed provenance, let's keep page-level context.
                
                text = page.extract_text()
                if not text:
                    continue
                
                # To get character-level bounding boxes precisely is complex. 
                # Simplification: We will store page-level provenance for the chunks for now,
                # or we can try to map characters back. 
                # Start index for this page in the hypothetical "full document string" 
                # if we were concatenating, but we process page by page or globally.
                
                extracted_data.append({
                    "page_index": i,
                    "page_label": str(page.page_number),
                    "file_name": file_name,
                    "text": text,
                    "words": words, # List of dicts: {'text': '...', 'x0': ..., 'top': ...}
                    "width": page.width,
                    "height": page.height
                })
    except Exception as e:
        print(f"Error parsing PDF {file_path}: {e}")
    
    return extracted_data

# -------------------------------------------------------------------------
# 2. SEMANTIC CHUNKING
# -------------------------------------------------------------------------
def semantic_chunking(parsed_data: List[Dict[str, Any]], chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
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
            
        # Naive sliding window chunking for demonstration
        # A more advanced semantic chunker would strictly respect sentence boundaries
        
        i = 0
        while i < len(text):
            end = min(i + chunk_size, len(text))
            chunk_text = text[i:end]
            
            # Simple metadata - for bbox we would ideally look up the specific words in this range
            # For this implementation, we will use the page bbox as a placeholder or 
            # attempt to find the bounding box of the text if possible provided 'words' data.
            # Calculating exact bbox for a substring is computationally intensive without character map.
            
            provenance = {
                "page_index": page_index,
                "page_label": page_label,
                "file_name": file_name,
                "bbox": [0, 0, float(page_data["width"]), float(page_data["height"])], # Default to full page if exact not calc
                "char_start": i, # Relative to page text start
                "char_end": end
            }
            
            chunks.append({
                "text_content": chunk_text,
                "provenance": provenance,
                "company_ticker": "UNKNOWN", # Placeholder, logic to extract this needed
                "fiscal_year": "FY24"        # Placeholder
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
# 5. DB SAVING
# -------------------------------------------------------------------------
async def save_to_db(nodes_data: List[Dict[str, Any]], session: AsyncSession):
    """
    Saves Node objects to the database.
    """
    db_nodes = []
    for data in nodes_data:
        node = Node(
            text_content=data["text_content"],
            embedding=data["embedding"],
            company_ticker=data.get("company_ticker", "UNKNOWN"),
            fiscal_year=data.get("fiscal_year", "UNKNOWN"),
            fiscal_quarter=data.get("fiscal_quarter"),
            level_depth=1, # E.g., raw chunk
            node_metadata={"provenance": data["provenance"]}
        )
        db_nodes.append(node)
    
    session.add_all(db_nodes)
    await session.commit()
    print(f"Saved {len(db_nodes)} nodes to database.")

# -------------------------------------------------------------------------
# ORCHESTRATOR
# -------------------------------------------------------------------------
async def process_document(file_path: str):
    print(f"Processing {file_path}...")
    
    # 1. Parse
    parsed_data = parse_pdf(file_path)
    print(f"Parsed {len(parsed_data)} pages.")
    
    # 2. Chunk
    chunks = semantic_chunking(parsed_data)
    print(f"Created {len(chunks)} chunks.")
    
    # 3. Embed
    chunks_with_embeddings = await generate_embeddings(chunks)
    print("Generated embeddings.")
    
    # 4. Cluster (Optional step - usually we cluster for higher level summarization)
    # The prompt asks to cluster them. We can use the cluster ID to potentially create parent nodes 
    # or just tag them. For now, let's just run it to show we can.
    clusters = cluster_chunks_semantically(chunks_with_embeddings)
    print(f"Formed {len(clusters)} clusters.")
    
    # 5. Save
    async with AsyncSessionLocal() as session:
        await save_to_db(chunks_with_embeddings, session)

if __name__ == "__main__":
    # Test run
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        asyncio.run(process_document(file_path))
    else:
        print("Please provide a file path as argument.")
