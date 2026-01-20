import os
import asyncio
import uuid
import numpy as np
import pdfplumber
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from openai import AsyncOpenAI
from sklearn.cluster import KMeans
from dotenv import load_dotenv
import re

from database import AsyncSessionLocal
from models import Node

load_dotenv()

# Initialize OpenAI Client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------------------------
# IMPROVED MARKER PARSER WITH BETTER PAGE TRACKING
# -------------------------------------------------------------------------
def parse_pdf_with_marker(file_path: str) -> List[Dict[str, Any]]:
    """
    Parses a PDF using Marker (OCR) and extracts text blocks with BBox metadata.
    Enhanced to maintain page-level information even in fallback scenarios.
    """
    print(f"Parsing {file_path} with Marker (OCR)...")
    extracted_data = []
    file_name = os.path.basename(file_path)

    try:
        # Initialize marker
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        rendered = converter(file_path)
        
        # Traverse the Rendered object
        if hasattr(rendered, 'children'):
            for i, page in enumerate(rendered.children):
                page_width = 0
                page_height = 0
                
                # Try to get page dimensions
                if hasattr(page, 'bbox') and page.bbox:
                    page_width = page.bbox[2] - page.bbox[0]
                    page_height = page.bbox[3] - page.bbox[1]
                
                # Collect all text from this page
                page_blocks = []
                
                if hasattr(page, 'children'):
                    for block in page.children:
                        # Extract text
                        text = ""
                        if hasattr(block, 'html'):
                            text = block.html
                        elif hasattr(block, 'text'):
                            text = block.text
                        elif hasattr(block, 'lines'):
                            text = "\n".join([getattr(l, 'html', str(l)) for l in block.lines])
                        else:
                            text = str(block)

                        # BBox
                        bbox = [0, 0, 0, 0]
                        if hasattr(block, 'bbox') and block.bbox:
                            bbox = list(block.bbox)
                        elif hasattr(block, 'polygon') and block.polygon:
                            xs = [p[0] for p in block.polygon]
                            ys = [p[1] for p in block.polygon]
                            if xs and ys:
                                bbox = [min(xs), min(ys), max(xs), max(ys)]
                        
                        if text.strip():
                            page_blocks.append({
                                "text": text,
                                "bbox": bbox
                            })
                
                # If we got blocks for this page, add them
                if page_blocks:
                    for block_data in page_blocks:
                        extracted_data.append({
                            "page_index": i,
                            "page_label": str(i + 1),
                            "file_name": file_name,
                            "text": block_data["text"],
                            "width": page_width,
                            "height": page_height,
                            "bbox": block_data["bbox"],
                            "is_markdown": True
                        })
                            
        # IMPROVED FALLBACK: Split by page markers if traversal failed
        if not extracted_data:
            print("Marker traversal yielded no blocks, attempting smart fallback...")
            full_text, _, _ = text_from_rendered(rendered)
            
            # Try to split by page markers in the markdown
            # Marker often includes page breaks or we can split by form feeds
            page_splits = split_text_into_pages(full_text)
            
            if len(page_splits) > 1:
                # We successfully split into pages
                for idx, page_text in enumerate(page_splits):
                    if page_text.strip():
                        extracted_data.append({
                            "page_index": idx,
                            "page_label": str(idx + 1),
                            "file_name": file_name,
                            "text": page_text,
                            "width": 0,
                            "height": 0,
                            "bbox": [0, 0, 0, 0],
                            "is_markdown": True
                        })
            else:
                # Last resort: use PDFPlumber for page info, Marker text for content
                print("Using hybrid approach: PDFPlumber pages + Marker text...")
                extracted_data = hybrid_parse(file_path, full_text)

    except Exception as e:
        print(f"Error parsing PDF with Marker {file_path}: {e}")
        print("Falling back to PDFPlumber...")
        return parse_pdf(file_path)
    
    return extracted_data


def split_text_into_pages(text: str) -> List[str]:
    """
    Attempts to split Marker output into pages using common delimiters.
    """
    # Try different page break patterns
    patterns = [
        r'\n---\n',  # Horizontal rule (common in Marker output)
        r'\f',        # Form feed character
        r'\n##\s*Page\s*\d+',  # Page headers
        r'\n#{1,2}\s*\d+\s*\n',  # Numbered headers
    ]
    
    for pattern in patterns:
        pages = re.split(pattern, text)
        if len(pages) > 1:
            return pages
    
    # No clear delimiter found
    return [text]


def hybrid_parse(file_path: str, marker_text: str) -> List[Dict[str, Any]]:
    """
    Uses PDFPlumber to get page boundaries, but uses Marker's OCR text.
    This maintains page-level accuracy even when Marker's structure is unclear.
    """
    extracted_data = []
    file_name = os.path.basename(file_path)
    
    try:
        with pdfplumber.open(file_path) as pdf:
            # Get total character count to estimate page breaks
            total_chars = len(marker_text)
            chars_per_page = total_chars // len(pdf.pages) if len(pdf.pages) > 0 else total_chars
            
            for i, page in enumerate(pdf.pages):
                # Estimate text for this page
                start_char = i * chars_per_page
                end_char = min((i + 1) * chars_per_page, total_chars)
                
                # Add some overlap to avoid cutting mid-sentence
                if i < len(pdf.pages) - 1:
                    end_char += 100  # Look ahead 100 chars
                    # Try to break at sentence end
                    chunk = marker_text[start_char:end_char]
                    last_period = chunk.rfind('.')
                    if last_period > chars_per_page - 200:  # If period is reasonably close to target
                        end_char = start_char + last_period + 1
                
                page_text = marker_text[start_char:end_char]
                
                if page_text.strip():
                    extracted_data.append({
                        "page_index": i,
                        "page_label": str(page.page_number),
                        "file_name": file_name,
                        "text": page_text,
                        "width": page.width,
                        "height": page.height,
                        "bbox": [0, 0, page.width, page.height],
                        "is_markdown": True
                    })
                    
    except Exception as e:
        print(f"Error in hybrid parse: {e}")
        # Ultimate fallback: just return the full text with estimated pages
        estimated_pages = max(1, len(marker_text) // 3000)  # ~3000 chars per page
        chars_per_page = len(marker_text) // estimated_pages
        
        for i in range(estimated_pages):
            start = i * chars_per_page
            end = min((i + 1) * chars_per_page, len(marker_text))
            extracted_data.append({
                "page_index": i,
                "page_label": str(i + 1),
                "file_name": file_name,
                "text": marker_text[start:end],
                "width": 612,  # Standard letter width in points
                "height": 792,  # Standard letter height in points
                "bbox": [0, 0, 612, 792],
                "is_markdown": True
            })
    
    return extracted_data


def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Original PDFPlumber parser - kept as ultimate fallback.
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
                    "height": page.height,
                    "bbox": [0, 0, page.width, page.height]
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
                "bbox": page_data.get("bbox", [0, 0, 0, 0]), 
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
# 5.1 GENERIC SUMMARIZER FOR HIGHER LEVELS
# -------------------------------------------------------------------------
async def summarize_text_list(texts: List[str], context: str) -> str:
    """
    Summarizes a list of text content into a cohesive narrative.
    """
    if not texts:
        return "No content available."
        
    combined_text = "\n\n---\n\n".join(texts)
    
    prompt = f"""
    You are a financial analyst. Synthesize the following texts into a cohesive summary for: {context}.
    
    INPUT TEXTS:
    {combined_text[:20000]} # Truncate if too long, though higher levels usually fit
    
    TASK:
    Write a comprehensive narrative summary (1-2 paragraphs). Focus on key financial trends, strategic updates, and risks.
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful financial analyst assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Level summarization error: {e}")
        return "Error generating summary."

# -------------------------------------------------------------------------
# 5.2 HIERARCHY BUILDER (L2, L3, L4)
# -------------------------------------------------------------------------
async def build_hierarchy_layers(company: str, year: str, quarter: Optional[str], session: AsyncSession):
    """
    Builds Level 2 (Quarter), Level 3 (Year), and Level 4 (Root) nodes.
    """
    print(f"Building hierarchy for {company} {year} {quarter if quarter else ''}...")

    # --- LEVEL 2: QUARTER NODE ---
    if quarter:
        # 1. Fetch L1 (Topic) Nodes
        stmt = select(Node).where(
            Node.company_ticker == company,
            Node.fiscal_year == year,
            Node.fiscal_quarter == quarter,
            Node.level_depth == 1
        )
        result = await session.execute(stmt)
        l1_nodes = result.scalars().all()
        
        if l1_nodes:
            # 2. Summarize
            l1_texts = [f"Topic: {n.topic}\n{n.text_content}" for n in l1_nodes]
            q_summary = await summarize_text_list(l1_texts, f"{company} {year} {quarter}")
            
            # 3. Create or Update L2 Node
            # Check if exists
            stmt_check = select(Node).where(
                Node.company_ticker == company,
                Node.fiscal_year == year,
                Node.fiscal_quarter == quarter,
                Node.level_depth == 2
            )
            result = await session.execute(stmt_check)
            l2_node = result.scalar_one_or_none()
            
            if not l2_node:
                l2_node = Node(
                    node_id=uuid.uuid4(),
                    text_content=q_summary,
                    topic=f"Overview {year} {quarter}",
                    company_ticker=company,
                    fiscal_year=year,
                    fiscal_quarter=quarter,
                    level_depth=2,
                    node_metadata={"child_count": len(l1_nodes)}
                )
                session.add(l2_node)
            else:
                l2_node.text_content = q_summary
                l2_node.node_metadata["child_count"] = len(l1_nodes)
                session.add(l2_node)
            
            await session.flush()
            
            # 4. Link L1s to L2
            for n in l1_nodes:
                n.parent_node_id = l2_node.node_id
                session.add(n)
            
            print(f"Updated L2 Node for {quarter}")

    # --- LEVEL 3: YEAR NODE ---
    # 1. Fetch L2 (Quarter) Nodes
    stmt = select(Node).where(
        Node.company_ticker == company,
        Node.fiscal_year == year,
        Node.level_depth == 2
    )
    result = await session.execute(stmt)
    l2_nodes = result.scalars().all()
    
    if l2_nodes:
        # 2. Summarize
        l2_texts = [f"Quarter: {n.fiscal_quarter}\n{n.text_content}" for n in l2_nodes]
        y_summary = await summarize_text_list(l2_texts, f"{company} {year}")
        
        # 3. Create or Update L3 Node
        stmt_check = select(Node).where(
            Node.company_ticker == company,
            Node.fiscal_year == year,
            Node.level_depth == 3
        )
        result = await session.execute(stmt_check)
        l3_node = result.scalar_one_or_none()
        
        if not l3_node:
            l3_node = Node(
                node_id=uuid.uuid4(),
                text_content=y_summary,
                topic=f"Yearly Narrative {year}",
                company_ticker=company,
                fiscal_year=year,
                fiscal_quarter=None,
                level_depth=3,
                node_metadata={"child_count": len(l2_nodes)}
            )
            session.add(l3_node)
        else:
            l3_node.text_content = y_summary
            l3_node.node_metadata["child_count"] = len(l2_nodes)
            session.add(l3_node)
            
        await session.flush()
        
        # 4. Link L2s to L3
        for n in l2_nodes:
            n.parent_node_id = l3_node.node_id
            session.add(n)
            
        print(f"Updated L3 Node for {year}")

    # --- LEVEL 4: ROOT NODE ---
    # 1. Fetch L3 (Year) Nodes
    stmt = select(Node).where(
        Node.company_ticker == company,
        Node.level_depth == 3
    )
    result = await session.execute(stmt)
    l3_nodes = result.scalars().all()
    
    if l3_nodes:
        # 2. Summarize
        l3_texts = [f"Year: {n.fiscal_year}\n{n.text_content}" for n in l3_nodes]
        root_summary = await summarize_text_list(l3_texts, f"{company} Narrative Arc")
        
        # 3. Create or Update L4 Node
        stmt_check = select(Node).where(
            Node.company_ticker == company,
            Node.level_depth == 4
        )
        result = await session.execute(stmt_check)
        l4_node = result.scalar_one_or_none()
        
        if not l4_node:
            l4_node = Node(
                node_id=uuid.uuid4(),
                text_content=root_summary,
                topic=f"{company} Narrative Arc",
                company_ticker=company,
                fiscal_year="ALL",
                fiscal_quarter=None,
                level_depth=4,
                node_metadata={"child_count": len(l3_nodes)}
            )
            session.add(l4_node)
        else:
            l4_node.text_content = root_summary
            l4_node.node_metadata["child_count"] = len(l3_nodes)
            session.add(l4_node)
            
        await session.flush()
        
        # 4. Link L3s to L4
        for n in l3_nodes:
            n.parent_node_id = l4_node.node_id
            session.add(n)
            
        print(f"Updated L4 Root Node for {company}")


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
    # parsed_data = parse_pdf(file_path) # Old parser
    parsed_data = parse_pdf_with_marker(file_path) # New Marker parser
    print(f"Parsed {len(parsed_data)} blocks/pages.")
    
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
    
    # 5. Summarize & Save Hierarchy (L0 & L1)
    async with AsyncSessionLocal() as session:
        await save_hierarchy_to_db(clusters, session)
        
        # 6. Build Higher Layers (L2, L3, L4)
        await build_hierarchy_layers(company_ticker, fiscal_year, fiscal_quarter, session)
        await session.commit() # Commit all hierarchy changes


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        asyncio.run(process_document(file_path))
    else:
        print("Please provide a file path as argument.")
