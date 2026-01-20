import os
import asyncio
import uuid
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from openai import AsyncOpenAI
from openpyxl import load_workbook
from dotenv import load_dotenv

from database import AsyncSessionLocal
from models import Node
from document_processor import generate_embeddings, build_hierarchy_layers

load_dotenv()

# Initialize OpenAI Client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Financial metrics that we expect in the Excel file
FINANCIAL_METRICS = [
    "Sales",
    "Expenses",
    "Operating Profit",
    "Other Income",
    "Depreciation",
    "Interest",
    "Profit before tax",
    "Tax",
    "Net profit",
    "OPM"
]

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------

def parse_quarter_to_fiscal_year(quarter_str: str) -> Tuple[str, str]:
    """
    Parses a quarter string like 'Jun-23' and returns (fiscal_year, fiscal_quarter).
    
    Indian fiscal year runs from April to March.
    Q1: Apr-Jun, Q2: Jul-Sep, Q3: Oct-Dec, Q4: Jan-Mar
    
    Examples:
    - Jun-23 -> FY24 Q1 (since Jun falls in Apr-Jun of FY24)
    - Sep-23 -> FY24 Q2
    - Dec-23 -> FY24 Q3
    - Mar-24 -> FY24 Q4
    """
    try:
        # Parse the quarter string
        month_abbr, year_short = quarter_str.strip().split('-')
        
        # Map month abbreviations
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        
        month = month_map.get(month_abbr, 0)
        if month == 0:
            raise ValueError(f"Invalid month: {month_abbr}")
        
        # Convert 2-digit year to 4-digit
        year = int('20' + year_short)
        
        # Determine fiscal year and quarter based on Indian fiscal year (Apr-Mar)
        if 4 <= month <= 6:  # Apr-Jun
            fiscal_year = f"FY{str(year - 2000 + 1).zfill(2)}"
            fiscal_quarter = "Q1"
        elif 7 <= month <= 9:  # Jul-Sep
            fiscal_year = f"FY{str(year - 2000 + 1).zfill(2)}"
            fiscal_quarter = "Q2"
        elif 10 <= month <= 12:  # Oct-Dec
            fiscal_year = f"FY{str(year - 2000 + 1).zfill(2)}"
            fiscal_quarter = "Q3"
        else:  # Jan-Mar
            fiscal_year = f"FY{str(year - 2000).zfill(2)}"
            fiscal_quarter = "Q4"
        
        return fiscal_year, fiscal_quarter
    
    except Exception as e:
        print(f"Error parsing quarter '{quarter_str}': {e}")
        return "UNKNOWN", "UNKNOWN"


# -------------------------------------------------------------------------
# EXCEL PARSING
# -------------------------------------------------------------------------

def parse_quarters_sheet(file_path: str) -> List[Dict[str, Any]]:
    """
    Parses quarterly financial data from the 'Data Sheet' which contains the actual values.
    The 'Quarters' sheet contains formulas that reference the Data Sheet.
    
    Returns a list of dictionaries, each containing:
    - quarter: Quarter string (e.g., 'Jun-23')
    - fiscal_year: Fiscal year (e.g., 'FY24')
    - fiscal_quarter: Quarter (e.g., 'Q1')
    - metrics: Dict of financial metrics
    """
    print(f"Parsing Excel file: {file_path}")
    
    try:
        import pandas as pd
        
        # Read the Data Sheet which has the actual data
        df = pd.read_excel(file_path, sheet_name='Data Sheet', header=None)
        
        print(f"Data Sheet shape: {df.shape}")
        
        # Find the "Quarters" section and "Report Date" row
        # There may be multiple "Report Date" rows (yearly and quarterly)
        # We want the one under the "Quarters" section
        
        # First find the row with "Quarters" label
        quarters_row_idx = None
        for idx, row in df.iterrows():
            if row[0] == 'Quarters':
                quarters_row_idx = idx
                break
        
        if quarters_row_idx is None:
            print("Error: Could not find 'Quarters' section in Data Sheet")
            return []
        
        print(f"Found 'Quarters' section at row {quarters_row_idx}")
        
        # Now find "Report Date" row after the Quarters section
        report_date_row_idx = None
        for idx in range(quarters_row_idx + 1, min(quarters_row_idx + 10, df.shape[0])):
            if df.iloc[idx, 0] == 'Report Date':
                report_date_row_idx = idx
                break
        
        if report_date_row_idx is None:
            print("Error: Could not find 'Report Date' row after 'Quarters' section")
            return []
        
        print(f"Found 'Report Date' at row {report_date_row_idx}")
        
        # Extract quarter dates from the Report Date row
        from datetime import datetime
        
        quarter_dates = []
        for col_idx in range(1, df.shape[1]):  # Skip column 0 (labels)
            date_val = df.iloc[report_date_row_idx, col_idx]
            if pd.notna(date_val) and isinstance(date_val, (datetime, pd.Timestamp)):
                quarter_dates.append((col_idx, date_val))
        
        print(f"Found {len(quarter_dates)} quarters")
        
        # Now extract financial metrics
        # Rows after Report Date contain the financial data
        metrics_mapping = {
            'Sales': 'Sales',
            'Expenses': 'Expenses',
            'Other Income': 'Other Income', 
            'Depreciation': 'Depreciation',
            'Interest': 'Interest',
            'Profit before tax': 'Profit before tax',
            'Tax': 'Tax',
            'Net profit': 'Net profit',
            'Operating Profit': 'Operating Profit'
        }
        
        # Build data for each quarter
        quarterly_data = []
        
        for col_idx, date_val in quarter_dates:
            # Convert date to quarter string format (e.g., "Jun-23")
            month_abbr = date_val.strftime('%b')
            year_short = date_val.strftime('%y')
            quarter_str = f"{month_abbr}-{year_short}"
            
            # Determine fiscal year and quarter
            fiscal_year, fiscal_quarter = parse_quarter_to_fiscal_year(quarter_str)
            
            metrics = {}
            
            # Extract metrics from rows below Report Date
            for row_idx in range(report_date_row_idx + 1, df.shape[0]):
                label = df.iloc[row_idx, 0]
                
                if pd.notna(label) and str(label).strip() in metrics_mapping:
                    metric_name = str(label).strip()
                    metric_value = df.iloc[row_idx, col_idx]
                    
                    if pd.notna(metric_value):
                        metrics[metric_name] = metric_value
            
            # Calculate OPM if not present
            if 'Operating Profit' in metrics and 'Sales' in metrics:
                if metrics['Sales'] != 0:
                    opm = (metrics['Operating Profit'] / metrics['Sales']) * 100
                    metrics['OPM'] = f"{opm:.0f}%"
                else:
                    metrics['OPM'] = "N/A"
            
            if metrics:
                quarterly_data.append({
                    'quarter': quarter_str,
                    'fiscal_year': fiscal_year,
                    'fiscal_quarter': fiscal_quarter,
                    'metrics': metrics
                })
                print(f"  Extracted {quarter_str} ({fiscal_year} {fiscal_quarter}) with {len(metrics)} metrics")
        
        print(f"Successfully parsed data for {len(quarterly_data)} quarters")
        return quarterly_data
    
    except Exception as e:
        print(f"Error parsing Excel file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


# -------------------------------------------------------------------------
# LLM NARRATIVE GENERATION
# -------------------------------------------------------------------------

async def generate_quarter_narrative(quarter_data: Dict[str, Any], company_ticker: str) -> str:
    """
    Uses LLM to convert quarterly financial metrics into a narrative description.
    This makes the data more suitable for embedding and semantic search.
    """
    quarter = quarter_data['quarter']
    metrics = quarter_data['metrics']
    
    # Build a structured prompt
    metrics_text = "\n".join([f"- {key}: {value}" for key, value in metrics.items()])
    
    prompt = f"""You are a financial analyst assistant. Convert the following quarterly financial metrics into a clear, detailed narrative description.

Company: {company_ticker}
Quarter: {quarter}

Financial Metrics:
{metrics_text}

TASK:
Write a comprehensive narrative (3-4 sentences) that:
1. Clearly states the quarter and key financial performance indicators
2. Mentions specific numbers for Sales, Operating Profit, Net Profit, and OPM
3. Highlights trends (positive or negative) and any notable observations
4. Uses professional financial language

Keep it factual and data-driven. This narrative will be used for semantic search and analysis.

OUTPUT (narrative only, no extra formatting):"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional financial analyst who creates clear, factual narratives from financial data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        narrative = response.choices[0].message.content.strip()
        return narrative
    
    except Exception as e:
        print(f"Error generating narrative for {quarter}: {e}")
        # Fallback to simple description
        return f"Quarter {quarter} for {company_ticker}: {metrics_text}"


# -------------------------------------------------------------------------
# CHUNK CREATION AND DB STORAGE
# -------------------------------------------------------------------------

async def create_financial_chunks(quarterly_data: List[Dict[str, Any]], 
                                  company_ticker: str,
                                  session: AsyncSession) -> Dict[str, List[Dict[str, Any]]]:
    """
    Creates chunks from quarterly financial data, organized by fiscal year and quarter.
    Returns a dict mapping (fiscal_year, fiscal_quarter) to list of chunk data.
    """
    chunks_by_period = {}
    
    for quarter_data in quarterly_data:
        fiscal_year = quarter_data['fiscal_year']
        fiscal_quarter = quarter_data['fiscal_quarter']
        
        # Generate narrative
        narrative = await generate_quarter_narrative(quarter_data, company_ticker)
        
        print(f"Generated narrative for {quarter_data['quarter']} ({fiscal_year} {fiscal_quarter})")
        
        # Create chunk data
        chunk = {
            "text_content": narrative,
            "company_ticker": company_ticker,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
            "provenance": {
                "source": "excel",
                "quarter": quarter_data['quarter'],
                "metrics": quarter_data['metrics']
            }
        }
        
        key = (fiscal_year, fiscal_quarter)
        if key not in chunks_by_period:
            chunks_by_period[key] = []
        
        chunks_by_period[key].append(chunk)
    
    return chunks_by_period


async def save_financial_data_to_db(chunks_by_period: Dict[Tuple[str, str], List[Dict[str, Any]]], 
                                    company_ticker: str,
                                    session: AsyncSession):
    """
    Saves financial data chunks to the database with proper hierarchy.
    Creates Level 0 chunks and a Level 1 "Revenue" topic node.
    """
    for (fiscal_year, fiscal_quarter), chunks in chunks_by_period.items():
        print(f"Processing {company_ticker} {fiscal_year} {fiscal_quarter} financial data...")
        
        # Generate embeddings for all chunks
        chunks_with_embeddings = await generate_embeddings(chunks)
        
        # Pre-generate IDs for children
        child_ids = [uuid.uuid4() for _ in chunks_with_embeddings]
        
        # Create summary for Topic Node (Level 1)
        topic_text = "\n\n".join([c["text_content"] for c in chunks_with_embeddings])
        
        # Generate embedding for the topic summary
        topic_node_data = [{"text_content": topic_text}]
        await generate_embeddings(topic_node_data)
        topic_embedding = topic_node_data[0].get("embedding")
        
        # Create Level 1 Topic Node with "Revenue" topic
        topic_node = Node(
            node_id=uuid.uuid4(),
            text_content=topic_text,
            topic="Revenue",  # Fixed topic for easy filtering
            embedding=topic_embedding,
            company_ticker=company_ticker,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            level_depth=1,
            node_metadata={
                "source": "excel",
                "child_node_ids": [str(uid) for uid in child_ids],
                "child_count": len(chunks_with_embeddings)
            }
        )
        
        session.add(topic_node)
        await session.flush()
        
        # Create Level 0 Chunk Nodes
        for i, chunk_data in enumerate(chunks_with_embeddings):
            chunk_node = Node(
                node_id=child_ids[i],
                parent_node_id=topic_node.node_id,
                text_content=chunk_data["text_content"],
                embedding=chunk_data["embedding"],
                company_ticker=company_ticker,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                level_depth=0,
                node_metadata={"provenance": chunk_data["provenance"]}
            )
            session.add(chunk_node)
        
        print(f"Saved {len(chunks_with_embeddings)} chunks with 'Revenue' topic for {fiscal_year} {fiscal_quarter}")
    
    await session.commit()


# -------------------------------------------------------------------------
# ORCHESTRATOR
# -------------------------------------------------------------------------

async def process_excel_document(file_path: str, company_ticker: str = "UNKNOWN"):
    """
    Main orchestrator for processing Excel files with financial data.
    """
    print(f"\n{'='*60}")
    print(f"Processing Excel: {file_path}")
    print(f"Company: {company_ticker}")
    print(f"{'='*60}\n")
    
    # 1. Parse the Quarters sheet
    quarterly_data = parse_quarters_sheet(file_path)
    
    if not quarterly_data:
        print("No quarterly data found in Excel file")
        return
    
    # 2. Create chunks organized by fiscal period
    async with AsyncSessionLocal() as session:
        chunks_by_period = await create_financial_chunks(quarterly_data, company_ticker, session)
        
        # 3. Save to database
        await save_financial_data_to_db(chunks_by_period, company_ticker, session)
        
        # 4. Build higher hierarchy layers for each unique fiscal year/quarter
        processed_periods = set()
        for (fiscal_year, fiscal_quarter) in chunks_by_period.keys():
            if (fiscal_year, fiscal_quarter) not in processed_periods:
                await build_hierarchy_layers(company_ticker, fiscal_year, fiscal_quarter, session)
                processed_periods.add((fiscal_year, fiscal_quarter))
        
        await session.commit()
    
    print(f"\n{'='*60}")
    print(f"Completed Excel processing for {company_ticker}")
    print(f"{'='*60}\n")


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
        # Extract company from path if possible
        # Expected: documents/company/finance.xlsx
        company = "UNKNOWN"
        rel_path = os.path.relpath(file_path, "documents")
        parts = rel_path.split(os.sep)
        if len(parts) >= 1:
            company = parts[0]
        
        asyncio.run(process_excel_document(file_path, company_ticker=company))
    else:
        print("Usage: python excel_processor.py <path_to_excel_file>")
