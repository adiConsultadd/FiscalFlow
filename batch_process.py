
import os
import asyncio
from document_processor import process_document
from excel_processor import process_excel_document

DOCUMENTS_DIR = "documents"

async def batch_process():
    for root, dirs, files in os.walk(DOCUMENTS_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Extract metadata from path
            # Expected format: documents/{company}/{year}/{quarter}/{file.pdf}
            # OR: documents/{company}/finance.xlsx
            rel_path = os.path.relpath(file_path, DOCUMENTS_DIR)
            parts = rel_path.split(os.sep)
            
            # Process Excel files
            if file.lower().endswith(".xlsx"):
                if len(parts) >= 2:  # documents/company/finance.xlsx
                    company = parts[0]
                    try:
                        print(f"--> Found Excel: {file_path}, Company: {company}")
                        await process_excel_document(
                            file_path=file_path,
                            company_ticker=company
                        )
                    except Exception as e:
                        print(f"Failed to process Excel {file_path}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"Skipping Excel file outside expected structure: {rel_path}")
            
            # Process PDF files
            elif file.lower().endswith(".pdf"):
                if len(parts) >= 3:
                    company = parts[0]
                    year = parts[1]
                    quarter = parts[2] # This might depend on depth
                    
                    # Heuristic: Check if quarter is indeed a quarter folder (Q1-Q4)
                    # If the file is directly in year folder, quarter might be None or derived from filename
                    # Given prompt: "documents/Company/Year/Quarter" -> parts[0]=Company, [1]=Year, [2]=Quarter
                    
                    if len(parts) == 4: # documents/Company/Year/Quarter/File.pdf
                        quarter = parts[2]
                    else:
                         # Fallback if structure is slightly different
                        print(f"Skipping metadata extraction for structured path: {rel_path} - verify structure.")
                        quarter = None

                    try:
                        print(f"--> Found: {file_path}, Metadata: {company}, {year}, {quarter}")
                        await process_document(
                            file_path=file_path,
                            company_ticker=company,
                            fiscal_year=year,
                            fiscal_quarter=quarter
                        )
                    except Exception as e:
                        print(f"Failed to process {file_path}: {e}")
                else:
                    print(f"Skipping file outside expected structure: {rel_path}")

if __name__ == "__main__":
    asyncio.run(batch_process())
