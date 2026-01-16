
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
import os
import shutil
from document_processor import process_document

app = FastAPI(title="FiscalFlow API")

DOCUMENTS_DIR = "documents"

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    company: str = Form(...),
    year: str = Form(...),
    quarter: str = Form(...),
    file: UploadFile = File(...)
):
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
    return {"message": "Welcome to FiscalFlow API"}
