import logging
import uuid
import asyncio
import os
from typing import Optional, Dict
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.core.orchestrator import Orchestrator

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Request Models
class GenerateRequest(BaseModel):
    source_path: str
    project_name: Optional[str] = None
    is_github: bool = False
    github_token: Optional[str] = None
    recursive: bool = True

# Job Management (Simple In-Memory for MVP)
# Store structure: {message, status, result}
jobs: Dict[str, Dict] = {}

async def run_documentation_job(job_id: str, request: GenerateRequest):
    """
    Background task wrapper for the Orchestrator.
    """
    jobs[job_id]["status"] = "processing"
    try:
        orchestrator = Orchestrator()
        result = await orchestrator.run(
            source_path=request.source_path,
            output_dir="docs",  # Default output directory
            recursive=request.recursive
        )
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@router.post("/generate")
async def generate_documentation(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Triggers the documentation generation process.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "request": request.dict(exclude={"github_token"}) # Don't store token in plain text
    }
    
    background_tasks.add_task(run_documentation_job, job_id, request)
    
    return {"job_id": job_id, "status": "queued"}

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Retrieves the status of a documentation job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]

@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Retrieves the result content of a completed job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet.")
        
    return job.get("result")

@router.get("/jobs/{job_id}/download")
async def download_documentation(job_id: str):
    """
    Downloads the generated README file.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet.")
    
    result = job.get("result")
    if not result or "output_file" not in result:
        raise HTTPException(status_code=500, detail="Result file path missing.")
        
    file_path = result["output_file"]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File has been deleted or moved.")
        
    return FileResponse(
        path=file_path, 
        filename=os.path.basename(file_path), 
        media_type='text/markdown'
    )
