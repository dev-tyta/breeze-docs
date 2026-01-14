import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.core.tools.github_publisher import GithubPublisherTool
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Request Model
class PublishRequest(BaseModel):
    repo_name: str = Field(..., description="Target repository name (owner/repo)")
    file_path: str = Field(..., description="Path to the file to create/update (e.g., README.md)")
    content: str = Field(..., description="Content to write to the file")
    branch: str = Field("main", description="Target branch")
    commit_message: Optional[str] = None
    access_token: Optional[str] = Field(None, description="GitHub Personal Access Token. If not provided, tries to use server-configured token.")

@router.post("/publish")
async def publish_to_github(request: PublishRequest):
    """
    Publishes (creates or updates) a file in a GitHub repository.
    """
    settings = get_settings()
    
    # Determine which token to use
    token = request.access_token
    if not token and settings.github_access_token:
        token = settings.github_access_token.get_secret_value()
        
    if not token:
        raise HTTPException(
            status_code=400, 
            detail="No access token provided. Please provide one in the request or configure the server."
        )

    tool = GithubPublisherTool()
    
    tool_input = {
        "repo_name": request.repo_name,
        "file_path": request.file_path,
        "content": request.content,
        "branch": request.branch,
        "access_token": token,
        "commit_message": request.commit_message
    }
    
    try:
        # Run the tool
        # Note: The tool is async, so we await it.
        result = await tool.run(tool_input)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Publishing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Publishing failed: {str(e)}")
