import logging
import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from src.config.settings import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/login")
async def login_with_github():
    """
    Redirects the user to GitHub's OAuth authorization page.
    """
    settings = get_settings()
    if not settings.github_client_id:
        raise HTTPException(status_code=500, detail="GitHub Client ID not configured.")
    
    # Scope: 'repo' is needed to read/write repositories
    scope = "repo"
    redirect_uri = "http://localhost:8000/api/v1/auth/callback" # Update for production
    
    github_auth_url = (
        f"https://github.com/login/oauth/authorize"
        f" ?client_id={settings.github_client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&scope={scope}"
    )
    return RedirectResponse(url=github_auth_url)

@router.get("/callback")
async def github_callback(code: str):
    """
    Exchanges the authorization code for an access token.
    """
    settings = get_settings()
    if not settings.github_client_id or not settings.github_client_secret:
        raise HTTPException(status_code=500, detail="GitHub Credentials not configured.")
        
    token_url = "https://github.com/login/oauth/access_token"
    payload = {
        "client_id": settings.github_client_id,
        "client_secret": settings.github_client_secret.get_secret_value(),
        "code": code
    }
    headers = {"Accept": "application/json"}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, json=payload, headers=headers)
        
    if response.status_code != 200:
        logger.error(f"GitHub Token Exchange Failed: {response.text}")
        raise HTTPException(status_code=400, detail="Failed to retrieve access token.")
        
    data = response.json()
    access_token = data.get("access_token")
    
    if not access_token:
         logger.error(f"No access token in response: {data}")
         raise HTTPException(status_code=400, detail="Invalid response from GitHub.")
         
    # In a real app, you would:
    # 1. Create a session or JWT for the user.
    # 2. Store the access_token securely associated with that session.
    # 3. Redirect the user to the frontend dashboard.
    
    return {
        "status": "success",
        "access_token": access_token, 
        "start_url": "/api/v1/docs/dashboard" # Hypothetical next step
    }
