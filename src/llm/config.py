from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class LLMConfig(BaseModel):
    """Configuration settings for the LLM"""
    model_name: str = Field(default="claude-3-sonnet-20240229")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    api_key_env_var: str = Field(default="CLAUDE_API_KEY")
    timeout: int = Field(default=30)
    retry_attempts: int = Field(default=3)