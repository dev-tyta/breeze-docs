import os
import logging
from typing import Optional, Any
from functools import wraps
import time
import asyncio

from src.exceptions import ConfigurationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries: int = 3, initial_wait: float = 1.0):
    """Decorator for retrying operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            wait_time = initial_wait

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        wait_time *= 2
                    else:
                        logger.error(f"All {max_retries} attempts failed")
                        raise last_exception
        return wrapper
    return decorator

def validate_api_key(api_key: Optional[str]) -> str:
    """Validate and return API key"""
    if not api_key:
        raise ConfigurationError("API key not found in environment variables")
    return api_key