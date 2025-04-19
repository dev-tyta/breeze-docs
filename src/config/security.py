import logging
from typing import Optional
import google.generativeai # Import the google.generativeai library
from src.config.settings import get_settings # Import the LLMSettings class for configuration

# Configure logging for this module
logger = logging.getLogger(__name__)
# Basic logging setup (can be enhanced in src/utils/logging.py)
# Ensure this doesn't duplicate handlers if src/utils/logging.py is used
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- API Key Validation ---

def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Validates the format and potentially the validity of an API key by
    attempting a low-cost API call.

    Args:
        api_key: The API key string to validate.

    Returns:
        True if the API key is considered valid (passes basic checks and a test API call),
        False otherwise.
    """
    # Basic check: ensure the key is a non-empty string
    if not isinstance(api_key, str) or not api_key.strip():
        logger.warning("API key validation failed: Key is None or empty.")
        return False

    # --- Implement actual validation logic via API call ---
    try:
        google.generativeai.configure(api_key=api_key)
        # models = list(google.generativeai.list_models())

        # If the list_models call completes without raising an exception, the key is valid.
        logger.info("API key validated successfully via list_models API call.")
        # logger.info("Available models:")
        # for model in models:
        #     logger.info(f"- {model.name}")
        return True

    except Exception as e:
        # Catch any exception that occurs during the API call (e.g., authentication errors, network issues)
        logger.warning(f"API key validation failed: Test API call to list_models failed - {e}")
        return False
    # --- End actual validation logic ---


# Example usage of the validate_api_key function
# if __name__ == "__main__":
#     settings = get_settings()
#     api_key = settings.llm.bree_llm_api_key.get_secret_value()  # Retrieve the API key from the settings
#     if validate_api_key(api_key):
#         print("API key is valid.")
#     else:
#         print("API key is invalid.")
