import re
import logging
from typing import Optional, Any

# Configure logging for this module
logger = logging.getLogger(__name__)
# Basic logging setup (can be enhanced in src/utils/logging.py)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def sanitize_input(input_string: Optional[Any]) -> str:
    """
    Sanitizes an input string by cleaning up whitespace and handling non-string types.

    Args:
        input_string: The input to sanitize. Can be a string or None.

    Returns:
        A sanitized string. Returns an empty string if the input is None or not a string.
    """
    # Handle None or non-string inputs
    if input_string is None:
        logger.debug("sanitize_input received None, returning empty string.")
        return ""
    if not isinstance(input_string, str):
        logger.warning(f"sanitize_input received non-string type: {type(input_string)}. Attempting to convert.")
        try:
            input_string = str(input_string)
        except Exception as e:
            logger.error(f"Failed to convert input to string: {e}", exc_info=True)
            return "" # Return empty string if conversion fails

    # Strip leading/trailing whitespace
    sanitized_string = input_string.strip()

    # Replace multiple internal whitespace characters (spaces, tabs, newlines) with a single space
    # This handles cases like multiple spaces, tabs, or line breaks within the text
    sanitized_string = re.sub(r'\s+', ' ', sanitized_string)

    # Add more sanitization steps here if needed, e.g.:
    # - Removing specific characters
    # - Escaping characters (less common for LLM prompts, but depends on context)
    # - Limiting length

    logger.debug(f"Sanitized input (first 50 chars): {sanitized_string[:50]}...")

    return sanitized_string

# # Example Usage (for testing purposes)
# if __name__ == "__main__":
#     test_strings = [
#         "  Hello,   World!  \n",
#         "This\t has\t\ttabs.",
#         "Multiple\nLines\n\nHere.",
#         "   Leading and trailing.   ",
#         None,
#         12345,
#         "",
#         "  "
#     ]

#     for test_str in test_strings:
#         original_type = type(test_str).__name__
#         sanitized = sanitize_input(test_str)
#         print(f"Original ({original_type}): '{test_str}'")
#         print(f"Sanitized: '{sanitized}'\n")

