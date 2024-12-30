class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass

class ConfigurationError(LLMError):
    """Raised when there's a configuration issue"""
    pass

class APIError(LLMError):
    """Raised when there's an API-related error"""
    pass

class ParseError(LLMError):
    """Raised when there's an output parsing error"""
    pass