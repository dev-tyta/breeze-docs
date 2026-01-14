import os
import logging # Import logging
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional # Import Optional if needed

# Configure logging for this module
logger = logging.getLogger(__name__)
# Basic logging setup (can be enhanced in src/utils/logging.py)
# Ensure this doesn't duplicate handlers if src/utils/logging.py is used
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Define the settings specific to the Language Model
# This class consolidates configuration details for the LLM client and its usage parameters.
class LLMSettings(BaseSettings):
    """Settings for Language Model configuration."""

    # API key for the LLM service. Using SecretStr for security.
    # Loaded from the environment variable BREE_LLM_API_KEY (due to env_prefix).
    api_key: SecretStr = Field(
        ..., # Ellipsis indicates this field is required
        # Removed env="LLM_API_KEY" to rely on env_prefix="BREE_"
    )

    # Name of the LLM model to use (e.g., "gemini-pro", "gemini-2.0-flash")
    # Loaded from the environment variable BREE_LLM_MODEL_NAME (due to env_prefix).
    model_name: str = Field(
        "gemini-2.0-flash", # Default model name
        # Removed env="LLM_MODEL_NAME" to rely on env_prefix="BREE_"
    )

    # Maximum number of tokens for the LLM response.
    # Loaded from the environment variable BREE_LLM_MAX_TOKENS.
    max_tokens: int = Field(
        8192, # Default max tokens
        # Removed env="LLM_MAX_TOKENS" to rely on env_prefix="BREE_"
    )

    # Maximum number of iterations for agentic workflows (if applicable).
    # Loaded from the environment variable BREE_LLM_MAX_ITERATIONS.
    max_iterations: int = Field(
        9, # Default max iterations
        # Removed env="LLM_MAX_ITERATIONS" to rely on env_prefix="BREE_"
    )

    # Temperature setting for controlling response randomness.
    # Loaded from the environment variable BREE_LLM_TEMPERATURE.
    temperature: float = Field(
        0.7, # Default temperature
        # Removed env="LLM_TEMPERATURE" to rely on env_prefix="BREE_"
    )

    # Timeout for LLM requests in seconds.
    # Loaded from the environment variable BREE_LLM_TIMEOUT_SECONDS.
    timeout: int = Field(
        30, # Default timeout
        # Removed env="LLM_TIMEOUT_SECONDS" to rely on env_prefix="BREE_"
    )

    # Number of retry attempts for failed LLM calls.
    # Loaded from the environment variable BREE_LLM_RETRY_ATTEMPTS.
    retry_attempts: int = Field(
        3, # Default retry attempts
        # Removed env="LLM_RETRY_ATTEMPTS" to rely on env_prefix="BREE_"
    )

    # --- GitHub Authentication Settings ---
    github_access_token: Optional[SecretStr] = Field(
        None, # Optional GitHub access token
        # env="BREE_LLM_GITHUB_API_KEY
    )

    # --- Circuit Breaker Settings ---
    # Number of consecutive failures before the circuit opens.
    # Loaded from BREE_LLM_CB_FAILURE_THRESHOLD.
    cb_failure_threshold: int = Field(
        5, # Default failure threshold
        env="BREE_LLM_CB_FAILURE_THRESHOLD"
    )

    # Time in seconds the circuit stays open before transitioning to HALF-OPEN.
    # Loaded from BREE_LLM_CB_RECOVERY_TIMEOUT_SECONDS.
    cb_recovery_timeout_seconds: int = Field(
        30, # Default recovery timeout
        env="BREE_LLM_CB_RECOVERY_TIMEOUT_SECONDS"
    )

    # Number of successful calls required in HALF-OPEN state to transition back to CLOSED.
    # Loaded from BREE_LLM_CB_EXPECTED_SUCCESSES.
    cb_expected_successes: int = Field(
        2, # Default expected successes
        env="BREE_LLM_CB_EXPECTED_SUCCESSES"
    )
    # --- End Circuit Breaker Settings ---


    # Configuration for Pydantic Settings for LLMSettings
    # Specifies how settings are loaded (e.g., from .env file, environment variables)
    model_config = SettingsConfigDict(
        env_file=".env", # Optional: Load settings from a .env file in the project root
        env_file_encoding="utf-8",
        extra="ignore", # Ignore extra environment variables not defined in the model
        env_prefix="BREE_LLM_", # Apply prefix "BREE_LLM_" to environment variables for these settings
        # Example env vars: BREE_LLM_API_KEY, BREE_LLM_MODEL_NAME, etc.
    )

# Define the main application settings
# This class holds all different configuration sections of the application.
class AppSettings(BaseSettings):
    """Main application settings."""

    # Environment the application is running in (e.g., "development", "production", "testing")
    # Loaded from the environment variable BREE_APP_ENV (due to env_prefix).
    app_env: str = Field(
        "development", # Default environment
        # Removed env="APP_ENV" to rely on env_prefix="BREE_"
    )

    # Nested LLM settings using the LLMSettings model defined above.
    # This will automatically load LLMSettings based on its own configuration.
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # GitHub Configuration
    github_access_token: Optional[SecretStr] = Field(None, env="GITHUB_PERSONAL_ACCESS_TOKEN")
    
    # OAuth App Configuration (Optional, for "Login with GitHub")
    github_client_id: Optional[str] = Field(None, env="GITHUB_CLIENT_ID")
    github_client_secret: Optional[SecretStr] = Field(None, env="GITHUB_CLIENT_SECRET")
    
    # Configuration for Pydantic Settings for the main AppSettings
    model_config = SettingsConfigDict(
        env_file=".env", # Optional: Load settings from a .env file
        env_file_encoding="utf-8",
        extra="ignore", # Ignore extra environment variables not defined in the model
        env_prefix="BREE_", # Apply prefix "BREE_" to environment variables for these settings
        # Example env vars: BREE_APP_ENV
        # Note: The prefix for nested models (like LLMSettings) is handled by the nested model's config.
    )


# Use lru_cache to create a singleton instance of settings
# This ensures settings are loaded only once during the application's lifetime.
@lru_cache()
def get_settings() -> AppSettings:
    """Gets the application settings instance (cached)."""
    # This will trigger the loading of settings from environment variables and .env file
    logger.info("Loading application settings.")
    return AppSettings()

# Example usage (optional, for testing purposes)
# This block will only run if the script is executed directly
if __name__ == "__main__":
    # Ensure logging is set up if running this file directly
    if not logging.getLogger().handlers:
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # To test, create a .env file in the same directory or set environment variables
    # Example .env content:
    # BREE_LLM_API_KEY="your_gemini_api_key_here"
    # BREE_LLM_MODEL_NAME="gemini-1.5-pro-latest"
    # BREE_LLM_TIMEOUT_SECONDS=120
    # BREE_APP_ENV="production"
    # BREE_LLM_CB_FAILURE_THRESHOLD=3
    # BREE_LLM_CB_RECOVERY_TIMEOUT_SECONDS=60
    # BREE_LLM_CB_EXPECTED_SUCCESSES=1

    print("Loading settings...")
    try:
        settings = get_settings()
        print("\n--- App Settings ---")
        print(f"App Environment: {settings.app_env}")

        print("\n--- LLM Settings ---")
        # Accessing secrets requires .get_secret_value()
        # print(f"LLM API Key: {settings.llm.api_key.get_secret_value()}") # Be cautious printing secrets!
        print(f"LLM Model Name: {settings.llm.model_name}")
        print(f"LLM Max Tokens: {settings.llm.max_tokens}")
        print(f"LLM Max Iterations: {settings.llm.max_iterations}")
        print(f"LLM Temperature: {settings.llm.temperature}")
        print(f"LLM Timeout: {settings.llm.timeout} seconds")
        print(f"LLM Retry Attempts: {settings.llm.retry_attempts}")
        print(f"Circuit Breaker Failure Threshold: {settings.llm.cb_failure_threshold}")
        print(f"Circuit Breaker Recovery Timeout: {settings.llm.cb_recovery_timeout_seconds} seconds")
        print(f"Circuit Breaker Expected Successes: {settings.llm.cb_expected_successes}")

    except Exception as e:
        logger.error(f"Failed to load settings: {e}", exc_info=True)
        print(f"Error loading settings: {e}") # Also print to console for immediate feedback
        # Handle fatal configuration error, perhaps exit
        # exit(1) # Don't exit in example block, just show the error
