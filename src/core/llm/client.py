import os
import logging
from typing import AsyncGenerator, Optional, Any, Dict
from pydantic import BaseModel, Field, SecretStr # Import SecretStr
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_google_genai import GoogleGenerativeAI

# Import components from our config and utils modules
from src.config.settings import LLMSettings, get_settings # Import get_settings
from src.config.security import validate_api_key # Import validate_api_key
from src.utils.error_handling import retry_with_backoff, CircuitBreaker, CircuitBreakerOpenError, APIError, ConfigurationError # Import CircuitBreakerOpenError
from src.utils.sanitization import sanitize_input # Import sanitize_input


# Configure logging for this module
logger = logging.getLogger(__name__)
# Basic logging setup (can be enhanced in src/utils/logging.py)
# Ensure this doesn't duplicate handlers if src/utils/logging.py is used
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Define the Gemini Client class, now accepting settings and circuit breaker instances
class GeminiClient(BaseModel):
    """
    Production-grade Gemini client with configuration, safety controls,
    retry logic, and circuit breaker protection.
    """
    # Define fields that Pydantic should validate and assign
    # Renamed from _settings to settings to comply with Pydantic rules
    settings: LLMSettings = Field(..., exclude=True) # Store the LLM settings instance, exclude from model export
    circuit_breaker: CircuitBreaker = Field(...) # Store the Circuit Breaker instance

    # LangChain GoogleGenerativeAI client instance
    client: Optional[GoogleGenerativeAI] = Field(None, exclude=True) # Exclude client instance from model export

    # Pydantic configuration
    class Config:
        arbitrary_types_allowed = True # Allow non-Pydantic types like GoogleGenerativeAI client
        # allow_population_by_field_name = True # Not strictly needed here, but useful if field names differ from init args

    # Constructor now accepts explicit dependencies and passes them to super().__init__
    def __init__(self, settings: LLMSettings, circuit_breaker: CircuitBreaker, **data):
        """
        Initializes the GeminiClient with provided settings and circuit breaker.
        """
        # Pass the explicitly provided settings and circuit_breaker, along with any other data,
        # to Pydantic's BaseModel constructor for validation and assignment.
        # Pydantic will assign the 'settings' argument to the 'settings' field,
        # and the 'circuit_breaker' argument to the 'circuit_breaker' field.
        super().__init__(settings=settings, circuit_breaker=circuit_breaker, **data)

        # Pydantic's super().__init__ has already validated and assigned
        # settings to self.settings and circuit_breaker to self.circuit_breaker.
        # We can now proceed with client-specific initialization.
        logger.info("GeminiClient initializing...")
        self._initialize_client()
        logger.info("GeminiClient initialized.")

    def _initialize_client(self) -> None:
        """
        Validates the API key and configures the internal GoogleGenerativeAI client.
        """
        # Access settings via the corrected field name self.settings
        api_key = self.settings.api_key.get_secret_value()

        # Validate API key using the utility function
        # Note: The validate_api_key in security.py does a test call.
        # This might be redundant if settings loading already requires a valid key,
        # but it adds an extra layer here if the key could change dynamically.
        # Consider if this check is needed every time the client is initialized.
        if not validate_api_key(api_key):
             logger.error("Invalid Gemini API key provided.")
             raise ValueError("Invalid Gemini API key")

        logger.info(f"Initializing GoogleGenerativeAI client with model: {self.settings.model_name}")
        try:
            self.client = GoogleGenerativeAI(
                model=self.settings.model_name,
                api_key=api_key,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                    # Add other safety settings as needed
                },
                timeout=self.settings.timeout # Use timeout from settings
            )
            logger.info("GoogleGenerativeAI client configured successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize GoogleGenerativeAI client: {e}", exc_info=True)
            # Depending on severity, you might want to raise a more specific exception or handle differently
            raise ConfigurationError(f"Failed to initialize LLM client: {e}") from e # Wrap in custom exception


    # Apply retry decorator, handle circuit breaker manually inside
    @retry_with_backoff(max_retries=get_settings().llm.retry_attempts) # Use retry_attempts from settings
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None, # Make optional to use model default if None
        max_tokens: Optional[int] = None # Make optional to use model default if None
    ) -> str:
        """
        Generates text using the LLM with sanitization, retry, and circuit breaker.

        Args:
            prompt: The input prompt string.
            temperature: Controls randomness (0.0 to 1.0). Uses settings default if None.
            max_tokens: Maximum number of output tokens. Uses settings default if None.

        Returns:
            The generated text response.

        Raises:
            CircuitBreakerOpenError: If the circuit breaker is open.
            APIError: For errors during the API call.
            Exception: For other unexpected errors after retries are exhausted.
        """
        if self.client is None:
            logger.error("Attempted to call generate before client was initialized.")
            raise RuntimeError("LLM client is not initialized") # Or ConfigurationError

        sanitized_prompt = sanitize_input(prompt)
        logger.debug(f"Generating text for sanitized prompt (first 50 chars): {sanitized_prompt[:50]}...")

        # Use provided temperature/max_tokens if not None, otherwise use settings defaults
        # Access settings via the corrected field name self.settings
        effective_temperature = temperature if temperature is not None else self.settings.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self.settings.max_tokens

        try:
            # Check circuit breaker state BEFORE the API call starts
            self.circuit_breaker._check_state() # This raises CircuitBreakerOpenError if OPEN

            # If not OPEN, proceed with the API call
            logger.debug("Making generate API call.")
            response = await self.client.ainvoke(
                sanitized_prompt,
                temperature=effective_temperature,
                max_output_tokens=effective_max_tokens
            )
            # If the call succeeds, record success with the circuit breaker
            self.circuit_breaker.record_success()
            logger.debug("Text generation API call successful.")
            return response

        except CircuitBreakerOpenError:
             # Re-raise CircuitBreakerOpenError as it's handled by the caller
             logger.debug("Generate call blocked by Circuit Breaker (OPEN).")
             raise
        except Exception as e:
            # If any other exception occurs during the API call
            self.circuit_breaker.record_failure() # Record failure with the circuit breaker
            logger.error(f"Text generation API call failed: {e}", exc_info=True)
            raise APIError(f"LLM API generate call failed: {e}") from e # Wrap in custom exception


    # Apply retry decorator, handle circuit breaker manually inside
    @retry_with_backoff(max_retries=get_settings().llm.retry_attempts) # Use retry_attempts from settings
    async def stream(
        self,
        prompt: str,
        temperature: Optional[float] = None # Make optional to use model default if None
    ) -> AsyncGenerator[str, None]:
        """
        Streams response tokens from the LLM with sanitization, retry, and circuit breaker.

        Args:
            prompt: The input prompt string.
            temperature: Controls randomness (0.0 to 1.0). Uses settings default if None.

        Yields:
            Chunks of the generated text response.

        Raises:
            CircuitBreakerOpenError: If the circuit breaker is open.
            APIError: For errors during the API call.
            Exception: For other unexpected errors after retries are exhausted.
        """
        if self.client is None:
            logger.error("Attempted to call stream before client was initialized.")
            raise RuntimeError("LLM client is not initialized") # Or ConfigurationError

        sanitized_prompt = sanitize_input(prompt)
        logger.debug(f"Streaming text for sanitized prompt (first 50 chars): {sanitized_prompt[:50]}...")

        # Use provided temperature if not None, otherwise use settings default
        # Access settings via the corrected field name self.settings
        effective_temperature = temperature if temperature is not None else self.settings.temperature

        try:
            # Check circuit breaker state BEFORE the API call starts
            self.circuit_breaker._check_state() # This raises CircuitBreakerOpenError if OPEN

            # If not OPEN, proceed with the API call
            logger.debug("Starting text streaming API call.")
            async for chunk in self.client.astream( # Keep stream_async for this client class
                sanitized_prompt,
                temperature=effective_temperature
            ):
                yield chunk

            # If the loop completes without exception, record success
            self.circuit_breaker.record_success()
            logger.debug("Text streaming API call successful.")

        except CircuitBreakerOpenError:
             # Re-raise CircuitBreakerOpenError
             logger.debug("Streaming blocked by Circuit Breaker (OPEN).")
             raise
        except Exception as e:
            # If any other exception occurs during the API call or streaming
            self.circuit_breaker.record_failure() # Record failure with the circuit breaker
            logger.error(f"Text streaming API call failed: {e}", exc_info=True)
            raise APIError(f"LLM API stream call failed: {e}") from e # Wrap in custom exception


    def rotate_api_key(self, new_key: str) -> None:
        """
        Safely rotates the API credentials and re-initializes the client.

        Args:
            new_key: The new API key string.

        Raises:
            ValueError: If the new key is invalid according to validate_api_key.
            ConfigurationError: If re-initialization fails with the new key.
        """
        logger.info("Attempting to rotate API key.")
        # Validate the new API key using the utility function
        if not validate_api_key(new_key):
            logger.warning("New API key failed validation.")
            raise ValueError("Invalid API key format or validation failed")

        # Update the API key in the settings instance
        # Note: This updates the settings instance held by THIS client instance.
        # If get_settings() was called elsewhere, it might hold an older version
        # unless a mechanism to reload/update settings globally is implemented.
        # Access settings via the corrected field name self.settings
        self.settings.api_key = SecretStr(new_key)
        logger.info("API key updated in settings. Re-initializing client.")

        # Re-initialize the client with the new key
        # _initialize_client already includes error handling and raises ConfigurationError
        self._initialize_client()
        logger.info("API key rotation complete. Client re-initialized.")

# Example of how you might instantiate the client (likely in main.py or an LLM service factory)
if __name__ == "__main__":
    # Ensure logging is set up if running this file directly
    if not logging.getLogger().handlers:
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Load settings
    try:
        app_settings = get_settings()
        llm_settings = app_settings.llm
    except Exception as e:
        logger.error(f"Failed to load settings: {e}", exc_info=True)
        # Handle fatal configuration error, perhaps exit
        # exit(1) # Exit if settings cannot be loaded

    # Initialize Circuit Breaker using settings
    try:
        llm_circuit_breaker = CircuitBreaker(
            failure_threshold=llm_settings.cb_failure_threshold,
            recovery_timeout=llm_settings.cb_recovery_timeout_seconds,
            expected_successes=llm_settings.cb_expected_successes
        )
    except ValueError as e:
         logger.error(f"Failed to initialize Circuit Breaker with invalid parameters: {e}", exc_info=True)
         # exit(1) # Exit if CB config is invalid
    except Exception as e:
         logger.error(f"An unexpected error occurred initializing Circuit Breaker: {e}", exc_info=True)
         # exit(1)


    # Instantiate the client, passing in settings and circuit breaker
    try:
        # Pass the loaded llm_settings and the initialized circuit_breaker instance
        gemini_client = GeminiClient(settings=llm_settings, circuit_breaker=llm_circuit_breaker)
        logger.info("GeminiClient instantiated successfully.")

        # Example usage (requires a valid API key set in environment or .env)
        async def run_example():
            print("\n--- Running LLM Client Example ---")
            try:
                print("\nTesting generate...")
                response = await gemini_client.generate("Tell me a short story about a brave knight.")
                print("Generated Response:")
                print(response)

                print("\nTesting stream...")
                print("Streaming Response:")
                async for chunk in gemini_client.stream("Tell me another short story."):
                    print(chunk, end="")
                print("\nStreaming complete.")

            except CircuitBreakerOpenError:
                logger.error("\nCircuit breaker is open. Cannot make LLM calls.")
            except APIError as e:
                logger.error(f"\nAn API error occurred during LLM call: {e}")
            except Exception as e:
                logger.error(f"\nAn unexpected error occurred during LLM call: {e}")

        import asyncio
        asyncio.run(run_example())

    except ValueError as e:
        logger.error(f"Failed to instantiate GeminiClient due to invalid API key: {e}", exc_info=True)
        # exit(1) # Exit if client instantiation fails due to invalid key
    except ConfigurationError as e:
        logger.error(f"Failed to instantiate GeminiClient due to configuration error: {e}", exc_info=True)
        # exit(1) # Exit if client instantiation fails due to config error
    except Exception as e:
        logger.error(f"An unexpected error occurred during GeminiClient instantiation: {e}", exc_info=True)
        # exit(1)
