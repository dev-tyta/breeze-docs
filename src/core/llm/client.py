import os
import logging
from typing import AsyncGenerator, Optional, Any, Dict
from pydantic import BaseModel, Field, SecretStr
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Import components from our config and utils modules
from src.config.settings import LLMSettings, get_settings
from src.config.security import validate_api_key
from src.utils.error_handling import retry_with_backoff, CircuitBreaker, CircuitBreakerOpenError, APIError, ConfigurationError
from src.utils.sanitization import sanitize_input

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class GeminiClient(BaseModel):
    """
    Production-grade Gemini client with configuration, safety controls,
    retry logic, and circuit breaker protection.
    Uses strict google-generativeai library to avoid LangChain incompatibility issues on Py3.14.
    """
    settings: LLMSettings = Field(..., exclude=True)
    circuit_breaker: CircuitBreaker = Field(...)
    
    # We hold the model object from google.generativeai
    model: Optional[Any] = Field(None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, settings: LLMSettings, circuit_breaker: CircuitBreaker, **data):
        super().__init__(settings=settings, circuit_breaker=circuit_breaker, **data)
        self._initialize_client()

    def _initialize_client(self) -> None:
        api_key = self.settings.api_key.get_secret_value()

        if not validate_api_key(api_key):
             logger.error("Invalid Gemini API key provided.")
             raise ValueError("Invalid Gemini API key")

        try:
            genai.configure(api_key=api_key)
            
            # Function mappings for safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.settings.model_name,
                safety_settings=safety_settings,
                # system_instruction can be passed here if needed, but we usually embed it in prompt
            )

            logger.info(f"Google GenerativeAI model '{self.settings.model_name}' configured successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize GenerativeModel: {e}", exc_info=True)
            raise ConfigurationError(f"Failed to initialize LLM client: {e}") from e

    @retry_with_backoff(max_retries=3) # Hardcoded fallback or use settings
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        if self.model is None:
            raise RuntimeError("LLM client is not initialized")

        sanitized_prompt = sanitize_input(prompt)
        
        effective_temperature = temperature if temperature is not None else self.settings.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self.settings.max_tokens

        generation_config = genai.GenerationConfig(
            temperature=effective_temperature,
            max_output_tokens=effective_max_tokens
        )

        try:
            self.circuit_breaker._check_state()
            
            # The async method for google-generativeai is generate_content_async
            response = await self.model.generate_content_async(
                sanitized_prompt,
                generation_config=generation_config
            )
            
            self.circuit_breaker.record_success()
            return response.text

        except CircuitBreakerOpenError:
             logger.debug("Generate call blocked by Circuit Breaker (OPEN).")
             raise
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Text generation API call failed: {e}", exc_info=True)
            raise APIError(f"LLM API generate call failed: {e}") from e

    @retry_with_backoff(max_retries=3)
    async def stream(
        self,
        prompt: str,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        if self.model is None:
            raise RuntimeError("LLM client is not initialized")

        sanitized_prompt = sanitize_input(prompt)
        effective_temperature = temperature if temperature is not None else self.settings.temperature

        generation_config = genai.GenerationConfig(
            temperature=effective_temperature
        )

        try:
            self.circuit_breaker._check_state()
            
            # stream=True returns a generator (or async generator depending on method)
            response_stream = await self.model.generate_content_async(
                sanitized_prompt,
                generation_config=generation_config,
                stream=True
            )
            
            async for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

            self.circuit_breaker.record_success()

        except CircuitBreakerOpenError:
             raise
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Text streaming API call failed: {e}", exc_info=True)
            raise APIError(f"LLM API stream call failed: {e}") from e

    def rotate_api_key(self, new_key: str) -> None:
        self.settings.api_key = SecretStr(new_key)
        self._initialize_client()


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
