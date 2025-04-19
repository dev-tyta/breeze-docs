import asyncio
import time
import random
import logging
from functools import wraps
from typing import Callable, Any, Awaitable

# Configure logging for this module
logger = logging.getLogger(__name__)
# Basic logging setup (can be enhanced in src/utils/logging.py)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Retry with Exponential Backoff Decorator ---

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """
    A decorator to retry an async function with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of times to retry the function execution.
        base_delay: The base delay in seconds before the first retry.
        max_delay: The maximum delay in seconds between retries.
    """
    def decorator(func: Callable[..., Awaitable[Any]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    # Attempt to call the decorated function
                    return await func(*args, **kwargs)
                except Exception as e:
                    # If it's the last attempt, re-raise the exception
                    if attempt == max_retries:
                        logger.error(f"Attempt {attempt + 1}/{max_retries + 1} failed. Max retries reached. Raising exception: {e}")
                        raise
                    # Calculate delay with exponential backoff and jitter
                    # delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay) # Basic jitter
                    delay = min(max_delay, base_delay * (2 ** attempt)) # Exponential backoff, capped by max_delay
                    jitter = random.uniform(0, delay * 0.1) # Add 10% jitter
                    sleep_time = delay + jitter

                    logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed. Retrying in {sleep_time:.2f} seconds...")
                    # Wait before the next attempt
                    await asyncio.sleep(sleep_time)

        return wrapper
    return decorator


# --- Circuit Breaker Implementation ---

class CircuitBreaker:
    """
    Implements the Circuit Breaker pattern to prevent repeated calls to a
    failing service.

    States:
        CLOSED: Normal operation. Calls are allowed.
        OPEN: Service is failing. Calls are blocked (fail fast).
        HALF-OPEN: Service might be recovering. A limited number of calls are allowed.
    """
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30, expected_successes: int = 2):
        """
        Initializes the CircuitBreaker.

        Args:
            failure_threshold: Number of consecutive failures before the circuit opens.
            recovery_timeout: Time in seconds the circuit stays open before transitioning to HALF-OPEN.
            expected_successes: Number of successful calls required in HALF-OPEN state
                                to transition back to CLOSED.
        """
        if failure_threshold <= 0 or recovery_timeout <= 0 or expected_successes <= 0:
             raise ValueError("failure_threshold, recovery_timeout, and expected_successes must be positive.")

        self._state = "CLOSED"
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._expected_successes = expected_successes

        self._consecutive_failures = 0
        self._last_failure_time = 0
        self._successful_attempts_in_half_open = 0

        logger.info(f"CircuitBreaker initialized in state: {self._state}")

    @property
    def state(self) -> str:
        """Returns the current state of the circuit breaker."""
        return self._state

    def _check_state(self) -> None:
        """Checks and potentially transitions the circuit breaker state."""
        if self._state == "OPEN":
            # If in OPEN state, check if the recovery timeout has passed
            if time.time() - self._last_failure_time > self._recovery_timeout:
                # If timeout passed, transition to HALF-OPEN
                self._state = "HALF-OPEN"
                self._successful_attempts_in_half_open = 0
                logger.warning("Circuit Breaker transitioned to HALF-OPEN state.")
            else:
                # If timeout not passed, remain OPEN
                logger.warning("Circuit Breaker is OPEN. Failing fast.")
                raise CircuitBreakerOpenError("Circuit breaker is open") # Raise a specific error

        elif self._state == "HALF-OPEN":
             # In HALF-OPEN, calls are allowed but state is checked after the call
             pass # State transition happens in record_success/record_failure

        # In CLOSED state, calls are always allowed initially
        # State transition to OPEN happens in record_failure

    def record_failure(self) -> None:
        """Records a failure and updates the circuit breaker state."""
        if self._state == "CLOSED":
            self._consecutive_failures += 1
            logger.warning(f"Failure recorded in CLOSED state. Consecutive failures: {self._consecutive_failures}")
            # If failures exceed threshold, open the circuit
            if self._consecutive_failures >= self._failure_threshold:
                self._state = "OPEN"
                self._last_failure_time = time.time()
                logger.error(f"Failure threshold ({self._failure_threshold}) reached. Circuit Breaker transitioned to OPEN state.")

        elif self._state == "HALF-OPEN":
            # If a failure occurs in HALF-OPEN, immediately transition back to OPEN
            self._state = "OPEN"
            self._consecutive_failures = 1 # Reset consecutive failures for the new OPEN state
            self._last_failure_time = time.time()
            logger.error("Failure recorded in HALF-OPEN state. Circuit Breaker transitioned back to OPEN state.")

        # If in OPEN state, record_failure doesn't change state, it's already open.
        # We could update last_failure_time, but it's not strictly necessary for the basic logic.


    def record_success(self) -> None:
        """Records a success and updates the circuit breaker state."""
        if self._state == "CLOSED":
            # Reset consecutive failures on success in CLOSED state
            self._consecutive_failures = 0
            logger.info("Success recorded in CLOSED state. Consecutive failures reset.")

        elif self._state == "HALF-OPEN":
            self._successful_attempts_in_half_open += 1
            logger.warning(f"Success recorded in HALF-OPEN state. Successful attempts: {self._successful_attempts_in_half_open}/{self._expected_successes}")
            # If enough successes in HALF-OPEN, close the circuit
            if self._successful_attempts_in_half_open >= self._expected_successes:
                self._state = "CLOSED"
                self._consecutive_failures = 0 # Reset failures
                self._successful_attempts_in_half_open = 0 # Reset successes
                logger.warning("Required successes in HALF-OPEN reached. Circuit Breaker transitioned to CLOSED state.")

        # If in OPEN state, success doesn't change state.

    def protect(self, func: Callable[..., Awaitable[Any]]):
        """
        Decorator to protect an async function with the circuit breaker.
        Checks the circuit state before executing the function.
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check the circuit breaker state before attempting the call
            self._check_state() # This will raise CircuitBreakerOpenError if OPEN

            try:
                # If not OPEN, execute the protected function
                result = await func(*args, **kwargs)
                # If successful, record success
                self.record_success()
                return result
            except Exception as e:
                # If an exception occurs during execution, record failure
                self.record_failure()
                # Re-raise the exception
                raise

        return wrapper

# Define a custom exception for when the circuit breaker is open
class CircuitBreakerOpenError(Exception):
    """Custom exception raised when the circuit breaker is open."""
    pass

# --- Custom Application Exceptions ---
# These exceptions provide more specific error types for different failure scenarios.

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

# Example Usage (for testing purposes) - Requires an async function to protect
# async def unreliable_service():
#     """Simulates a service that sometimes fails."""
#     if random.random() < 0.6: # 60% chance of failure
#         raise ConnectionError("Simulated network issue")
#     else:
#         print("Service call successful!")
#         return "Success"

# async def main():
#     # Initialize circuit breaker with specific parameters
#     cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10, expected_successes=2)

#     # Protect the unreliable service function
#     protected_service = cb.protect(unreliable_service)

#     print("--- Starting Circuit Breaker Test ---")
#     for i in range(10):
#         print(f"\nAttempt {i+1}:")
#         try:
#             result = await protected_service()
#             print(f"Result: {result}")
#         except CircuitBreakerOpenError:
#             print("Call blocked by Circuit Breaker (OPEN).")
#         except Exception as e:
#             print(f"Call failed with error: {e}")

#         print(f"Current Circuit Breaker state: {cb.state}")
#         await asyncio.sleep(1) # Small delay between attempts

#     print("\n--- Test Complete ---")

# if __name__ == "__main__":
#     # To run the example, uncomment the main() and unreliable_service() functions
#     # and run this script directly.
#     # asyncio.run(main())
#     pass # Keep pass if example is commented out
