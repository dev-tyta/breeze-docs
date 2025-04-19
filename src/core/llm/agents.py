import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, Generic, Awaitable

# Import dependencies
from src.core.llm.client import GeminiClient
from src.core.llm.prompter import Prompter
from src.utils.error_handling import CircuitBreakerOpenError, APIError, CircuitBreaker
from src.config.settings import get_settings
import asyncio
import os
from dotenv import load_dotenv
# Potentially import schemas if agents will work with structured data
# from src.core.schemas.models import AgentInput, AgentOutput # Example


# Define a generic type for agent input and output if needed, or use Any/Dict
# InputType = TypeVar('InputType')
# OutputType = TypeVar('OutputType')

# Configure logging for this module
logger = logging.getLogger(__name__)
# Basic logging setup (can be enhanced in src/utils/logging.py)
# Ensure this doesn't duplicate handlers if src/utils/logging.py is used
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the documentation generation system.

    Defines the common interface and dependencies for agents that interact
    with the LLM client and prompter.
    """

    def __init__(self, llm_client: GeminiClient, prompter: Prompter):
        """
        Initializes the BaseAgent with required dependencies.

        Args:
            llm_client: An instance of the LLM client (e.g., GeminiClient).
            prompter: An instance of the Prompter.
        """
        if not isinstance(llm_client, GeminiClient):
             logger.error(f"Invalid LLM client type: {type(llm_client)}")
             raise TypeError("llm_client must be an instance of GeminiClient")
        if not isinstance(prompter, Prompter):
             logger.error(f"Invalid Prompter type: {type(prompter)}")
             raise TypeError("prompter must be an instance of Prompter")

        self._llm_client = llm_client
        self._prompter = prompter
        logger.info(f"BaseAgent initialized with LLMClient and Prompter.")

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method to execute the agent's task.

        This method must be implemented by all concrete agent classes.
        It defines the specific logic for how the agent uses the LLM,
        prompter, and potentially other tools to achieve its goal.

        Args:
            *args: Positional arguments for the agent's task.
            **kwargs: Keyword arguments for the agent's task.

        Returns:
            The result of the agent's task. The type of the result
            will depend on the specific agent implementation.
        """
        pass # Abstract methods do not have an implementation

    # Common methods or properties could be added here, e.g.,
    # - Access to settings (could be passed via client or prompter, or directly)
    # - Common error handling logic specific to agents
    # - Methods for interacting with tools (if tools are managed by agents)

    # Example of a potential common method (requires tool registry or tool access)
    def _use_tool(self, tool_name: str, tool_input: Any) -> Any:
        """Helper method to dispatch to a registered tool."""
        # Logic to look up and execute a tool
        pass


# Example of a concrete agent implementation (inheriting from BaseAgent)
# # This would typically go in a separate file like src/core/agents/documentation.py
# class DocumentationGenerationAgent(BaseAgent):
#     """Agent responsible for generating documentation for code elements."""

#     async def run(self, code_snippet: str, file_path: str, language: str) -> str:
#         """
#         Generates documentation for a given code snippet using the LLM.
#         """
#         logger.info(f"Running DocumentationGenerationAgent for {file_path}")

#         # 1. Use the prompter to build the prompt
#         prompt = self._prompter.for_code_element(
#             code_snippet=code_snippet,
#             file_path=file_path,
#             language=language
#         )
#         logger.debug(f"Generated prompt for documentation:\n{prompt[:200]}...")

#         # 2. Use the LLM client to get the response
#         try:
#             # Assuming generate returns the text content directly or has a .content attribute
#             response = await self._llm_client.generate(prompt)
#             # If generate returns an object, access the text content
#             documentation_text = getattr(response, 'content', str(response))
#             logger.debug("Received response from LLM client.")
#             return documentation_text
#         except CircuitBreakerOpenError:
#             logger.error("LLM client circuit breaker is open. Cannot generate documentation.")
#             raise # Re-raise the specific error
#         except APIError as e:
#             logger.error(f"API error during documentation generation: {e}")
#             raise # Re-raise the specific error
#         except Exception as e:
#             logger.error(f"An unexpected error occurred during documentation generation: {e}", exc_info=True)
#             raise # Re-raise other exceptions


# Usage example (this would typically be in a different module)

# async def run_documentation_agent():
    
#     try:
#         app_settings = get_settings()
#         llm_settings = app_settings.llm
#     except Exception as e:
#         logger.error(f"Failed to load settings: {e}", exc_info=True)

#     # Initialize Circuit Breaker using settings
#     try:
#         llm_circuit_breaker = CircuitBreaker(
#             failure_threshold=llm_settings.cb_failure_threshold,
#             recovery_timeout=llm_settings.cb_recovery_timeout_seconds,
#             expected_successes=llm_settings.cb_expected_successes
#         )
#     except ValueError as e:
#          logger.error(f"Failed to initialize Circuit Breaker with invalid parameters: {e}", exc_info=True)
#          # exit(1) # Exit if CB config is invalid
#     except Exception as e:
#          logger.error(f"An unexpected error occurred initializing Circuit Breaker: {e}", exc_info=True)
#          # exit(1)
    
#     llm_client = GeminiClient(settings=llm_settings, circuit_breaker=llm_circuit_breaker)
#     prompter = Prompter()
    
#     # Create an instance of the agent
#     agent = DocumentationGenerationAgent(llm_client=llm_client, prompter=prompter)
    
#     # Example code file to document
#     code_file = "/home/testys/Documents/GitHub/breeze_docs/data/samples/sample_parsers/hello.js"
    
#     try:
#         with open(code_file, "r") as file:
#             code_snippet = file.read()
            
#         # Run the agent to generate documentation
#         documentation = await agent.run(code_snippet, code_file, "javascript")
#         print("\n===== Generated Documentation =====")
#         print(documentation)
        
#     except FileNotFoundError:
#         print(f"Error: File {code_file} not found")
#     except Exception as e:
#         print(f"Error generating documentation: {str(e)}")

# if __name__ == "__main__":
#     asyncio.run(run_documentation_agent())