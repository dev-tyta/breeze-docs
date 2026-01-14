import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.core.llm.client import GeminiClient
from src.core.llm.prompter import Prompter

# Configure logging for this module
logger = logging.getLogger(__name__)

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
        pass
