import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, Generic, Awaitable

# Configure logging for this module
logger = logging.getLogger(__name__)
# Basic logging setup (can be enhanced in src/utils/logging.py)
# Ensure this doesn't duplicate handlers if src/utils/logging.py is used
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define a generic type for tool input and output if needed, or use Any/Dict
# ToolInputType = TypeVar('ToolInputType')
# ToolOutputType = TypeVar('ToolOutputType')

class BaseTool(ABC):
    """
    Abstract base class for all tools used by agents in the documentation
    generation system.

    Defines the common interface for tools that interact with the environment
    (file system, network, etc.) or perform specific data processing tasks.
    """

    def __init__(self, name: str, description: str):
        """
        Initializes the BaseTool with a name and description.

        Args:
            name: The unique name of the tool.
            description: A brief description of what the tool does.
        """
        if not name or not description:
            logger.error("Tool name and description cannot be empty.")
            raise ValueError("Tool name and description are required.")

        self._name = name
        self._description = description
        logger.info(f"BaseTool initialized: {self._name}")

    @property
    def name(self) -> str:
        """Returns the name of the tool."""
        return self._name

    @property
    def description(self) -> str:
        """Returns the description of the tool."""
        return self._description

    @abstractmethod
    async def run(self, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method to execute the tool's specific logic.

        This method must be implemented by all concrete tool classes.
        It defines how the tool processes input and produces an output.

        Args:
            tool_input: The primary input for the tool. The type will depend
                        on the specific tool implementation.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The result of the tool's execution. The type will depend on the
            specific tool implementation.
        """
        pass # Abstract methods do not have an implementation

    # Common methods or properties could be added here if needed, e.g.,
    # - Access to settings (can be accessed via get_settings() or passed in init)
    # - Common validation logic for tool inputs
