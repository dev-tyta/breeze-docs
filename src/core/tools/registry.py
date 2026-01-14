import logging
from typing import Dict, List, Optional, Any
import asyncio

# Import the base tool interface
from src.core.tools.base import BaseTool

# Configure logging for this module
logger = logging.getLogger(__name__)
# The basic logging setup block is removed here.
# Logging should be configured centrally by calling setup_logging() from src/utils/logging.py
# in the application's entry point (e.g., main.py).

class ToolRegistry:
    """
    A registry for managing and providing access to available tools.

    Agents or other components can use this registry to discover and retrieve tools
    needed to perform specific tasks.
    """

    def __init__(self):
        """
        Initializes the ToolRegistry.
        """
        # Internal dictionary to store tool instances, mapping name to instance
        self._tools: Dict[str, BaseTool] = {}
        logger.info("ToolRegistry initialized.")

    def register_tool(self, tool: BaseTool) -> None:
        """
        Registers a tool instance with the registry.

        Args:
            tool: An instance of a class inheriting from BaseTool.

        Raises:
            TypeError: If the provided object is not a BaseTool instance.
            ValueError: If a tool with the same name is already registered.
        """
        if not isinstance(tool, BaseTool):
            logger.error(f"Attempted to register object that is not a BaseTool: {type(tool)}")
            raise TypeError("Only instances of BaseTool can be registered.")

        if tool.name in self._tools:
            logger.warning(f"Tool with name '{tool.name}' is already registered. Overwriting.")
            # Depending on requirements, you might raise an error instead of warning and overwriting
            # raise ValueError(f"Tool with name '{tool.name}' is already registered.")

        self._tools[tool.name] = tool
        logger.info(f"Tool registered: '{tool.name}'")

    def get_tool(self, tool_name: str) -> BaseTool:
        """
        Retrieves a registered tool by its name.

        Args:
            tool_name: The name of the tool to retrieve.

        Returns:
            The registered BaseTool instance.

        Raises:
            ValueError: If no tool with the given name is found.
        """
        try:
            tool = self._tools[tool_name]
            logger.debug(f"Retrieved tool: '{tool_name}'")
            return tool
        except KeyError:
            logger.error(f"Tool with name '{tool_name}' not found in the registry.")
            raise ValueError(f"Tool with name '{tool_name}' not found.")

    def list_tools(self) -> List[Dict[str, str]]:
        """
        Lists the names and descriptions of all registered tools.

        Returns:
            A list of dictionaries, each containing 'name' and 'description'
            for a registered tool.
        """
        tool_list = [{"name": tool.name, "description": tool.description} for tool in self._tools.values()]
        logger.debug(f"Listing {len(tool_list)} registered tools.")
        return tool_list

    # Example of how agents might use the registry (conceptual)
    # In a concrete agent class:
    # self._tool_registry.get_tool("filesystem_crawler").run(directory="/path/to/repo")


# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Ensure logging is set up if running this file directly
    if not logging.getLogger().handlers:
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
         logging.getLogger().setLevel(logging.DEBUG) # Set level to DEBUG for more output

    # Create a dummy tool for testing
    class DummyTool(BaseTool):
        def __init__(self, name: str, description: str):
            super().__init__(name, description)

        async def run(self, tool_input: Any) -> Any:
            logger.info(f"DummyTool '{self.name}' running with input: {tool_input}")
            await asyncio.sleep(0.01) # Simulate async work
            return f"Processed: {tool_input}"

    # Instantiate the registry
    registry = ToolRegistry()

    # Create and register dummy tools
    dummy_crawler = DummyTool("dummy_crawler", "A dummy file crawler tool.")
    dummy_parser = DummyTool("dummy_parser", "A dummy code parser tool.")

    registry.register_tool(dummy_crawler)
    registry.register_tool(dummy_parser)

    print("\n--- Registered Tools ---")
    for tool_info in registry.list_tools():
        print(f"- {tool_info['name']}: {tool_info['description']}")

    print("\n--- Getting and Running a Tool ---")
    try:
        retrieved_tool = registry.get_tool("dummy_crawler")
        # Note: Running async tool requires an async context
        result = asyncio.run(retrieved_tool.run("some_path"))
        print(f"Tool run result: {result}")
        print(f"Successfully retrieved tool: {retrieved_tool.name}")

    except ValueError as e:
        print(f"Error getting tool: {e}")

    print("\n--- Attempting to Get Non-existent Tool ---")
    try:
        registry.get_tool("non_existent_tool")
    except ValueError as e:
        print(f"Correctly caught error: {e}")
