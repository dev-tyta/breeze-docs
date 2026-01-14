import logging
from typing import Optional, Any

from src.core.agents.base import BaseAgent
from src.core.llm.client import GeminiClient
from src.core.llm.prompter import Prompter
from src.core.tools.universal_parser import UniversalParser
from src.core.schemas.models import ModuleParser

logger = logging.getLogger(__name__)

class CodeUnderstandingAgent(BaseAgent):
    """
    Agent responsible for analyzing and understanding code snippets or files.
    It uses the UniversalParser to extract structured information.
    """

    def __init__(self, llm_client: GeminiClient, prompter: Prompter):
        super().__init__(llm_client, prompter)
        # Instantiate UniversalParser directly with the client
        self.parser = UniversalParser(llm_client)

    async def run(self, file_path: str) -> Optional[ModuleParser]:
        """
        Analyzes the provided file and returns a structured ModuleParser object.

        Args:
            file_path: The absolute path to the file to analyze.

        Returns:
            ModuleParser object containing the parsed structure, or None if failed.
        """
        logger.info(f"CodeUnderstandingAgent processing: {file_path}")
        try:
            # In the future, we could check extension and pick a specific parser.
            # For now, UniversalParser handles everything via LLM.
            # BaseTool.run takes tool_input/file_path
            parsed_module = await self.parser.run(tool_input=file_path)
            return parsed_module
        except Exception as e:
            logger.error(f"CodeUnderstandingAgent failed for {file_path}: {e}", exc_info=True)
            return None
