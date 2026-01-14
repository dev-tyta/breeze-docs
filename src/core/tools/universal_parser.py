import os
import logging
import json
import re
from typing import Any, Dict, Optional, List
from pathlib import Path
from pydantic import ValidationError

from src.core.tools.base import BaseTool
from src.core.llm.client import GeminiClient
from src.core.schemas.models import ModuleParser
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

class UniversalParser(BaseTool):
    """
    A universal code parser powered by an LLM (Gemini).
    It parses source code into a structured ModuleParser object.
    """

    def __init__(self, llm_client: GeminiClient):
        super().__init__(name="universal_parser", description="Parses code using LLM")
        self.llm_client = llm_client

    async def run(self, tool_input: str, *args: Any, **kwargs: Any) -> Optional[ModuleParser]:
        """
        Parses the given file path or code content.
        
        Args:
            tool_input: Absolute file path to the file to parse.
        
        Returns:
            ModuleParser object or None if parsing fails.
        """
        file_path = tool_input
        if not os.path.exists(file_path):
             logger.error(f"File not found: {file_path}")
             raise ValueError(f"File not found: {file_path}")
             
        try:
             with open(file_path, "r", encoding="utf-8") as f:
                 content = f.read()
        except Exception as e:
             logger.error(f"Failed to read file {file_path}: {e}")
             raise
             
        return await self.parse_content(content, file_path)

    async def parse_content(self, content: str, file_path: str) -> Optional[ModuleParser]:
        """
        Parses code content using the LLM.
        """
        file_name = os.path.basename(file_path)
        prompt = self._construct_prompt(content, file_name)
        
        try:
            # We want JSON output. Gemini supports response_mime_type="application/json" generally,
            # but via LangChain interface or raw we need to be careful. 
            # The existing GeminiClient.generate returns a string.
            # We will ask for JSON in the prompt and then parse it.
            
            response = await self.llm_client.generate(prompt)
            return self._parse_llm_response(response, file_path, content)
            
        except Exception as e:
            logger.error(f"Error during LLM parsing of {file_path}: {e}", exc_info=True)
            return None

    def _construct_prompt(self, content: str, file_name: str) -> str:
        # Get the schema as a simplified JSON structure to guide the model
        schema_hint = """
        {
            "name": "module_name",
            "file_path": "path/to/file",
            "imports": ["import os", ...],
            "functions": [
                {
                    "name": "func_name",
                    "start_line": 1,
                    "end_line": 10,
                    "content": "def func()...",
                    "body": "...",
                    "type": "function",
                    "parameters": [{"name": "arg1", "type_annotation": "int", "default_value": "None"}],
                    "docstring": "..."
                }
            ],
            "classes": [
                {
                    "name": "ClassName",
                    "start_line": 12,
                    "end_line": 20,
                    "content": "class ClassName...",
                    "body": "...",
                    "type": "class",
                    "docstring": "..."
                }
            ],
            "global_variables": [
                {"name": "VAR", "type_annotation": "str", "value": "'val'"}
            ],
            "docstring": "Module docstring"
        }
        """
        
        return f"""
        You are an expert code parser. Your task is to analyze the following source code file named '{file_name}' and extract its structure into a JSON format.
        
        Strictly follow this JSON schema structure:
        {schema_hint}
        
        Requirements:
        1. Extract all imports, functions, classes, and global variables.
        2. For functions and classes, provide PRECISE start_line and end_line (1-indexed).
        3. Extract docstrings if available.
        4. Capture the full content of the element in 'content'.
        5. Capture the body content (excluding signature) in 'body'.
        6. Extract parameters for functions.
        7. The output MUST be valid JSON. Do not include markdown code block markers (```json ... ```).
        
        Source Code:
        {content}
        """

    def _parse_llm_response(self, response: str, file_path: str, raw_content: str) -> Optional[ModuleParser]:
        # Clean up the response (remove markdown code blocks if present)
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        elif clean_response.startswith("```"):
            clean_response = clean_response[3:]
            
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        
        clean_response = clean_response.strip()

        try:
            data = json.loads(clean_response)
            
            # Ensure file_path is set correctly (LLM might hallucinate it)
            data['file_path'] = file_path
            data['raw_content'] = raw_content
            # Ensure name matches file name (or what LLM found)
            if 'name' not in data:
                 data['name'] = os.path.splitext(os.path.basename(file_path))[0]

            # Validate with Pydantic
            # Note: We need to be careful with optional fields in the schema.
            # ModuleParser expects certain types. We rely on Pydantic's parsing.
            module_parser = ModuleParser(**data)
            return module_parser
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LLM for {file_path}: {e}")
            logger.debug(f"Raw response: {response}")
            return None
        except ValidationError as e:
            logger.error(f"Pydantic validation failed for {file_path}: {e}")
            logger.debug(f"Data: {data}")
            return None
