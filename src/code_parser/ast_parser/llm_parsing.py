import sys
import os
from typing import Dict, Any
from pathlib import Path
import logging

parent_dir = "/home/testys/Documents/GitHub/breeze_docs"
sys.path.append(str(parent_dir))

from src.llm.core import BreeLLM
from llm.internals.parser_structure import ModuleParser


class LLMCodeParser:
    """
    Code Parser using the LLM Capabilities and prompt engineering.

    """
    DEFAULT_FILE_EXTENSIONS = {".py": "python",
                             ".java": "java",
                             ".c": "c",
                             ".cpp": "cpp",
                             ".js": "javascript",
                             ".ts": "typescript",
                             ".html": "html",
                             ".css": "css",
                             ".scss": "scss",
                             ".json": "json",
                             ".xml": "xml",
                             ".yaml": "yaml",
                             ".yml": "yaml",
                             ".sh": "bash",
                             ".bat": "batch",
                             ".ps1": "powershell",
                             ".sql": "sql",
                             ".php": "php",
                             ".rb": "ruby",
                             ".go": "go",
                             ".rs": "rust",
                             ".kt": "kotlin",
                             ".swift": "swift",
                             ".pl": "perl",
                             ".lua": "lua",
                             ".r": "r",
                             ".m": "matlab",
                             ".ipynb": "jupyter",
                             ".md": "markdown",
                             ".txt": "text",
                             ".log": "log",
                             ".jl": "julia"}
    
    def __init__(self, file_path):
        self.file_name = os.path.basename(file_path)

        
    @property
    def language(self) -> str:
        # use the extensions to get the language for the codebase.
        name: str = self.DEFAULT_FILE_EXTENSIONS.get(os.path.splitext(self.file_name)[1], "unknown")

        return name
    

    def _define_prompt(self, content:str) -> str:
        prompt = f"""
                Analyze the following {self.language} code and its structure.
                Identify all imports, global_variables, functions, classes, variables and other essential components of the codebase.
                For functions, extract the parameters, decorators, docstring, contents, return types, and implementation.
                for classes, identify methods, attributes, parent_classes, and docstrings.


                Code to analyze:
                ```{self.language}
                {content}
                ```
                Provide a structured response that can be parsed into the OutputParser.
            """
        
        return prompt
    

    def parse(self, content:str) 