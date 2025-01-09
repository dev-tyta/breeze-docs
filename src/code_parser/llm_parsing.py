import sys
import os
import json
from typing import Dict, Any
from pathlib import Path
import logging

from src.llm.core import BreeLLM
from src.llm.internals.parser_structure import ModuleParser


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
        logging.basicConfig(level=logging.INFO)
        logging.info("LLM Code Parser Initialized")
        self.file_path = file_path
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
        logging.info(f"Prompt defined: {prompt}")
        return prompt
    

    def parse(self, content:str) -> ModuleParser:
        try:
            input_prompt = self._define_prompt(content=content)
            logging.info(f"Prompt defined: {input_prompt}")
            self.model = BreeLLM(input_prompt=input_prompt, query="", output_struct=ModuleParser, tools=None)
            logging.info("Model initialized")
            output = self.model.generate_response()
            logging.info(f"Model output: {output}")
            return ModuleParser(**output)
        except Exception as e:
            logging.error(f"Error parsing code: {str(e)}")
            return None
        
    
    def _parse_to_json(self, parsed:ModuleParser) -> Dict[str, Any]:
        dict_output = parsed.dict()
        logging.info(f"Output parsed to dictionary: {dict_output}")
        with open(f"./{os.path.splitext(self.file_name)[0]}.json", "w") as file:
            json.dump(dict_output, file)

        return dict_output
    
    def parse_file(self) -> ModuleParser:
        with open(self.file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        out_parse = self.parse(content)
        
        self._parse_to_json(out_parse)

        return out_parse

    
    def parse_directory(self, directory_path:str) -> Dict[str, ModuleParser]:
        parsed_files = {}
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(tuple(self.DEFAULT_FILE_EXTENSIONS.keys())):
                    file_path = os.path.join(root, file)
                    parsed_files[file] = self.parse_file(file_path=file_path)
        
        return parsed_files
    
    
# Usage Example
parser = LLMCodeParser(file_path="/workspaces/breeze-docs/src/code_parser/ast_parser.py")
out = parser.parse_file()
print(out)