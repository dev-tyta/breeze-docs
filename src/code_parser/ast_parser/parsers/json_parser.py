import json
from typing import Dict, Any
import sys
from pathlib import Path
import logging

parent_dir = "/home/testys/Documents/GitHub/breeze_docs"
sys.path.append(str(parent_dir))

from src.code_parser.ast_parser.core.base_parser import BaseParser
from src.code_parser.ast_parser.core.ast_node import ASTNode


class JSONParser(BaseParser):
    """
    JSON Parser using the built-in json module.
    """

    @property
    def language(self) -> str:
        return "json"
    
        
    def parse(self, content: str) -> Dict[str, Any]:
        """Parse JSON using built-in json module"""
        try:
            return json.loads(content)
        except Exception as e:
            raise ValueError(f"JSON parsing error: {str(e)}")


    def parse_file(self, file_path:str) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        return self.parse(content)


            
    