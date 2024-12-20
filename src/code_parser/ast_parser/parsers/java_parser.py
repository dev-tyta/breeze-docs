import javalang
from typing import Any, Dict
import sys
from pathlib import Path
import logging


# Add parent directory to system path to resolve imports
parent_dir = "/home/testys/Documents/GitHub/breeze_docs"
sys.path.append(str(parent_dir))

from src.code_parser.ast_parser.core.base_parser import BaseParser
from src.code_parser.ast_parser.core.ast_node import ASTNode


class JavaParser(BaseParser):
    """
    Java code parser using the javalang module.
    """

    @property
    def language(self) -> str:
        return "java"
    
    def parse(self, content:str) -> ASTNode:
        try:
            tree = javalang.parse.parse(content)
            return self._java_ast_to_dict(tree)
        except Exception as e:
            raise ValueError(f"Java parsing error: {str(e)}")
        

    def parse_file(self, file_path:str) -> ASTNode:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        return self.parse(content)
    
    def _java_ast_to_dict(self, node) -> Dict[str, Any]:
        """Helper method to convert Java AST to dictionary"""
        if hasattr(node, '__dict__'):
            return {
                'type': node.__class__.__name__,
                'attributes': {
                    key: self._java_ast_to_dict(value)
                    for key, value in node.__dict__.items()
                    if not key.startswith('_')
                }
            }
        elif isinstance(node, list):
            return [self._java_ast_to_dict(x) for x in node]
        else:
            return node
    

# Usage Example
java_parser = JavaParser()
ast = java_parser.parse_file("/home/testys/Documents/GitHub/breeze_docs/data/samples/sample_parsers/hello.java")
print(ast)