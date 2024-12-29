import ast
from typing import Dict, Any
import sys
from pathlib import Path
import logging


# Add parent directory to system path to resolve imports
parent_dir = "/home/testys/Documents/GitHub(/breeze_docs"
sys.path.append(str(parent_dir))

from src.code_parser.ast_parser.core.base_parser import BaseParser
from src.code_parser.ast_parser.core.ast_node import ASTNode


class PythonParser(BaseParser):
    """
    Python code parser using the built-in ast module.
    """

    @property
    def language(self) -> str:
        return "python"
    

    def parse(self, content: str) -> ASTNode:
        try: 
            tree = ast.parse(content)
            return self._convert_python_ast(tree)
        except Exception as e:
            raise ValueError(f"Python parsing error: {str(e)}")
        

    def parse_file(self, file_path:str) -> ASTNode:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        return self.parse(content)
    

    def _convert_python_ast(self, node: Any) -> ASTNode:
        """
        Convert Python AST to ASTNode
        """
        if isinstance(node, ast.AST):
            ast_node = ASTNode(
            type_name = node.__class__.__name__,
            language = "python"
            )
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        child = self._convert_python_ast(item)
                        if child:
                            ast_node.add_child(child)
                else:
                    child = self._convert_python_ast(value)
                    if child:
                        ast_node.add_child(child)

            return ast_node
        
        elif isinstance(node, (str, int, float, bool)):
            return ASTNode(
                type_name = "literal",
                value = node,
                language = "python"
            )
        return None

    def _ast_to_dict(self, node) -> Dict[str, Any]:
        """Helper method to convert Python AST to dictionary"""
        if isinstance(node, ast.AST):
            fields = {}
            for field, value in ast.iter_fields(node):
                fields[field] = self._ast_to_dict(value)
            return {node.__class__.__name__: fields}
        elif isinstance(node, list):
            return [self._ast_to_dict(x) for x in node]
        else:
            return node
    
    


# Usage Example
python_parser = PythonParser()
ast_out = python_parser.parse_file("/home/testys/Documents/GitHub/breeze_docs/data/samples/sample_parsers/services.py")
print(ast_out)

