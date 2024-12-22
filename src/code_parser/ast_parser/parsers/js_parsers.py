import esprima
import sys
from typing import Dict, Any
from pathlib import Path
import logging

parent_dir = "/home/testys/Documents/GitHub/breeze_docs"
sys.path.append(str(parent_dir))

from src.code_parser.ast_parser.core.base_parser import BaseParser
from src.code_parser.ast_parser.core.ast_node import ASTNode

class JavaScriptParser(BaseParser):
    """
    JavaScript code parser using the esprima module.
    """

    @property
    def language(self) -> str:
        return "javascript"
    

    def parse(self, content: str) -> ASTNode:
        try: 
            tree = esprima.parseScript(content)
            return self._js_ast_to_dict(tree)
        except Exception as e:
            raise ValueError(f"JavaScript parsing error: {str(e)}")
        

    def parse_file(self, file_path:str) -> ASTNode:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        return self.parse(content)
    
    def _convert_js_ast(self, node: Any) -> ASTNode:
        """
        Convert JavaScript AST to dictionary
        """
        if isinstance(node, dict):
            return {k: self._convert_js_ast(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [self._convert_js_ast(x) for x in node]
        else:
            return node

    def _js_ast_to_dict(self, node) -> ASTNode:
        """Helper method to convert JavaScript AST to dictionary"""
        if hasattr(node, 'toDict'):
            return node.toDict()
        elif isinstance(node, dict):
            return {k: self._js_ast_to_dict(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [self._js_ast_to_dict(x) for x in node]
        else:
            return node
        


# Usage example
parser = JavaScriptParser()
out_js = parser.parse_file("/home/testys/Documents/GitHub/breeze_docs/data/samples/sample_parsers/hello.js")
print(out_js)