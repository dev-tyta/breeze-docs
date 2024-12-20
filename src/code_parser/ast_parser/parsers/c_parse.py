#TODO: Fix bugs on the CParse class. Might have to try using the documentation to get more insight.

from pycparser import c_parser, preprocess_file
import pycparser
import subprocess
import tempfile
from typing import Dict, Any
import sys
from pathlib import Path
import logging


# Add parent directory to system path to resolve imports
parent_dir = "/home/testys/Documents/GitHub/breeze_docs"
sys.path.append(str(parent_dir))

from src.code_parser.ast_parser.core.base_parser import BaseParser
from src.code_parser.ast_parser.core.ast_node import ASTNode


class ParseC(BaseParser):
    """
    C and C++ code parser using the pycparser module.
    """
    
    @property
    def language(self) -> str:
        return "c"

    def parse(self, content: str) -> ASTNode:
        """Parse C code string into AST"""
        try:
            # Parse the preprocessed content
            parser = c_parser.CParser()
            ast = parser.parse(content)
            return self._c_ast_to_dict(ast) 
           
        except Exception as e:
            raise ValueError(f"C parsing error: {str(e)}")
            
    def parse_file(self, file_path:str) -> ASTNode:
        content = preprocess_file(file_path)
        
        return self.parse(content)
    
    def _c_ast_to_dict(self, node) -> Dict[str, Any]:
        """Helper method to convert C AST to dictionary"""
        if isinstance(node, tuple):
            return [self._c_ast_to_dict(n) for n in node]
        elif isinstance(node, pycparser.c_ast.Node):
            return {
                'type': node.__class__.__name__,
                'attributes': {
                    attr: self._c_ast_to_dict(getattr(node, attr))
                    for attr in node.attr_names
                },
                'children': {
                    c[0]: self._c_ast_to_dict(c[1])
                    for c in node.children()
                }
            }
        else:
            return node

    def _convert_c_ast(self, node) -> ASTNode:
        """Convert C AST to custom ASTNode format"""
        if node is None:
            return None
            
        ast_node = ASTNode(
            type_name=node.__class__.__name__,
            language="c"
        )
        
        # Add attributes
        for attr in node.attr_names:
            ast_node.attributes[attr] = getattr(node, attr)
        
        # Process children
        for name, child in node.children():
            child_node = self._convert_c_ast(child)
            if child_node:
                child_node.attributes['name'] = name
                ast_node.add_child(child_node)
                
        return ast_node
        
# Usage Example
c_parsing = ParseC()
ast = c_parsing.parse_file("/home/testys/Documents/GitHub/breeze_docs/data/samples/sample_parsers/hello.c")
print(ast)