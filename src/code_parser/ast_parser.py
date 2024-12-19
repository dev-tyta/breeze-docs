#TODO:Use language-specific parsers to analyze the code structure and extract relevant elements (functions, classes, modules).
#TODO:Arrange the parsed information into a consistent JSON-like format.
#TODO:Identify relationships between code files, such as import/export dependencies and cross-module function calls.
#TODO:Generate a high-level understanding of the architecture for later prompts.

import os
import ast
import logging
import tree_sitter


# class SyntaxParser:
#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
#         logging.basicConfig(level=logging.INFO)
#         self.parsed_code = {}
#         s

    
#     def parse_python_file(self, file_path):


# testing parser
parser = tree_sitter.Parser()
with open("/home/testys/Documents/GitHub/TheTherapist/api.py", "rb") as f:
    tree = parser.parse(f.raw, encoding="utf8")
    tree = parser.tree
    print(tree)