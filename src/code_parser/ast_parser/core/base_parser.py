from abc import ABC, abstractmethod
from typing import Optional
import sys
from pathlib import Path
import logging


# Add parent directory to system path to resolve imports
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent.parent.parent.parent
sys.path.append(str(parent_dir))

from src.code_parser.ast_parser.core.ast_node import ASTNode


class BaseParser(ABC):
    """
    Base class for parsers for all languages parsers
    """

    @abstractmethod
    def parse(self, content:str ) -> Optional[ASTNode]:
        """
        Parse the content of a code file and return the AST root node.
        """
        pass

    @abstractmethod
    def parse_file(self, file_path:str) -> Optional[ASTNode]:
        """
        Parse a code file and return the AST root node.
        """
        pass

    @property
    @abstractmethod
    def language(self) -> str:
        """
        The programming language of the parser.
        """
        pass