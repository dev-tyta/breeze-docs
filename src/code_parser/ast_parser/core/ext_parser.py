from abc import ABC, abstractmethod
from typing import Optional, Any
from src.code_parser.ast_parser.core.ast_node import ASTNode


class ASTExternalParser(ABC):
    """
    Base class for external parsers for transversing ASTs.
    """

    def visit(self, node:ASTNode) -> Any:
        """
        Visit a node and return result
        """

        method = f"visit_{node.type_name}"
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)
    

    @abstractmethod
    def generic_visit(self, node:ASTNode) -> Any:
        """
        Default visitor method
        """
        pass

