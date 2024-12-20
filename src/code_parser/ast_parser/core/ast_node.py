from typing import Any, List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ASTNode:
    """
    Base class for AST nodes with essential details.


    """

    type_name: str
    value: Any = None
    children: List["ASTNode"] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_code: Optional[str] = None
    source_file: Optional[str] = None
    language: Optional[str] = None

    def add_child(self, child: "ASTNode") -> None:
        """
        Add a chile node to this node
        """
        self.children.append(child)

    def find_children(self, type_name: str) -> List["ASTNode"]:
        """
        Find all children nodes of a given type
        """
        return [child for child in self.children if child.type_name == type_name]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the AST node to a dictionary
        """
        fields = {
            "type": self.type_name,
            "value": self.value,
            "attributes": self.attributes,

        }
        if self.children:
            fields["children"] = [child.to_dict() for child in self.children]

        return fields