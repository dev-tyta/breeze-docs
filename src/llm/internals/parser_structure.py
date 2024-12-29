from typing import List, Dict, Optional, Any
import os
from pydantic import BaseModel, Field, model_validator, root_validator


class ElementParser(BaseModel):
    """Output Structure for the parsed code elements"""
    name: str
    start_line: int
    end_line: int
    content: str
    type: str = "element"


class FunctionParser(ElementParser):
    """Output Structure for parsed functions"""
    type: str = "function"
    parameters: List[Dict[str, str]] = Field(default_factory=list)
    decorators: List[str] = Field(default_factory=list)
    docstring: Optional[str] = None
    content: Optional[str] = Field(default=None, alias="body")
    return_annotation: Optional[str] = None


class ClassParser(ElementParser):
    """Output Structure for parsed classes"""
    type: str = "class"
    decorators: List[str] = Field(default_factory=list)
    docstring: Optional[str] = None
    content: Optional[str] = Field(default=None, alias="body")
    bases: List[str] = Field(default_factory=list)


class ModuleParser(BaseModel):
    """Output Structure for the parsed module"""
    name: str
    file_path: str
    type: str = "module"
    imports: List[str] = Field(default_factory=list)
    functions: List[FunctionParser] = Field(default_factory=list)
    classes: List[ClassParser] = Field(default_factory=list)
    docstring: Optional[str] = None
    global_variables: List[Dict[str, Any]] = Field(default_factory=list)
    raw_content: Optional[str] = None

    @root_validator
    def validate_file_path(cls, values):
        if not os.path.exists(values.get('file_path')):  # Fixed exists check
            raise ValueError(f"File path {values.get('file_path')} does not exist")
        return values
    
    @root_validator
    def validate_line_numbers(cls, values):
        """Add validation for line numbers in elements"""
        for element_list in [values.get('functions', []), values.get('classes', [])]:
            for element in element_list:
                if element.start_line >= element.end_line:
                    raise ValueError(f"Invalid line numbers for {element.name}: start_line must be less than end_line")
        return values