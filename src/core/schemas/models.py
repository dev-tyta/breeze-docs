import os
import logging
from typing import List, Dict, Optional, Any, Union # Import Union for flexibility
from pydantic import BaseModel, Field, model_validator, ValidationError # Import ValidationError
from datetime import datetime # Import datetime for last_modified field

# Configure logging for this module
logger = logging.getLogger(__name__)
# Basic logging setup (can be enhanced in src/utils/logging.py)
# Ensure this doesn't duplicate handlers if src/utils/logging.py is used
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Basic Code Element Schemas ---

class Parameter(BaseModel):
    """Represents a function or method parameter."""
    name: str = Field(..., description="The name of the parameter.")
    type_annotation: Optional[str] = Field(None, description="The type annotation of the parameter, if any.")
    default_value: Optional[str] = Field(None, description="The default value of the parameter, if any.")
    # Add more fields if needed, e.g., 'kind' (positional, keyword, etc.)

class Variable(BaseModel):
    """Represents a variable (e.g., global variable, class attribute)."""
    name: str = Field(..., description="The name of the variable.")
    type_annotation: Optional[str] = Field(None, description="The type annotation of the variable, if any.")
    value: Optional[Any] = Field(None, description="The value of the variable, if determined.")
    # Add more fields if needed, e.g., 'is_constant'

class ElementParser(BaseModel):
    """Base structure for a parsed code element (function, class, etc.)."""
    name: str = Field(..., description="The name of the code element.")
    start_line: int = Field(..., description="The starting line number of the element in the source file (1-based).")
    end_line: int = Field(..., description="The ending line number of the element in the source file (inclusive).")
    content: str = Field(..., description="The full source code content of the element, including signature/decorators.")
    type: str = Field("element", description="The type of the code element (e.g., 'function', 'class').")

class FunctionParser(ElementParser):
    """Structure for a parsed function or method."""
    type: str = Field("function", description="The type of the code element (always 'function').")
    parameters: List[Parameter] = Field(default_factory=list, description="A list of parameters for the function.") # Using dedicated Parameter model
    decorators: List[str] = Field(default_factory=list, description="A list of decorator names applied to the function.")
    docstring: Optional[str] = Field(None, description="The docstring of the function, if any.")
    # Renamed 'content' alias to 'body_content' for clarity, using Optional[str]
    body_content: Optional[str] = Field(default=None, alias="body", description="The source code content of the function body, excluding signature and docstring.")
    return_annotation: Optional[str] = Field(None, description="The return type annotation of the function, if any.")


class ClassParser(ElementParser):
    """Structure for a parsed class."""
    type: str = Field("class", description="The type of the code element (always 'class').")
    decorators: List[str] = Field(default_factory=list, description="A list of decorator names applied to the class.")
    docstring: Optional[str] = Field(None, description="The docstring of the class, if any.")
    # Renamed 'content' alias to 'body_content' for clarity, using Optional[str]
    body_content: Optional[str] = Field(default=None, alias="body", description="The source code content of the class body, excluding signature and docstring.")
    bases: List[str] = Field(default_factory=list, description="A list of base class names this class inherits from.")
    # Could add nested fields for methods and class variables here if needed
    # methods: List[FunctionParser] = Field(default_factory=list)
    # class_variables: List[Variable] = Field(default_factory=list)


class ModuleParser(BaseModel):
    """Structure for a parsed code module (file)."""
    name: str = Field(..., description="The name of the module (usually the file name without extension).")
    file_path: str = Field(..., description="The absolute or relative path to the source file.")
    type: str = Field("module", description="The type of the code element (always 'module').")
    imports: List[str] = Field(default_factory=list, description="A list of import statements in the module.")
    functions: List[FunctionParser] = Field(default_factory=list, description="A list of functions defined in the module.")
    classes: List[ClassParser] = Field(default_factory=list, description="A list of classes defined in the module.")
    docstring: Optional[str] = Field(None, description="The module-level docstring, if any.")
    global_variables: List[Variable] = Field(default_factory=list, description="A list of global variables defined in the module.") # Using dedicated Variable model
    raw_content: Optional[str] = Field(None, description="The full raw source code content of the file.")

    # --- Pydantic Validators ---

    @model_validator(mode='after') # Use 'after' mode for cross-field validation
    def validate_file_path_exists(self) -> 'ModuleParser':
        """Validates that the file path exists on the filesystem."""
        if not os.path.exists(self.file_path):
            logger.error(f"Validation Error: File path does not exist - {self.file_path}")
            raise ValueError(f"File path {self.file_path} does not exist")
        return self

    @model_validator(mode='after') # Use 'after' mode
    def validate_element_line_numbers(self) -> 'ModuleParser':
        """Validates line numbers for functions and classes within the module."""
        for element_list in [self.functions, self.classes]:
            for element in element_list:
                if element.start_line is not None and element.end_line is not None: # Check if lines are provided
                    # start_line must be less than or equal to end_line for single-line elements
                    if element.start_line > element.end_line:
                         logger.error(f"Validation Error: Invalid line numbers for {element.name} in {self.file_path} - start_line ({element.start_line}) must be less than or equal to end_line ({element.end_line}).")
                         raise ValueError(f"Invalid line numbers for {element.name}: start_line ({element.start_line}) must be less than or equal to end_line ({element.end_line})")
        return self

    # Note: Pydantic v2 uses model_validator. root_validator is deprecated.


# --- Schema for File Metadata ---

class FileMetadata(BaseModel):
    """Represents basic metadata for a source code file."""
    path: str = Field(..., description="The relative path to the file from the root directory.")
    name: str = Field(..., description="The name of the file.")
    extension: str = Field(..., description="The file extension.")
    size: int = Field(..., description="The size of the file in bytes.")
    language: str = Field(..., description="The detected programming language of the file.")
    last_modified: datetime = Field(..., description="The last modified timestamp of the file.")
    # Could add a field for raw content if needed before full parsing
    raw_content: Optional[str] = Field(None, description="The full raw source code content.")


# --- Example of a potential Agent Input/Output Schema ---
# These would be used to define the expected input and output format for agents.

class AgentInput(BaseModel):
    """Base schema for agent input."""
    task_id: str = Field(..., description="A unique identifier for the task.")
    # Add common input fields here

class DocumentCodeElementInput(AgentInput):
    """Input schema for an agent documenting a single code element."""
    code_element: Union[FunctionParser, ClassParser] = Field(..., description="The parsed code element to document.")
    module_info: ModuleParser = Field(..., description="Information about the module containing the element.")
    # Could add context info, settings, etc.

class AgentOutput(BaseModel):
    """Base schema for agent output."""
    task_id: str = Field(..., description="The unique identifier for the completed task.")
    status: str = Field(..., description="The status of the task (e.g., 'success', 'failure').")
    # Add common output fields here

class DocumentationOutput(AgentOutput):
    """Output schema for a documentation generation agent."""
    generated_docs: str = Field(..., description="The generated documentation in markdown format.")
    # Could add metadata, errors encountered, etc.

# --- Example Usage ---
# if __name__ == "__main__":
#     # Ensure logging is set up if running this file directly
#     if not logging.getLogger().handlers:
#          logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
#     # Example of creating a ModuleParser instance (requires a dummy file)
#     dummy_file_path = "temp_dummy_module.py"
#     try:
#         with open(dummy_file_path, "w") as f:
#             f.write("# Dummy file\n\ndef my_func():\n    pass\n\nclass MyClass:\n    pass")
#
#         # Create dummy data conforming to the schemas
#         param_schema = Parameter(name="x", type_annotation="int", default_value="0")
#         func_schema = FunctionParser(
#             name="my_func",
#             start_line=2,
#             end_line=3,
#             content="def my_func():\n    pass",
#             body="    pass",
#             parameters=[param_schema],
#             docstring='"""A dummy function."""'
#         )
#         class_schema = ClassParser(
#             name="MyClass",
#             start_line=5,
#             end_line=6,
#             content="class MyClass:\n    pass",
#             body="    pass",
#             bases=["object"],
#             docstring='"""A dummy class."""'
#         )
#         var_schema = Variable(name="GLOBAL_VAR", type_annotation="str", value='"hello"')
#
#         module_data = {
#             "name": "temp_dummy_module",
#             "file_path": dummy_file_path,
#             "imports": ["os"],
#             "functions": [func_schema.model_dump()], # Use model_dump() to get dictionary representation
#             "classes": [class_schema.model_dump()],
#             "docstring": '"""A dummy module."""',
#             "global_variables": [var_schema.model_dump()],
#             "raw_content": "# Dummy file\n\ndef my_func():\n    pass\n\nclass MyClass:\n    pass"
#         }
#
#         # Validate the data against the ModuleParser schema
#         try:
#             parsed_module = ModuleParser(**module_data)
#             print("Schema validation successful!")
#             print(parsed_module.model_dump_json(indent=2)) # Print as JSON
#         except ValidationError as e:
#             print("Schema validation failed!")
#             print(e.errors())
#
#     finally:
#         # Clean up the dummy file
#         if os.path.exists(dummy_file_path):
#             os.remove(dummy_file_path)
#             print(f"\nCleaned up {dummy_file_path}")
#
#     # Example of creating a FileMetadata instance
#     file_meta = FileMetadata(
#         path="src/utils/example.py",
#         name="example.py",
#         extension=".py",
#         size=1024,
#         language="python",
#         last_modified=datetime.now()
#     )
#     print("\nFile Metadata Schema:")
#     print(file_meta.model_dump_json(indent=2))
#
#     # Example of validation failure (invalid line numbers)
#     # try:
#     #     invalid_module_data = {
#     #         "name": "invalid_module",
#     #         "file_path": "non_existent_file.py", # This will also fail
#     #         "functions": [{"name": "bad_func", "start_line": 5, "end_line": 3, "content": "...", "type": "function"}],
#     #         "classes": [], "imports": [], "global_variables": [], "docstring": None, "raw_content": None
#     #     }
#     #     invalid_module = ModuleParser(**invalid_module_data)
#     # except ValidationError as e:
#     #     print("\nSchema validation correctly failed for invalid data:")
#     #     print(e.errors())
