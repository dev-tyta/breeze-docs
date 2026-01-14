import logging
from typing import Dict, Any, List, Optional

# Configure logging for this module
logger = logging.getLogger(__name__)
# Basic logging setup (can be enhanced in src/utils/logging.py)
# Ensure this doesn't duplicate handlers if src/utils/logging.py is used
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define a base system message to set the LLM's role and guidelines
BASE_SYSTEM_MESSAGE = """
You are an expert software engineer and technical writer. Your task is to analyze code and generate clear, concise, and accurate documentation for both technical and non-technical users.

Follow these guidelines:
- Explain the purpose and functionality of the code.
- Describe inputs, outputs, and any side effects.
- Mention dependencies and important considerations.
- Use clear and simple language, avoiding jargon where possible, especially for non-technical explanations.
- Format the output according to the user's specific instructions or templates.
"""

# Define templates or structures for different types of prompts
# These can be simple strings or more complex dictionaries/objects
PROMPT_TEMPLATES = {
    "document_code_element": """
Analyze the following code element (function, class, or module) and generate documentation.

Code Element:
```
{code_snippet}
```

File Path: {file_path}
Language: {language}

Documentation Requirements:
- Provide a summary of its purpose.
- Explain its parameters/arguments.
- Describe what it returns (if any).
- Detail any exceptions it might raise.
- Explain any important internal logic or side effects.
- Keep the explanation concise but informative.

Format the output in Markdown. Focus specifically on documenting this single element.
""",
    "generate_usage_example": """
Provide a realistic and clear usage example for the following code element.

Code Element:
```
{code_snippet}
```

File Path: {file_path}
Language: {language}

Usage Example Requirements:
- Show how to initialize/call the element.
- Include typical inputs and demonstrate how to handle outputs.
- Keep the example self-contained and easy to understand.
- If applicable, show a basic use case.

Format the output as a code block in the specified language, followed by a brief explanation in Markdown.
""",
    "summarize_file": """
Analyze the following code file and provide a high-level summary.

File Path: {file_path}
Language: {language}

Code Content:
```
{code_content}
```

Summary Requirements:
- Describe the overall purpose of the file.
- List the main functions, classes, or components defined in the file.
- Explain how these components relate to each other within the file.
- Mention any external dependencies imported by this file.

Format the output in Markdown.
""",
    "summarize_architecture": """
Analyze the provided information about the codebase structure and relationships
and generate a high-level overview of the architecture.

Codebase Structure Information:
{structure_info}

Architecture Summary Requirements:
- Describe the main directories and their roles.
- Explain the key modules and how they interact.
- Illustrate the data flow or control flow if possible.
- Mention important dependencies or external services.

Format the output in Markdown.
""",
    "generate_project_readme": """
Generate a high-quality, production-ready README.md for this project.

Project Name: {project_name}

List of Feature Highlights (Code Examples & Summaries):
{feature_highlights}

Project Structure Information:
{project_structure}

Structure:
1.  **Header**: Project Name and a catchy tagline/description.
2.  **Overview**: A compelling introduction to the project's purpose and goals.
3.  **🔥 Key Features**: A bulleted list of 3-5 main capabilities.
4.  **🚀 Installation**: Standard installation instructions (pip/git).
5.  **📂 Project Structure**: A standard directory tree representation (`tree` command style).
6.  **🏁 Feature Highlights**: Use the provided "List of Feature Highlights" to create this section. For each key component, show a *usage code snippet* first, followed by a brief 1-sentence explanation. This replaces bulky API documentation.
7.  **🤝 Contributing**: Standard placeholder text for contributions.
8.  **📄 License**: Standard Apache 2.0 or MIT license placeholder.

Style Guidelines:
- Use emojis in headers as shown above.
- Focus on *showing* how it works (code-first) rather than *telling* (text-heavy).
- Keep descriptions concise and marketing-friendly.
- Markdown format.
""",
    "generate_feature_highlight": """
Create a "Feature Highlight" for the following code element.
This should be a clean, distinct usage example demonstrating the core value of this component.

Code Element:
```
{code_snippet}
```

File Path: {file_path}

Requirements:
1.  **Title**: A short, action-oriented title (e.g., "Define a Single Agent", "Calculate Metrics").
2.  **Code Block**: A self-contained, realistic Python code snippet showing how to initialize and use this component.
3.  **Description**: A very brief (1-2 sentences) explanation of what the code does.

Do NOT include parameter lists, return types tables, or dry technical details. Focus on USAGE.
"""
}


class Prompter:
    """
    Manages the creation and formatting of prompts for the LLM.
    """

    def __init__(self, base_system_message: str = BASE_SYSTEM_MESSAGE, templates: Dict[str, str] = PROMPT_TEMPLATES):
        """
        Initializes the Prompter with base messages and templates.

        Args:
            base_system_message: The default system message to include in prompts.
            templates: A dictionary of prompt templates by name.
        """
        self._base_system_message = base_system_message
        self._templates = templates
        logger.info("Prompter initialized.")

    def _build_prompt(self, template_name: str, **kwargs: Any) -> str:
        """
        Builds a prompt string using a specified template and provided data.

        Args:
            template_name: The name of the template to use (key in self._templates).
            **kwargs: Data to format the template string.

        Returns:
            The formatted prompt string.

        Raises:
            ValueError: If the template name is not found.
        """
        try:
            template = self._templates[template_name]
            # Use f-string formatting with provided kwargs
            # Note: This requires template strings to use f-string syntax like {variable_name}
            # For more complex templating with logic, consider using a library like Jinja2
            prompt_content = template.format(**kwargs)
            logger.debug(f"Built prompt using template '{template_name}'.")
            return f"{self._base_system_message}\n\n{prompt_content}"
        except KeyError:
            logger.error(f"Prompt template '{template_name}' not found.")
            raise ValueError(f"Prompt template '{template_name}' not found.")
        except Exception as e:
            logger.error(f"Failed to format prompt using template '{template_name}': {e}", exc_info=True)
            raise ValueError(f"Failed to format prompt: {e}") from e


    def for_code_element(self, code_snippet: str, file_path: str, language: str) -> str:
        """
        Builds a prompt for documenting a specific code element.

        Args:
            code_snippet: The source code of the element (function, class, etc.).
            file_path: The path to the file containing the element.
            language: The programming language of the code.

        Returns:
            The formatted prompt string.
        """
        return self._build_prompt(
            "document_code_element",
            code_snippet=code_snippet,
            file_path=file_path,
            language=language
        )

    def for_usage_example(self, code_snippet: str, file_path: str, language: str) -> str:
        """
        Builds a prompt for generating a usage example for a code element.

        Args:
            code_snippet: The source code of the element.
            file_path: The path to the file containing the element.
            language: The programming language of the code.

        Returns:
            The formatted prompt string.
        """
        return self._build_prompt(
            "generate_usage_example",
            code_snippet=code_snippet,
            file_path=file_path,
            language=language
        )

    def for_file_summary(self, file_path: str, language: str, code_content: str) -> str:
        """
        Builds a prompt for summarizing an entire code file.

        Args:
            file_path: The path to the file.
            language: The programming language.
            code_content: The full source code content of the file.

        Returns:
            The formatted prompt string.
        """
        return self._build_prompt(
            "summarize_file",
            file_path=file_path,
            language=language,
            code_content=code_content
        )

    def for_architecture_summary(self, structure_info: str) -> str:
        """
        Builds a prompt for summarizing the overall codebase architecture.

        Args:
            structure_info: Information about the codebase structure and relationships.

        Returns:
            The formatted prompt string.
        """
        return self._build_prompt(
            "summarize_architecture",
            structure_info=structure_info
        )

    def for_project_readme(self, project_name: str, feature_highlights: str, project_structure: str = "") -> str:
        """
        Builds a prompt for generating the top-level project README.

        Args:
            project_name: Name of the project.
            feature_highlights: Consolidated text of feature highlights (code + summary).
            project_structure: Textual representation of the project structure.

        Returns:
           The formatted prompt string.
        """
        return self._build_prompt(
            "generate_project_readme",
            project_name=project_name,
            feature_highlights=feature_highlights,
            project_structure=project_structure
        )

    def for_feature_highlight(self, code_snippet: str, file_path: str) -> str:
        """
        Builds a prompt for generating a feature highlight (usage example).
        """
        return self._build_prompt(
            "generate_feature_highlight",
            code_snippet=code_snippet,
            file_path=file_path
        )

    # Add more methods here for other prompt types as needed
    # e.g., for explaining relationships between elements, generating README content, etc.

# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Ensure logging is set up if running this file directly
    if not logging.getLogger().handlers:
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    prompter = Prompter()

    # Example: Prompt for documenting a function
    function_code = """
def calculate_square(x):
    \"\"\"Calculates the square of a number.\"\"\"
    return x * x
"""
    doc_prompt = prompter.for_code_element(
        code_snippet=function_code,
        file_path="my_module.py",
        language="Python"
    )
    print("--- Documentation Prompt ---")
    print(doc_prompt)
    print("-" * 25)

    # Example: Prompt for a usage example
    usage_prompt = prompter.for_usage_example(
        code_snippet=function_code,
        file_path="my_module.py",
        language="Python"
    )
    print("\n--- Usage Example Prompt ---")
    print(usage_prompt)
    print("-" * 25)

    # Example: Prompt for file summary
    file_content = """
import os

def main_function():
    pass

class HelperClass:
    pass
"""
    file_summary_prompt = prompter.for_file_summary(
        file_path="another_file.py",
        language="Python",
        code_content=file_content
    )
    print("\n--- File Summary Prompt ---")
    print(file_summary_prompt)
    print("-" * 25)
