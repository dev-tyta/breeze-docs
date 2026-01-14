import logging
from typing import Optional, List

from src.core.agents.base import BaseAgent
from src.core.schemas.models import ModuleParser, FunctionParser, ClassParser
from src.core.llm.client import GeminiClient
from src.core.llm.prompter import Prompter
from src.utils.text_processing import smart_chunk

logger = logging.getLogger(__name__)

class DocumentationGenerationAgent(BaseAgent):
    """
    Agent responsible for generating documentation for parsed code modules.
    Inherits from BaseAgent.
    """
    
    def __init__(self, llm_client: GeminiClient, prompter: Prompter):
        super().__init__(llm_client, prompter)

    async def run(self, module_info: ModuleParser) -> str:
        """
        Generates 'Feature Highlight' style documentation for the module.
        RATHER than bulky API docs, it identifies key components and generates usage examples.
        """
        logger.info(f"Generating feature highlights for module: {module_info.name}")
        
        doc_parts = []
        
        # 1. Module Overview (still useful)
        # Use simple summary, derived from docstring if available, or generate brief one.
        if module_info.docstring:
            doc_parts.append(f"**Module {module_info.name}**: {module_info.docstring.strip().splitlines()[0]}")
        else:
             doc_parts.append(f"**Module {module_info.name}**")
        
        # 2. Identify Key Components for Highlights
        # Heuristic: Public Classes and Functions (not starting with _)
        candidates = []
        candidates.extend([(c.name, c.content, 'class') for c in module_info.classes if not c.name.startswith('_')])
        candidates.extend([(f.name, f.content, 'function') for f in module_info.functions if not f.name.startswith('_')])
        
        # Limit candidates if there are too many to avoid context blowout, or prioritize.
        # For now, take top 3-5
        top_candidates = candidates[:5] 
        
        for name, content, type_ in top_candidates:
            # Generate Feature Highlight (Usage Example)
            prompt = self._prompter.for_feature_highlight(
                code_snippet=content,
                file_path=module_info.file_path
            )
            try:
                highlight = await self._llm_client.generate(prompt)
                doc_parts.append(highlight)
            except Exception as e:
                logger.warning(f"Failed to generate highlight for {name}: {e}")
        
        if not top_candidates and module_info.raw_content:
             # If no specific components, maybe summarize the file usage generally?
             # For now, just skip.
             pass

        # Join all parts
        return "\n\n".join(doc_parts)

    async def generate_project_readme(self, project_name: str, feature_highlights: List[str], project_structure: str = "") -> str:
        """
        Generates a high-level project README based on list of feature highlights.
        """
        logger.info(f"Generating project-level README for {project_name}")
        
        # Consolidate highlights into a single string for the prompt
        consolidated_highlights = "\n---\n".join(feature_highlights)
        
        prompt = self._prompter.for_project_readme(project_name, consolidated_highlights, project_structure)
        
        try:
            readme_content = await self._llm_client.generate(prompt)
            return readme_content
        except Exception as e:
            logger.error(f"Failed to generate project README: {e}")
            return f"# {project_name}\n\n*Automated README generation failed.*"
