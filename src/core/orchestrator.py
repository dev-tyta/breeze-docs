import asyncio
import logging
import os
from pathlib import Path
from typing import List
from datetime import datetime

from src.config.settings import get_settings
from src.core.llm.client import GeminiClient
from src.core.llm.prompter import Prompter
from src.core.agents.repo_crawler_agent import RepoCrawlerAgent
from src.core.agents.code_understanding_agent import CodeUnderstandingAgent
from src.core.agents.documentation_agent import DocumentationGenerationAgent
from src.utils.error_handling import CircuitBreaker

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrates the documentation generation pipeline.
    Manages the flow: RepoCrawler -> CodeUnderstanding -> DocumentationGeneration -> Save.
    """

    def __init__(self):
        self.settings = get_settings()
        
        # Initialize Gemini Client with Circuit Breaker
        # Note: We create a new CB instance here, or we could modify settings to include it globally if needed.
        # But per instruction, we initialize everything here.
        cb = CircuitBreaker(
            failure_threshold=self.settings.llm.cb_failure_threshold,
            recovery_timeout=self.settings.llm.cb_recovery_timeout_seconds,
            expected_successes=self.settings.llm.cb_expected_successes
        )
        self.llm_client = GeminiClient(settings=self.settings.llm, circuit_breaker=cb)
        self.prompter = Prompter()
        
        # Initialize Agents
        self.crawler_agent = RepoCrawlerAgent(self.llm_client, self.prompter)
        self.understanding_agent = CodeUnderstandingAgent(self.llm_client, self.prompter)
        self.docs_agent = DocumentationGenerationAgent(self.llm_client, self.prompter)
        
        # Concurrency limit
        self.concurrency_limit = 5

    async def run(self, source_path: str, output_dir: str = "docs", recursive: bool = False):
        """
        Executes the pipeline on the source path.
        
        Args:
            source_path: Path to file or directory.
            output_dir: Directory to save generated documentation.
            recursive: Whether to crawl recursively (passed to crawler).
        """
        logger.info(f"Starting pipeline for: {source_path}")
        
        # Determine project name for the output file
        if source_path.startswith("http") or "github.com" in source_path:
            # Simple heuristic for GitHub URLs or owner/repo strings
            project_name = source_path.rstrip("/").split("/")[-1]
            is_github = True
        elif not os.path.exists(source_path) and "/" in source_path and not source_path.startswith("/"):
             # owner/repo format
             project_name = source_path.split("/")[-1]
             is_github = True
        else:
             project_name = os.path.basename(os.path.abspath(source_path))
             is_github = False
        
        # 1. Crawl
        if os.path.isfile(source_path):
            # Manually construct metadata list for single file
            from src.core.schemas.models import FileMetadata
            
            files_metadata = [FileMetadata(
                path=source_path,
                name=os.path.basename(source_path),
                extension=os.path.splitext(source_path)[1],
                size=os.path.getsize(source_path),
                language="unknown",
                last_modified=datetime.fromtimestamp(os.path.getmtime(source_path)),
                raw_content="" # Will be read by parser later
            )]
            # Read content
            with open(source_path, "r", encoding="utf-8") as f:
                files_metadata[0].raw_content = f.read()

        else:
            files_metadata = await self.crawler_agent.run(source_path, is_github=is_github)
        
        if not files_metadata:
            logger.warning(f"No files found in {source_path}")
            return

        # Sort files by path for consistent structure
        files_metadata.sort(key=lambda x: x.path)
        logger.info(f"Found {len(files_metadata)} files. Starting processing...")

        # 2. Process Files in Parallel
        sem = asyncio.Semaphore(self.concurrency_limit)
        tasks = []
        
        file_count = 0
        for file_meta in files_metadata:
            # Basic filter: Skip hidden files, git, etc.
            # Crawler already filters, but let's be safe but less aggressive.
            # Only check if the filename starts with '.'
            if os.path.basename(file_meta.path).startswith('.'):
                 continue
            
            tasks.append(self._process_single_file(file_meta.path, sem))
            file_count += 1
            
        logger.info(f"Queued {file_count} files for processing.")
        
        feature_highlights_list = []
        
        if tasks:
            # Output project structure
            project_structure_tree = self._generate_project_structure_tree(files_metadata)
        
            # Gather returns results in the same order as tasks
            results = await asyncio.gather(*tasks)
            # Filter out failures
            for r in results:
                if r:
                    path, highlight_content = r
                    # Just collect the highlights. The 'highlight_content' contains the usage examples generated by the agent.
                    feature_highlights_list.append(highlight_content)
            
        # 3. Generate Project-Level README (The Main Output)
        logger.info("Generating Project-Level README with Feature Highlights...")
        # Pass the structure tree to the agent
        project_readme = await self.docs_agent.generate_project_readme(
            project_name, 
            feature_highlights_list,
            project_structure=project_structure_tree
        )
        
        # 4. Save
        output_filename = f"{project_name}_README.md"
        out_path = Path(output_dir) / output_filename
        
        # In this new "less bulky" mode, the project_readme IS the documentation.
        self._save_to_file(out_path, project_readme)
        logger.info(f"Successfully saved consolidated documentation to {out_path}")
        logger.info("Pipeline execution finished.")

        return {
            "project_name": project_name,
            "output_file": str(out_path),
            "content": project_readme
        }

    def _generate_project_structure_tree(self, files_metadata: List) -> str:
        """
        Generates a string representation of the project structure tree.
        """
        if not files_metadata:
            return ""
            
        # Determine root common path to shorten display
        try:
             # Just use the parent folder of the first file as a rough root
             # or simply structure based on relative paths from the source root
             # Since 'files_metadata' contains absolute paths, we need to relativize them.
             root_path = os.path.commonpath([f.path for f in files_metadata])
        except Exception:
             root_path = ""

        # Build a nested dictionary
        tree = {}
        for meta in files_metadata:
            if root_path:
                 rel_path = os.path.relpath(meta.path, root_path)
            else:
                 rel_path = os.path.basename(meta.path)
            
            parts = Path(rel_path).parts
            current_level = tree
            for part in parts:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

        # Recursive function to draw tree
        def _draw_tree(node, prefix="", is_last=True):
            lines = []
            keys = sorted(node.keys())
            for index, key in enumerate(keys):
                is_last_child = (index == len(keys) - 1)
                connector = "└── " if is_last_child else "├── "
                lines.append(f"{prefix}{connector}{key}")
                
                child_prefix = prefix + ("    " if is_last_child else "│   ")
                if node[key]: # If it has children (it's a directory)
                     lines.extend(_draw_tree(node[key], child_prefix, is_last_child))
            return lines

        tree_lines = _draw_tree(tree)
        # Add root folder name
        root_name = os.path.basename(root_path) if root_path else "Project Root"
        return f"{root_name}/\n" + "\n".join(tree_lines)


    async def _process_single_file(self, file_path: str, sem: asyncio.Semaphore) -> Optional[tuple[str, str]]:
        """
        Returns a tuple of (file_path, feature_highlight_content) or None on failure.
        """
        async with sem:
            try:
                logger.info(f"Processing: {file_path}")
                
                # Step A: Parse (Code Understanding)
                module_info = await self.understanding_agent.run(file_path)
                if not module_info:
                    logger.warning(f"Skipping {file_path} - Parsing failed.")
                    return None

                # Step B: Generate Feature Highlights (NOT detailed docs)
                highlights_content = await self.docs_agent.run(module_info)
                
                return (file_path, highlights_content)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}", exc_info=True)
                return None

    def _save_to_file(self, path: Path, content: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
