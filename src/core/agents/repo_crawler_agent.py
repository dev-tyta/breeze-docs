import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from src.core.agents.base import BaseAgent
from src.core.llm.client import GeminiClient
from src.core.llm.prompter import Prompter
from src.core.tools.filesystem_crawler import FileSystemCrawlerTool
from src.core.tools.github_repo_downloader import GithubRepoDownloaderTool
from src.core.schemas.models import FileMetadata

logger = logging.getLogger(__name__)

class RepoCrawlerAgent(BaseAgent):
    """
    Agent responsible for crawling repositories (local or remote) to discover files.
    """

    def __init__(self, llm_client: GeminiClient, prompter: Prompter):
        """
        Initializes the RepoCrawlerAgent.
        
        Args:
            llm_client: The LLM client instance.
            prompter: The prompter instance.
        """
        super().__init__(llm_client, prompter)
        self.fs_crawler = FileSystemCrawlerTool(root_dir=".") # Default root, will be overridden in run
        self.github_downloader = GithubRepoDownloaderTool()

    async def run(self, source_path: str, is_github: bool = False, github_token: Optional[str] = None) -> List[FileMetadata]:
        """
        Crawls the specified source path.

        Args:
            source_path: The path to the local directory or the GitHub repo name (owner/repo).
            is_github: Boolean flag to indicate if the source is a GitHub repository.
            github_token: Optional GitHub access token for private repositories or higher rate limits.

        Returns:
            A list of FileMetadata objects representing the discovered files.
        """
        logger.info(f"RepoCrawlerAgent starting crawl on: {source_path} (GitHub: {is_github})")
        
        try:
            if is_github:
                if not github_token:
                    logger.warning("No GitHub token provided. Rate limits may be restricted.")
                
                tool_input = {
                    "repo_name": source_path,
                    "access_token": github_token or "" 
                }
                # Note: GithubRepoDownloader currently returns List[Dict], we might need to adapt it to List[FileMetadata]
                # For now, let's assume we return the raw dicts or convert them.
                # The TODO implies this agent orchestrates tools.
                
                raw_files = await self.github_downloader.run(tool_input)
                
                # Convert to FileMetadata
                file_metadata_list = []
                for f in raw_files:
                    # Simple conversion, might need more robust logic
                    meta = FileMetadata(
                        path=f["path"],
                        name=f["path"].split("/")[-1],
                        extension="." + f["path"].split(".")[-1] if "." in f["path"] else "",
                        size=len(f["content"]),
                        language="unknown", # TODO: Detect language
                        last_modified=datetime.now(), # Default to now if not available
                        raw_content=f["content"]
                    )
                    file_metadata_list.append(meta)
                
                return file_metadata_list

            else:
                # Local file system crawl
                return await self.fs_crawler.run(source_path)

        except Exception as e:
            logger.error(f"RepoCrawlerAgent failed to crawl {source_path}: {e}", exc_info=True)
            raise e
