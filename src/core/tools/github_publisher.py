import logging
from typing import Dict, Any, Optional
from github import Github, GithubException, InputGitTreeElement

from src.core.tools.base import BaseTool

logger = logging.getLogger(__name__)

class GithubPublisherTool(BaseTool):
    """
    A tool for publishing (creating or updating) files in a GitHub repository.
    """

    def __init__(self):
        super().__init__(
            name="github_publisher",
            description="Publishes content to a GitHub repository by creating or updating a file."
        )

    async def run(self, tool_input: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes the file publication logic.

        Args:
            tool_input: A dictionary containing:
                - 'repo_name': str (e.g., 'owner/repo')
                - 'file_path': str (e.g., 'README.md')
                - 'content': str (The file content)
                - 'access_token': str (GitHub PAT with write access)
                - 'branch': str (Optional, default 'main')
                - 'commit_message': str (Optional)

        Returns:
            Dict containing 'status', 'commit_sha', and 'html_url'.
        """
        repo_name = tool_input.get("repo_name")
        file_path = tool_input.get("file_path")
        content = tool_input.get("content")
        access_token = tool_input.get("access_token")
        branch = tool_input.get("branch", "main")
        commit_message = tool_input.get("commit_message", f"Update {file_path} via Breeze-Docs")

        if not all([repo_name, file_path, content, access_token]):
            raise ValueError("Missing required fields: repo_name, file_path, content, access_token")

        try:
            g = Github(access_token)
            repo = g.get_repo(repo_name)

            # Check if file exists to determine if we update or create
            try:
                contents = repo.get_contents(file_path, ref=branch)
                # If we are here, file exists. Update it.
                logger.info(f"File {file_path} exists. Updating...")
                commit = repo.update_file(
                    path=file_path,
                    message=commit_message,
                    content=content,
                    sha=contents.sha,
                    branch=branch
                )
                action = "updated"
            except GithubException as e:
                if e.status == 404:
                    # File not found, create it
                    logger.info(f"File {file_path} not found. Creating...")
                    commit = repo.create_file(
                        path=file_path,
                        message=commit_message,
                        content=content,
                        branch=branch
                    )
                    action = "created"
                else:
                    raise e

            return {
                "status": "success",
                "action": action,
                "commit_sha": commit['commit'].sha,
                "html_url": commit['content'].html_url if 'content' in commit else None
            }

        except GithubException as e:
            logger.error(f"GitHub API Error: {e}", exc_info=True)
            raise e
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise e
