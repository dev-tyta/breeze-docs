import logging
from typing import List, Dict, Any, Optional, Union
from github import Github, GithubException # Import PyGithub components
import base64 # Needed to decode file content


# Import the BaseTool interface
from src.core.tools.base import BaseTool
from src.config.settings import get_settings


# Configure logging for this module
logger = logging.getLogger(__name__)
# The basic logging setup block is removed here.
# Logging should be configured centrally by calling setup_logging() from src/utils/logging.py
# in the application's entry point (e.g., main.py).


class GithubRepoDownloaderTool(BaseTool):
    """
    A tool for downloading files from a GitHub repository using a personal access token.

    Takes a repository name and access token as input and returns a list of
    file paths and their contents.
    """

    def __init__(self):
        """
        Initializes the GithubRepoDownloaderTool.
        """
        # Initialize the BaseTool with name and description
        super().__init__(
            name="github_repo_downloader",
            description="Downloads files from a specified GitHub repository using an access token."
        )
        logger.info("GithubRepoDownloaderTool initialized.")

    async def run(self, tool_input: Dict[str, str], *args: Any, **kwargs: Any) -> List[Dict[str, str]]:
        """
        Executes the GitHub repository downloading logic.

        Args:
            tool_input: A dictionary containing:
                        - 'repo_name': str, the name of the repository (e.g., 'owner/repo').
                        - 'access_token': str, the GitHub personal access token.
            *args: Additional positional arguments (not used by this tool).
            **kwargs: Additional keyword arguments (not used by this tool).

        Returns:
            A list of dictionaries, where each dictionary contains 'path' (str)
            and 'content' (str) for each file in the repository.

        Raises:
            ValueError: If 'repo_name' or 'access_token' are missing in tool_input.
            GithubException: For errors interacting with the GitHub API.
            Exception: For other unexpected errors.
        """
        repo_name = tool_input.get("repo_name")
        access_token = tool_input.get("access_token")

        if not repo_name or not access_token:
            logger.error("Missing 'repo_name' or 'access_token' in tool_input.")
            raise ValueError("tool_input must contain 'repo_name' and 'access_token'.")

        logger.info(f"Running GithubRepoDownloaderTool for repository: {repo_name}")
        downloaded_files: List[Dict[str, str]] = []

        try:
            # Authenticate with GitHub
            g = Github(access_token)
            logger.debug("Authenticated with GitHub.")

            # Get the repository
            repo = g.get_repo(repo_name)
            logger.debug(f"Accessed repository: {repo_name}")

            # Recursively fetch and process repository contents
            self._fetch_repo_contents(repo, "", downloaded_files)

            logger.info(f"Finished downloading. Found {len(downloaded_files)} files in {repo_name}.")
            return downloaded_files

        except GithubException as e:
            logger.error(f"GitHub API error while downloading repository {repo_name}: {e}", exc_info=True)
            # Wrap GitHub exceptions in a custom tool error if desired, or re-raise
            raise e # Re-raise the specific GitHub exception
        except Exception as e:
            logger.error(f"An unexpected error occurred while downloading repository {repo_name}: {e}", exc_info=True)
            raise e # Re-raise other exceptions


    def _fetch_repo_contents(self, repo: Any, path: str, downloaded_files: List[Dict[str, str]]) -> None:
        """
        Recursively fetches contents of a GitHub repository path.

        Args:
            repo: The PyGithub Repository object.
            path: The current path within the repository (e.g., "", "src/").
            downloaded_files: The list to append file details to.
        """
        try:
            # Get contents of the current path
            contents = repo.get_contents(path)

            # Iterate through contents
            for content_file in contents:
                if content_file.type == "dir":
                    # If it's a directory, recurse into it
                    logger.debug(f"Entering directory: {content_file.path}")
                    self._fetch_repo_contents(repo, content_file.path, downloaded_files)
                else:
                    # If it's a file, fetch its content
                    logger.debug(f"Fetching file content: {content_file.path}")
                    try:
                        # Get the file content and decode it
                        file_content = content_file.decoded_content.decode('utf-8')
                        downloaded_files.append({
                            "path": content_file.path,
                            "content": file_content
                        })
                        logger.debug(f"Fetched content for {content_file.path}")
                    except Exception as e:
                        logger.warning(f"Could not fetch or decode content for file {content_file.path}: {e}")
                        # Optionally append with None content or skip


        except GithubException as e:
            # Handle specific GitHub exceptions during traversal (e.g., empty directory)
            if e.status == 404 and "This is a directory" in str(e):
                 logger.debug(f"Path {path} is not a directory or is empty.")
            else:
                logger.warning(f"GitHub API error while fetching contents for path {path}: {e}")
                # Depending on error, you might want to stop or continue
        except Exception as e:
            logger.warning(f"An unexpected error occurred while fetching contents for path {path}: {e}")


# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Ensure logging is set up if running this file directly
    if not logging.getLogger().handlers:
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
         logging.getLogger().setLevel(logging.DEBUG) # Set level to DEBUG for more output

    # --- IMPORTANT ---
    # Replace with your actual GitHub repo and token for testing!
    # Do NOT commit your actual token to source control. Use environment variables.
    app_settings = get_settings()
    llm_settings = app_settings.llm
    TEST_REPO_NAME =  "dev-tyta/thery.ai"# e.g., "octocat/Spoon-Knife" for a public repo
    TEST_ACCESS_TOKEN = llm_settings.github_access_token.get_secret_value() # Needs 'repo' or 'public_repo' scope
    # You can generate a PAT in GitHub Settings -> Developer settings -> Personal access tokens

    async def run_github_downloader_example():
        if TEST_REPO_NAME == "" or TEST_ACCESS_TOKEN == "your_github_personal_access_token":
            print("Please replace TEST_REPO_NAME and TEST_ACCESS_TOKEN with your actual GitHub details for the example.")
            print("Note: Using a public repository like 'octocat/Spoon-Knife' is recommended for initial testing without needing a PAT.")
            return

        print(f"Running GitHubRepoDownloaderTool for {TEST_REPO_NAME}...")
        downloader_tool = GithubRepoDownloaderTool()

        try:
            # Prepare the tool input
            tool_input = {
                "repo_name": TEST_REPO_NAME,
                "access_token": TEST_ACCESS_TOKEN
            }

            # Run the tool
            downloaded_files = await downloader_tool.run(tool_input)

            print(f"\nSuccessfully downloaded details for {len(downloaded_files)} files.")
            print("First 5 files:")
            for i, file_info in enumerate(downloaded_files[:5]):
                print(f"- Path: {file_info['path']}")
                print(f"  Content (first 100 chars): {file_info['content'][:100]}...")
                if i == 4: break

        except ValueError as e:
            print(f"\nError running downloader: {e}")
        except GithubException as e:
            print(f"\nGitHub API Error: {e.status} - {e.data}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

    # Run the async example function
    import asyncio
    asyncio.run(run_github_downloader_example())
