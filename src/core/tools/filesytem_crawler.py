import os
import logging
import fnmatch
from datetime import datetime
from typing import List, Dict, Any, Optional, Union # Import Union
import asyncio # Import asyncio for example usage
import json # Import json for saving metadata

# Import the BaseTool interface and FileMetadata schema
from src.core.tools.base import BaseTool
from src.core.schemas.models import FileMetadata # Import FileMetadata schema

# Configure logging for this module
logger = logging.getLogger(__name__)


class FileSystemCrawlerTool(BaseTool):
    """
    A tool for recursively crawling a file system directory, filtering files,
    excluding directories, and extracting basic file metadata.

    Adapts the logic from the provided CodeAnalyzer class to fit the BaseTool interface.
    """

    # Default file extensions and ignore patterns as class attributes
    DEFAULT_FILE_EXTENSIONS: Dict[str, str] = {
        ".py": "python", ".java": "java", ".c": "c", ".cpp": "cpp", ".js": "javascript",
        ".ts": "typescript", ".html": "html", ".css": "css", ".scss": "scss", ".json": "json",
        ".xml": "xml", ".yaml": "yaml", ".yml": "yaml", ".sh": "bash", ".bat": "batch",
        ".ps1": "powershell", ".sql": "sql", ".php": "php", ".rb": "ruby", ".go": "go",
        ".rs": "rust", ".kt": "kotlin", ".swift": "swift", ".pl": "perl", ".lua": "lua",
        ".r": "r", ".m": "matlab", ".ipynb": "jupyter", ".md": "markdown", ".txt": "text",
        ".log": "log", ".jl": "julia"
    }

    DEFAULT_IGNORE_PATTERNS: List[str] = [
        ".git", ".vscode", "__pycache__", ".ipynb_checkpoints", "node_modules", "venv",
        "env", ".idea", ".vs", ".cache", ".pytest_cache", ".mypy_cache", ".tox", ".eggs",
        ".venv", ".serverless", "dist", "build", "target", "out", "logs", "tmp", "temp",
        "backup", "cache", "data", "log"
    ]

    def __init__(
        self,
        root_dir: str,
        file_extensions: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None
    ):
        """
        Initializes the FileSystemCrawlerTool.

        Args:
            root_dir: The root directory to start crawling from.
            file_extensions: Optional list of file extensions to include.
                             Defaults to DEFAULT_FILE_EXTENSIONS keys if None.
            ignore_patterns: Optional list of glob patterns to ignore files/directories.
                             Defaults to DEFAULT_IGNORE_PATTERNS if None.
        """
        # Initialize the BaseTool with name and description
        super().__init__(
            name="filesystem_crawler",
            description="Crawls a directory and lists files based on patterns, excluding ignored paths."
        )

        # Store configuration
        self._root_dir = root_dir
        # Use provided extensions or defaults, ensure they are lowercase for consistent comparison
        self._file_extensions = [ext.lower() for ext in (file_extensions or list(self.DEFAULT_FILE_EXTENSIONS.keys()))]
        self._ignore_patterns = ignore_patterns or self.DEFAULT_IGNORE_PATTERNS

        # Validate root directory exists
        if not os.path.isdir(self._root_dir):
            logger.error(f"Root directory does not exist or is not a directory: {self._root_dir}")
            raise ValueError(f"Root directory does not exist or is not a directory: {self._root_dir}")

        logger.info(f"FileSystemCrawlerTool initialized for directory: {self._root_dir}")
        logger.info(f"Including extensions: {self._file_extensions}")
        logger.info(f"Ignoring patterns: {self._ignore_patterns}")


    async def run(self, tool_input: str, *args: Any, **kwargs: Any) -> List[FileMetadata]:
        """
        Executes the file system crawling logic.

        Adapts the scan_repo and _scan_directory logic from the original CodeAnalyzer.

        Args:
            tool_input: The root directory path to crawl. This overrides the
                        directory provided during initialization if specified.
                        (Keeping tool_input as per BaseTool signature, but
                         using self._root_dir if tool_input is None or empty).
            *args: Additional positional arguments (not used by this tool).
            **kwargs: Additional keyword arguments (not used by this tool).

        Returns:
            A list of FileMetadata objects for files that match the criteria.

        Raises:
            ValueError: If the provided tool_input directory does not exist.
            Exception: For other file system errors during traversal.
        """
        # Use tool_input if provided and valid, otherwise use the initialized root_dir
        current_root_dir = tool_input if tool_input and os.path.isdir(tool_input) else self._root_dir

        if not os.path.isdir(current_root_dir):
             logger.error(f"Run input directory does not exist or is not a directory: {current_root_dir}")
             raise ValueError(f"Run input directory does not exist or is not a directory: {current_root_dir}")

        logger.info(f"Running FileSystemCrawlerTool on {current_root_dir}...")
        found_files_metadata: List[FileMetadata] = []
        total_files_scanned = 0

        try:
            # Use os.walk for recursive traversal
            for root, dirs, files in os.walk(current_root_dir):
                # Filter directories in place to skip ignored ones
                dirs[:] = [d for d in dirs if not self._should_ignore(os.path.join(root, d))]
                logger.debug(f"Scanning directory: {root}")

                for file in files:
                    total_files_scanned += 1
                    full_path = os.path.join(root, file)
                    logger.debug(f"Checking file: {full_path}")

                    # Check if the file itself should be ignored
                    if not self._should_ignore(full_path):
                         # Extract metadata if the file extension is included
                        metadata = self._extract_file_metadata(full_path)
                        if metadata:
                            found_files_metadata.append(metadata)
                            logger.debug(f"Included file: {full_path}")
                        else:
                            logger.debug(f"Excluded file by extension: {full_path}")
                    else:
                        logger.debug(f"Excluded file by ignore pattern: {full_path}")

        except Exception as e:
            logger.error(f"Error during file system traversal: {e}", exc_info=True)
            # Depending on requirements, you might raise a custom tool error here
            raise e # Re-raise the exception

        logger.info(f"Finished scanning. Scanned {total_files_scanned} files, found {len(found_files_metadata)} relevant files.")
        return found_files_metadata


    def _should_ignore(self, path: str) -> bool:
        """
        Check if the path should be ignored based on the configured ignore patterns.

        Parameters:
        -----------
            path: str
                The path to check (can be file or directory).

        Returns:
        --------
            bool
                True if the path should be ignored, False otherwise.
        """
        # Check against ignore patterns relative to the root directory
        relative_path = os.path.relpath(path, self._root_dir)
        if relative_path == ".": # Don't ignore the root itself
            return False

        # Check if any ignore pattern matches the relative path or the base name
        # fnmatch is simple, consider more robust .gitignore parsing for production
        return any(fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern) for pattern in self._ignore_patterns)


    def _extract_file_metadata(self, file_path: str) -> Optional[FileMetadata]:
        """
        Extract basic metadata from a file if its extension is included.

        Parameters:
        -----------
            file_path: str
                The path to the file.

        Returns:
        --------
            Optional[FileMetadata]
                The file metadata wrapped in a FileMetadata object, or None if the
                file extension is not in the included list or if there's an error.
        """
        try:
            file_name = os.path.basename(file_path)
            # Get extension and convert to lowercase for case-insensitive comparison
            file_ext = os.path.splitext(file_name)[1].lower()

            if file_ext not in self._file_extensions:
                return None # Exclude files with non-included extensions
            
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            # Get file stats safely
            stat_info = os.stat(file_path)

            metadata = FileMetadata(
                # Store path relative to the root directory provided during init
                path=os.path.relpath(file_path, self._root_dir),
                name=file_name,
                extension=file_ext,
                size=stat_info.st_size, # Size in bytes
                language=self._detect_language(file_ext), # Detect language based on lowercase extension
                last_modified=datetime.fromtimestamp(stat_info.st_mtime), # Last modified timestamp
                raw_content= raw_content
            )

            # Note: Language-specific parsing is NOT done here.
            # That will be the responsibility of separate parsing tools.

            logger.debug(f"Extracted metadata for {file_path}")
            return metadata

        except Exception as e:
            logger.warning(f"Could not extract metadata for file {file_path}: {e}", exc_info=True)
            # Return None or raise a specific error if metadata extraction fails
            return None


    def _detect_language(self, file_extension: str) -> str:
        """
        Detect the programming language based on the file extension.

        Parameters:
        -----------
            file_extension: str
                The file extension (should be lowercase, including the dot).

        Returns:
        --------
            str
                The detected language name, or "unknown" if not in the default map.
        """
        return self.DEFAULT_FILE_EXTENSIONS.get(file_extension, "unknown")
    
    async def save_crawled_metadata(self, output_filepath: str) -> bool:
        """
        Saves the crawled file metadata to a JSON file.
        
        Args:
            output_filepath: Path where to save the JSON file
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Use the existing crawler instance
            logger.info(f"Running crawler tool on {self._root_dir}...")
            file_metadata_list = await self.run(self._root_dir)
            logger.info(f"Crawler finished. Found {len(file_metadata_list)} files.")

            # Ensure output directory exists
            output_dir = os.path.dirname(output_filepath)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # Convert metadata to JSON-serializable format
            serializable_data = [meta.model_dump(mode='json') for meta in file_metadata_list]
            
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=4)
            logger.info(f"Metadata saved to {output_filepath}")
            return True
            
        except IOError as e:
            logger.error(f"Error saving metadata to file: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error while saving metadata: {e}")
            return False


# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Ensure logging is set up if running this file directly
    if not logging.getLogger().handlers:
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
         logging.getLogger().setLevel(logging.DEBUG) # Set level to DEBUG for more output

    # # Create a dummy directory structure for testing
    dummy_root = "/home/testys/Documents/GitHub/breeze_docs/data/samples"
    # os.makedirs(os.path.join(dummy_root, "src", "utils"), exist_ok=True)
    # os.makedirs(os.path.join(dummy_root, ".git"), exist_ok=True) # Ignored directory
    # os.makedirs(os.path.join(dummy_root, "__pycache__"), exist_ok=True) # Ignored directory

    # with open(os.path.join(dummy_root, "main.py"), "w") as f:
    #     f.write("print('hello')")
    # with open(os.path.join(dummy_root, "README.md"), "w") as f:
    #     f.write("# README")
    # with open(os.path.join(dummy_root, "config.yaml"), "w") as f:
    #     f.write("key: value")
    # with open(os.path.join(dummy_root, "src", "utils", "helper.py"), "w") as f:
    #     f.write("def helper(): pass")
    # with open(os.path.join(dummy_root, ".git", "config"), "w") as f: # File in ignored dir
    #     f.write("[core]")
    # with open(os.path.join(dummy_root, "temp.log"), "w") as f: # Ignored file pattern
    #     f.write("log entry")

    # print(f"Created dummy directory structure at {dummy_root}")

    # Initialize the crawler tool
    try:
        # Example 1: Default settings
        crawler_tool_default = FileSystemCrawlerTool(root_dir=dummy_root)
        print("\n--- Running Crawler with Default Settings ---")
        found_files_default = asyncio.run(crawler_tool_default.run(dummy_root)) # Pass root_dir to run
        print(f"\nFound {len(found_files_default)} files:")
        for file_meta in found_files_default:
            print(f"- {file_meta.path} ({file_meta.language})")
        
        # saving filemedata 
        output_filepath = os.path.join(dummy_root, "crawled_metadata.json")
        asyncio.run(crawler_tool_default.save_crawled_metadata(output_filepath))
        print(f"Metadata saved to {output_filepath}")

        # Example 2: Custom extensions and ignore patterns
        custom_extensions = [".py", ".md"]
        custom_ignore = ["temp_*", "*.yaml"]
        crawler_tool_custom = FileSystemCrawlerTool(
            root_dir=dummy_root,
            file_extensions=custom_extensions,
            ignore_patterns=custom_ignore
        )
        print("\n--- Running Crawler with Custom Settings ---")
        found_files_custom = asyncio.run(crawler_tool_custom.run(dummy_root)) # Pass root_dir to run
        print(f"\nFound {len(found_files_custom)} files:")
        for file_meta in found_files_custom:
            print(f"- {file_meta.path} ({file_meta.language})")

    except ValueError as e:
        print(f"\nError initializing or running crawler: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    # finally:
    #     # Clean up the dummy directory structure
    #     print(f"\nCleaning up dummy directory structure at {dummy_root}")
    #     import shutil
    #     if os.path.exists(dummy_root):
    #         shutil.rmtree(dummy_root)
    #         print("Cleanup complete.")
