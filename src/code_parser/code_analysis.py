#TODO:Implement recursive directory traversal using file I/O libraries.
#TODO:Implement file filtering based on file extensions or patterns.
#TODO:Implement directory exclusion based on a predefined ignore list or `.gitignore` configuration.
#TODO:Store details about each valid file, including path, size, and language.
#Recursive directory traversal: implemented in _scan_directory using os.walk
#File filtering: implemented in _extract_file_metadata using file extension checks
#Directory exclusion: implemented in _should_ignore using fnmatch patterns
#File details storage: implemented in file_details list and _extract_file_metadata

import os
from typing import List, Tuple, Dict, Any
import logging
import json 
import importlib.util as iutil
import fnmatch
from datetime import datetime


# class for analysing code files in a directory
class CodeAnalyzer:
    """
    Code analysis class to analyze code files in a directory.

    Attributes:
    -----------
    - directory: str
        The directory to analyze.
    - file_extensions: List[str]
        The list of file extensions to analyze.
    - ignore_patterns: List[str]
        The list of patterns to ignore.
    """

    DEFAULT_FILE_EXTENSIONS = {".py": "python",
                             ".java": "java",
                             ".c": "c",
                             ".cpp": "cpp",
                             ".js": "javascript",
                             ".ts": "typescript",
                             ".html": "html",
                             ".css": "css",
                             ".scss": "scss",
                             ".json": "json",
                             ".xml": "xml",
                             ".yaml": "yaml",
                             ".yml": "yaml",
                             ".sh": "bash",
                             ".bat": "batch",
                             ".ps1": "powershell",
                             ".sql": "sql",
                             ".php": "php",
                             ".rb": "ruby",
                             ".go": "go",
                             ".rs": "rust",
                             ".kt": "kotlin",
                             ".swift": "swift",
                             ".pl": "perl",
                             ".lua": "lua",
                             ".r": "r",
                             ".m": "matlab",
                             ".ipynb": "jupyter",
                             ".md": "markdown",
                             ".txt": "text",
                             ".log": "log",
                             ".jl": "julia"}
    
    DEFAULT_IGNORE_PATTERNS = [
        ".git", ".vscode", "__pycache__", ".ipynb_checkpoints", "node_modules", "venv", "env", 
        ".idea", ".vs", ".cache", ".pytest_cache", ".mypy_cache", ".tox", ".eggs", ".venv", 
        ".serverless", "dist", "build", "target", "out", "logs", "tmp", "temp", "backup", 
        "cache", "data", "log"
        ]
    

    # initialize the code analyzer
    def __init__(self, root_dir: str, file_extensions: List[str] = None, ignore_patterns: List[str] = None):
        """
        Initialize the code analyzer.

        Parameters:
        -----------
        - root_dir: str
            The root directory to analyze.
        - file_extensions: List[str]
            The list of file extensions to analyze.
        - ignore_patterns: List[str]
            The list of patterns to ignore.
        """
        self.root_dir = root_dir
        self.file_extensions = file_extensions or list(self.DEFAULT_FILE_EXTENSIONS.keys())
        self.ignore_patterns = ignore_patterns or self.DEFAULT_IGNORE_PATTERNS
        self.file_details = []
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.propagate = False

    
    # function to scan the repository
    def scan_repo(self) -> List[Dict[str, Any]]:
        """
        Scan the repository for code files.

        Returns:
        --------
        - List[Dict[str, Any]]
            The list of file details.
        """
        self.logger.info(f"Scanning repository at {self.root_dir}...")
        self.file_details = []
        repo_structure = self._scan_directory(self.root_dir)
        self.logger.info(f"Repository scanned successfully.")
        return self.file_details.append(repo_structure)
    
    
    # function to scan a directory for code files
    def _scan_directory(self, directory: str) -> None:
        """
        Recursively scan the directory for code files and make use of concurrent processing.
        
        Returns:
        --------
        - None
        """

        repository_structure = {
            "root_path": directory,
            "files": [],
            "directories": [],
            "total_files": 0,
            "metadata": []
        }

        for root, dirs, files in os.walk(directory):

            dirs[:] = [d for d in dirs if not self._should_ignore(d)]
            self.logger.info(f"Scanning directory: {root}")

            for file in files:
                full_path = os.path.join(root, file)
                self.logger.info(f"Scanning file: {full_path}")

                if not self._should_ignore(file):
                    file_metadata = self._extract_file_metadata(full_path)
                    if file_metadata:
                        self.logger.info(f"Found code file: {file_metadata['name']}")
                        repository_structure["files"].append(file_metadata["name"])
                        repository_structure["metadata"].append(file_metadata)


            
            relative_path = os.path.relpath(root, directory)
            if relative_path != ".":
                repository_structure["directories"].append(relative_path)

        repository_structure["total_files"] = len(repository_structure["files"])
        
        self.logger.info(f"Total files found: {repository_structure['total_files']}")
        
        return repository_structure
    
    
    # function to check if a path should be ignored
    def _should_ignore(self, path: str) -> bool:
        """
        Check if the path should be ignored based on the ignore patterns.

        Parameters:
        -----------
            path: str
                The path to check.
        
        Returns:
        --------
            bool
                True if the path should be ignored, False otherwise.
        """


        return any(fnmatch.fnmatch(path, pattern) for pattern in self.ignore_patterns)


    # function to extract metadata from a code file
    def _extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a code file.

        Parameters:
        -----------
            file_path: str
                The path to the file.
        
        Returns:
        --------
            Dict[str, Any]
                The file metadata.
        """

        self.logger.info(f"Extracting metadata for file: {file_path}")

        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1]
        if file_ext not in self.file_extensions:
            return None

        file_metadata = {
            "path": os.path.relpath(file_path, self.root_dir),
            "name": file_name,
            "extension": file_ext,
            "size": os.path.getsize(file_path),
            "language": self._detect_language(file_path),
            "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        }

        # Add language-specific parsing for additional metadata
        parser_method = getattr(self, f"_parse_{file_metadata['language']}", None)
        if parser_method:
            file_metadata.update(parser_method(file_path))

        self.logger.info(f"Metadata extracted: {file_metadata}")

        return file_metadata


    # function to detect the programming language of a code file
    def _detect_language(self, file_path: str) -> str:
        """
        Detect the programming language of a code file.

        Parameters:
        -----------
            file_path: str
                The path to the file.
        
        Returns:
        --------
            str
                The detected language.
        """

        return self.DEFAULT_FILE_EXTENSIONS.get(os.path.splitext(file_path)[1], "unknown")
    

    # Function to save the repository metadata to a JSON file
    def save_repo_metadata(self, output_path: str=None):
        """
        Save the repository metadata to a JSON file.

        Parameters:
        -----------
            output_path: str
                The path to save the metadata.
        
        Returns:
        --------
            None
        """

        output_path = output_path or os.path.join(self.root_dir, "repo_metadata.json")

        self.logger.info(f"Saving repository metadata to: {output_path}")

        with open(output_path, "w") as f:
            json.dump(self.file_details, f, indent=4)


def main():
    """
    Example usage of CodeAnalyzer
    """
    analyzer = CodeAnalyzer('/home/testys/Documents/GitHub/Breeze.ai')
    analyzer.scan_repo()
    analyzer.save_repo_metadata()

if __name__ == '__main__':
    main()