#TODO: Add tests for code_scanner.py
# Added tests for code_scanner.py

import sys
from pathlib import Path
import logging


# Add parent directory to system path to resolve imports
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from src.code_parser.code_scanner import CodeAnalyzer

def main():
    """
    Example usage of CodeAnalyzer
    """
    try:
        # Use Path for OS-independent paths
        repo_path = Path.home() / "Documents/GitHub/breeze_docs/data/samples/ANNA-AI"
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository path not found: {repo_path}")

        analyzer = CodeAnalyzer(str(repo_path))
        analyzer.scan_repo()
        analyzer.save_repo_metadata()
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        logging.error(f"Error during analysis: {str(e)}")

if __name__ == '__main__':
    main()