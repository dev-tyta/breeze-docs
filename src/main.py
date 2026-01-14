import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to sys.path to allow absolute imports from 'src'
# This handles the case where the script is run directly (e.g. python src/main.py)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.core.orchestrator import Orchestrator

async def main():
    """
    Main entry point for the breeze-docs CLI.
    """
    parser = argparse.ArgumentParser(description="Breeze-Docs: Automated Documentation Generator")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: generate
    generate_parser = subparsers.add_parser("generate", help="Generate documentation for a file or directory")
    generate_parser.add_argument("path", type=str, help="Path to the file or directory to document")
    generate_parser.add_argument("--recursive", "-r", action="store_true", help="Recursively document directories")
    generate_parser.add_argument("--output", "-o", type=str, default="docs", help="Output directory for documentation")

    args = parser.parse_args()

    if args.command == "generate":
        target_path = args.path
        
        logger.info(f"Starting documentation generation for: {target_path}")
        
        try:
             orchestrator = Orchestrator()
             await orchestrator.run(target_path, output_dir=args.output, recursive=args.recursive)
        except Exception as e:
             logger.error(f"Execution failed: {e}", exc_info=True)
             sys.exit(1)
             
        logger.info(f"Done.")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
             # Loop policy for windows to avoid some asyncio issues with subprocesses if used
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
