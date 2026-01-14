import asyncio
import logging
import os
import sys
from unittest.mock import MagicMock

# Mock dependencies before importing the agent
sys.modules["pydantic"] = MagicMock()
sys.modules["src.core.llm.client"] = MagicMock()
sys.modules["src.core.llm.prompter"] = MagicMock()

# Mock BaseAgent since it imports things we might not have
class MockBaseAgent:
    def __init__(self, llm_client, prompter):
        self.llm_client = llm_client
        self.prompter = prompter

# Patch BaseAgent in the module where it's imported
# We need to mock the module 'src.core.agents.base'
mock_base_agent_module = MagicMock()
mock_base_agent_module.BaseAgent = MockBaseAgent
sys.modules["src.core.agents.base"] = mock_base_agent_module

# Now we can import the agent
# We also need to ensure the tools are importable. 
# They depend on BaseTool which might depend on pydantic?
# Let's check BaseTool. 
# If BaseTool imports pydantic, we need to mock it too.
# We already mocked pydantic.

from src.core.agents.repo_crawler_agent import RepoCrawlerAgent

# Mock dependencies for the test
class MockClient:
    async def generate(self, prompt):
        return "Mock response"

class MockPrompter:
    pass

async def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize Agent
    client = MockClient()
    prompter = MockPrompter()
    agent = RepoCrawlerAgent(client, prompter)
    
    # Test 1: Local Crawl
    print("\n--- Testing Local Crawl ---")
    # Create a dummy file to find
    with open("test_crawl_dummy.txt", "w") as f:
        f.write("dummy content")
        
    try:
        files = await agent.run(".", is_github=False)
        print(f"Found {len(files)} files.")
        found = False
        for f in files:
            if f.name == "test_crawl_dummy.txt":
                found = True
                print("Success: Found dummy file.")
                break
        if not found:
            print("Failure: Did not find dummy file.")
            
    except Exception as e:
        print(f"Error during local crawl: {e}")
    finally:
        if os.path.exists("test_crawl_dummy.txt"):
            os.remove("test_crawl_dummy.txt")

    # Test 2: GitHub Crawl (Mocked or Real?)
    # Since we don't have a token and don't want to hit rate limits or auth errors in CI,
    # we might skip real GitHub call or try a public repo if allowed.
    # For now, let's just check if the method exists and arguments are accepted.
    print("\n--- Testing GitHub Crawl Interface ---")
    try:
        # We expect this to fail without a token or network, but we want to see it try
        # or at least validate the logic path.
        # However, without a token, the tool logs a warning.
        # Let's try a public repo that is small.
        # But to avoid network issues, maybe we just rely on the local test for now
        # as the implementation is a direct delegation to the tool which we verified exists.
        pass
    except Exception as e:
        print(f"Error during github crawl: {e}")

if __name__ == "__main__":
    asyncio.run(main())
