#TODO: Setup langchain connection to the LLM model: Should be able to send response and receive response from any part of the codebase.
#TODO: Make sure the LLM receives prompts structure and output structure in Pydantic Format (Refer to the Langchain Documentation for help)

import os
from langchain_anthropic.llms import AnthropicLLM
from langchain_core.language_models.llms import BaseLLM
from dotenv import load_dotenv


load_dotenv()


class BreeLLM(BaseLLM):
    """
    Main Class for Interacting witth the Bree Language Model for
    """

    def __init__(self, input_prompt, output_prompt, tools):
        self.llm_type: str = "claude"
        self.prompt = input_prompt
        self.output = output_prompt
        self.tools = tools
        self.claude_api_key = os.getenv("CLAUDE_API_KEY")

    
    def _call_llm(self):
        self.llm = AnthropicLLM(name="claude-3-5-sonnet-20240620", anthropic_api_key=self.claude_api_key)
        self.chain = self.prompt | self.llm | self.output


    def _prompt_llm(self, query):
        output = self.chain.invoke({"message": query})
        print(output)