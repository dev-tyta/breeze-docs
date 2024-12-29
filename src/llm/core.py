#TODO: Setup langchain connection to the LLM model: Should be able to send response and receive response from any part of the codebase.
#TODO: Make sure the LLM receives prompts structure and output structure in Pydantic Format (Refer to the Langchain Documentation for help)

import os
from langchain_anthropic.llms import AnthropicLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models.llms import BaseLLM
from dotenv import load_dotenv


load_dotenv()


class BreeLLM(BaseLLM):
    """
    Main Class for Interacting with the Bree Language Model.

    This class sets up a connection to the Bree Language Model (LLM) using the Langchain framework.
    
    Attr:
        
    """
    def __init__(self, input_prompt, query, output_struct, tools):
        self.llm_type: str = "claude"
        self.query = query
        self.input_prompt = input_prompt
        self.output = PydanticOutputParser(output_struct)
        self.prompt = PromptTemplate(
            template=self.input_prompt,
            input_variables=["query"],
            output_parser=self.output
        )
        self.tools = tools
        self.claude_api_key = os.getenv("CLAUDE_API_KEY")
        self.chain = None

    def _call_llm(self):
        self.llm = AnthropicLLM(name="claude-3-5-sonnet-20240620", anthropic_api_key=self.claude_api_key)
        self.chain = self.prompt | self.llm | self.output

    def _prompt_llm(self):
        if self.chain is None:
            self._call_llm()
        output = self.chain.invoke({"message": self.query})
        return output