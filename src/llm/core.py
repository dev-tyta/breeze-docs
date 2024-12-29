#TODO: Setup langchain connection to the LLM model: Should be able to send response and receive response from any part of the codebase.
#TODO: Make sure the LLM receives prompts structure and output structure in Pydantic Format (Refer to the Langchain Documentation for help)

import os
from langchain_anthropic import ChatAnthropic
from langchain_openai.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
# from langchain_core.language_models.llms import BaseLLM
from dotenv import load_dotenv
import logging
from pydantic import BaseModel
from pydantic import BaseModel


load_dotenv()


class BreeLLM:
    """
    Main Class for Interacting with the Bree Language Model.

    This class sets up a connection to the Bree Language Model (LLM) using the Langchain framework.
    
    Parameters:
        
    """
    def __init__(self, input_prompt, query, output_struct, tools):
        """
        Initialize the BreeLLM class.

        Parameters:
        input_prompt (str): The prompt template to be used for the LLM.
        self.output = PydanticOutputParser(model=output_struct)
        output_struct (Type[PydanticModel]): The expected output structure in Pydantic format.
        tools (Any): Additional tools or configurations for the LLM.
        """
        logging.basicConfig(level=logging.INFO)
        logging.info("BreeLLM Initialized")
        self.llm_type: str = "openai"
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
        logging.info(f"CLAUDE API Keys Loaded: {self.claude_api_key}")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        logging.info(f"OPENAI API Keys Loaded: {self.openai_api_key}")
        self.chain = None

    def _call_llm(self):
        logging.info("Setting up the LLM connection")
        # self.llm = ChatAnthropic(name="claude-3-5-sonnet-20240620", anthropic_api_key=self.claude_api_key)
        self.llm = OpenAI(api_key=self.openai_api_key)
        logging.info("LLM connection setup")
        self.chain = self.llm(self.prompt.format(query=self.query))
        logging.info("Langchain Connection Setup")

    def _prompt_llm(self):
        if not self.chain:
            self._call_llm()
        logging.info("Invoking the LLM")
        output = self.chain.invoke({"message": self.prompt.format(query=self.query)})
        logging.info("LLM invoked")
        return output
    

# Usage Example
class OutputModel(BaseModel):
    result: str

# Usage Example
# Define the input prompt, query, and expected output structure
llm = BreeLLM(input_prompt="Generate a sonnet from the following text: {query}", query="The quick brown fox jumps over the lazy dog", output_struct=OutputModel, tools=None)
output = llm._prompt_llm()
print(output)