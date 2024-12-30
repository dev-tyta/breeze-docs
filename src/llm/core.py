import os
import asyncio
from typing import Optional, List, Union, Type, Any
from anthropic import Anthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models.llms import BaseLLM
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import sys


parent_dir = "/home/testys/Documents/GitHub/breeze_docs"
sys.path.append(str(parent_dir))


from src.llm.config import LLMConfig
from src.llm.exceptions import LLMError, ConfigurationError, APIError, ParseError
from src.llm.utils import retry_with_backoff, validate_api_key, logger

load_dotenv()


class BreeLLM(BaseLLM, BaseModel):
    """
    Enhanced LLM implementation with improved structure and error handling.
    
    Attributes:
        config (LLMConfig): Configuration settings for the LLM
        client (Anthropic): Anthropic client instance
        output_parser (Optional[PydanticOutputParser]): Parser for structured output
    """
    
    config: Optional[LLMConfig] = Field(default_factory=LLMConfig)
    query: str
    input_prompt: str
    output_parser: Optional[PydanticOutputParser] = None
    client: Optional[Anthropic] = None
    prompt: Optional[PromptTemplate] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        input_prompt: str,
        query: str,
        output_struct: Optional[Type[BaseModel]] = None,
        config: Optional[LLMConfig] = None,
        **kwargs
    ):
        """
        Initialize the LLM with the given configuration.
        
        Args:
            input_prompt: Template for the prompt
            query: The input query
            output_struct: Optional Pydantic model for output parsing
            config: Optional configuration override
        """
        super().__init__(
            input_prompt=input_prompt,
            query=query,
            config=config,
            **kwargs
        )
        
        # Setup output parsing if needed
        if output_struct is not None:
            if not issubclass(output_struct, BaseModel):
                raise ConfigurationError("output_struct must be a Pydantic model")
            self.output_parser = PydanticOutputParser(pydantic_object=output_struct)
        
        # Initialize prompt template
        self._setup_prompt_template()
        
        # Initialize client
        self._setup_client()
        
    def _setup_prompt_template(self):
        """Setup the prompt template with format instructions if needed"""
        format_instructions = ""
        if self.output_parser:
            format_instructions = self.output_parser.get_format_instructions()
            
        self.prompt = PromptTemplate(
            template=self.input_prompt,
            input_variables=["message"],
            partial_variables={"format_instructions": format_instructions}
        )
        
    def _setup_client(self):
        """Setup the Anthropic client with validation"""
        api_key = validate_api_key(os.getenv(self.config.api_key_env_var))
        self.client = Anthropic(api_key=api_key)
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for the LLM type"""
        return "anthropic"
    
    @retry_with_backoff(max_retries=3)
    async def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for the given prompts with retry logic.
        
        Args:
            prompts: List of prompts to process
            stop: Optional stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments
            
        Returns:
            List of generated responses
            
        Raises:
            APIError: If API call fails after retries
            ParseError: If output parsing fails
        """
        try:
            formatted_prompt = self.prompt.format(message=prompts[0])
            response = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": formatted_prompt}]
            )
            
            output = response.content[0].text
            
            # Parse output if parser is configured
            if self.output_parser:
                try:
                    output = self.output_parser.parse(output)
                except Exception as e:
                    raise ParseError(f"Failed to parse output: {str(e)}")
                    
            return [output]
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise APIError(f"Failed to generate response: {str(e)}")
            
    async def generate_response(self, query: Optional[str] = None) -> str:
        """
        Convenience method for generating a single response.
        
        Args:
            query: Optional query override
            
        Returns:
            Generated response
        """
        query = query or self.query
        responses = await self._generate([query])
        return responses[0]
    


# Usage Example
class SonnetOutput(BaseModel):
    title: str
    lines: List[str]

# Initialize LLM
llm = BreeLLM(
    input_prompt="Generate a sonnet from the following text: {message}",
    query="The quick brown fox jumps over the lazy dog",
    output_struct=SonnetOutput,  # Optional,
    config=LLMConfig(model_name="claude-3-sonnet-20240229", max_tokens=512, temperature=0.7, api_key_env_var="CLAUDE_API_KEY", timeout=30, retry_attempts=3, retry_wait=1.0)
)

# Generate response
async def main():
    response = await llm.generate_response()
    print(response)

asyncio.run(main())