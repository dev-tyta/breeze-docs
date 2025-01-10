import os
import asyncio
from typing import Optional, List, Union, Type, Any
from langchain_openai import OpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models.llms import BaseLLM
from dotenv import load_dotenv
from pydantic import BaseModel, Field


from src.llm.config import LLMConfig
from src.llm.exceptions import LLMError, ConfigurationError, APIError, ParseError
from src.llm.utils import retry_with_backoff, validate_api_key, logger

load_dotenv()


class BreeLLM(BaseLLM):
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
    output_parser: Optional[Type[BaseModel]] = None
    prompt: Optional[PromptTemplate] = None
    llm: Optional[GoogleGenerativeAI] = None

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
            self.output_parser = output_struct
        
        logger.info(f"Output parser configured: {self.output_parser}")

        # Initialize prompt template
        self._setup_prompt_template()
        
        # Initialize client
        self._setup_client()
        logger.info(f"Initialized {self._llm_type} LLM with model: {self.config.model_name}")

        
    def _setup_prompt_template(self):
        """Setup the prompt template with format instructions if needed"""
        format_instructions = ""
        if self.output_parser:
            format_instructions = self.output_parser
            
        self.prompt = PromptTemplate(
            template=self.input_prompt,
            input_variables=["message"],
            partial_variables={"format_instructions": format_instructions}
        )
        
    def _setup_client(self):
        """Setup the OpenAI client with validation"""
        api_key = validate_api_key(os.getenv(self.config.api_key_env_var))
        logger.info(f"Using API key: {api_key}")
        self.llm = GoogleGenerativeAI(api_key=api_key,
                                      model=self.config.model_name,
                                      max_tokens=self.config.max_tokens,
                                      temperature=self.config.temperature,
                                      timeout=self.config.timeout)
        # self.llm = self.llm.with_structured_output(self.output_parser)
        
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for the LLM type"""
        return "gemini"
    
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
            logger.info(f"Generating response for prompt: {formatted_prompt}")
            # self.llm = self.llm.with_structured_output(self.output_parser)

            # Using Chain Method to generate response
            response = self.llm.invoke(formatted_prompt)
            logger.info(f"Response generated: {response}")

            output = response
            
            # Parse output if parser is configured
            if self.output_parser:
                try:
                    output = self.output_parser(output)
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
        logger.info(f"Generating response for query: {query}")
        responses = await self._generate([query])
        logger.info(f"Response generated: {responses[0]}")
        return responses[0]
    


# Usage Example
class SonnetResponse(BaseModel):
    """Pydantic model for sonnet response"""
    sonnet: str


# Initialize LLM
llm = BreeLLM(
    input_prompt="Generate a sonnet from the following text: {message}",
    query="The quick brown fox jumps over the lazy dog",
    output_struct=SonnetResponse,  # Optional,
    config=LLMConfig(model_name="gemini-1.5-flash", max_tokens=512, temperature=0.7, api_key_env_var="GOOGLE_API_KEY", timeout=30, retry_attempts=3, retry_wait=1.0)
)

# Generate response
async def main():
    response = await llm.generate_response()
    print(response)

asyncio.run(main())