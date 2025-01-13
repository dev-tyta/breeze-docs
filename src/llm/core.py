import os
import asyncio
from typing import Optional, List, Union, Type, Any, Dict
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import BaseLLM
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent


from src.llm.config import LLMConfig
from src.llm.exceptions import ConfigurationError, APIError, ParseError
from src.llm.utils import retry_with_backoff, validate_api_key, logger
from src.llm.g_col import LLMGarbageCollector

load_dotenv()


class BreeLLM(BaseLLM):
    """
    Enhanced LLM implementation with improved structure and error handling.
    
    Attributes:
        config (LLMConfig): Configuration settings for the LLM
        client (Anthropic): Anthropic client instance
        output_parser (Optional[PydanticOutputParser]): Parser for structured output
    """
    
    _gc = LLMGarbageCollector()
    config: Optional[LLMConfig] = Field(default_factory=LLMConfig)
    query: str
    input_prompt: str
    output_parser: Optional[Type[BaseModel]] = None
    prompt: Optional[PromptTemplate] = None
    llm: Optional[GoogleGenerativeAI] = None
    agent: Optional[Any] = None

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
            # self.output_parser = PydanticOutputParser(pydantic_object=output_struct)
            self.output_parser = output_struct

        # Register resource for garbage collection
        self._gc.register_resource(self)
        # Initialize prompt template
        self._setup_prompt_template()
        
        # Initialize agent
        self._setup_agent()

        # Initialize client
        self._setup_client()
        logger.info(f"Initialized {self._llm_type} LLM with model: {self.config.model_name}")

        
    def _setup_prompt_template(self):
        """Setup the prompt template with format instructions if needed"""
        self.prompt = PromptTemplate(
            template=self.input_prompt,
            input_variables=["message"],
        )
        logger.info(f"Prompt template: {self.prompt}")
        
    def _setup_client(self):
        """Setup the OpenAI client with validation"""
        api_key = validate_api_key(os.getenv(self.config.api_key_env_var))
        self.llm = GoogleGenerativeAI(api_key=api_key,
                                      model=self.config.model_name,
                                      max_tokens=self.config.max_tokens,
                                      temperature=self.config.temperature,
                                      timeout=self.config.timeout)


    def _setup_agent(self):
        """Setup the Pydantic AI agent for structured output parsing"""
        self.agent = Agent(
            model='gemini-1.5-flash',
            system_prompt="You are a assistant tool mainly for parsing output structures. You are given a structured output and you need to parse it into a Pydantic model.",
            result_type=self.output_parser)
        
    
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
            logger.info(f"Formatted prompt: {formatted_prompt}")
            with self._gc.track_resource(self.llm):
                response = self.llm.invoke(f"""{formatted_prompt}""")
                logger.info(f"Response: {response}")

            with self._gc.track_resource(self.agent):
                response = await self.parse_output_structure(response)
                logger.info(f"Parsed response: {response}")

            response = self.struct_to_dict([response.data], self.output_parser)

            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            self._gc.force_cleanup()
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

        return responses


    async def parse_output_structure(self, response:str) -> BaseModel:
        """
        Parse the output structure of the response.
        
        Args:
            response: The generated response
            
        Returns:
            Parsed output structure
            
        Raises:
            ParseError: If output parsing fails
        """
        if self.output_parser is None:
            raise ParseError("No output parser configured")
        
        try:
            parsed_output = await self.agent.run(response)
            return parsed_output
        except Exception as e:
            logger.error(f"Error parsing output: {str(e)}")
            raise ParseError(f"Failed to parse output: {str(e)}")
    
    def struct_to_dict(self, response:List,  structure: BaseModel) -> Dict[str, Any]:
        """
        Convert a structured output to a dictionary.
        
        Args:
            response: The generated structured response
            struct: The structured output
            
        Returns:
            Dictionary representation of the output
        """
        dict_output = {}
        list_of_keys = list(structure.model_fields.keys())

        for i in list_of_keys:
            output = response[0].__getattribute__(i)
            dict_output.update({i: output})

        return dict_output
    
    def __del__(self):
        """Cleanup when instance is deleted"""
        self._gc.force_cleanup()

# # Usage Example
# class SonnetResponse(BaseModel):
#     """Pydantic model for sonnet response"""
#     title: str
#     sonnet: str


# # Initialize LLM
# llm = BreeLLM(
#     input_prompt="Generate a title and sonnet from the following text: {message}",
#     query="The quick brown fox jumps over the lazy dog",
#     output_struct=SonnetResponse,  # Optional,
#     config=LLMConfig(model_name="gemini-1.5-flash", max_tokens=512, temperature=0.7, api_key_env_var="GEMINI_API_KEY", timeout=30, retry_attempts=3, retry_wait=1.0)
# )

# # # Generate response
# async def main():
#     response = await llm.generate_response()
#     print(response)


# asyncio.run(main())