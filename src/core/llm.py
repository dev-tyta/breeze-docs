# src/core/llm.py
import os
import asyncio
from typing import Optional, List, Union, Type, Any, Dict, TypeVar, Generic, cast
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.llms import BaseLLM
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from src.config.config import LLMConfig
from src.exceptions import ConfigurationError, APIError, ParseError
from src.utils.utils import retry_with_backoff, validate_api_key, logger
from src.g_col import LLMGarbageCollector

load_dotenv()

T = TypeVar('T', bound=BaseModel)


class BreeLLM(BaseLLM, Generic[T]):
    """
    Enhanced LLM implementation using Langchain Agents for agentic workflows.
    """

    _gc = LLMGarbageCollector()
    config: LLMConfig = Field(default_factory=LLMConfig)
    system_prompt: str
    user_query: str
    output_parser: Optional[PydanticOutputParser] = None
    prompt_template: Optional[PromptTemplate] = None
    llm: Optional[GoogleGenerativeAI] = None
    agent_executor: Optional[AgentExecutor] = None
    state: Dict[str, Any] = Field(default_factory=dict)
    tools: List[BaseTool] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        system_prompt: str,
        user_query: str,
        output_struct: Optional[Type[T]] = None,
        config: Optional[LLMConfig] = None,
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ):
        """
        Initialize the LLM with proper configuration.
        
        Args:
            system_prompt: The system instructions for the LLM
            user_query: The initial user query to process
            output_struct: Optional Pydantic model for structured output parsing
            config: Configuration for the LLM
            tools: List of tools available to the agent
        """
        # Initialize with proper parameters
        super().__init__(
            system_prompt=system_prompt,
            user_query=user_query,
            config=config or LLMConfig(),
            **kwargs
        )

        # Setup output parsing if specified
        if output_struct is not None:
            if not issubclass(output_struct, BaseModel):
                raise ConfigurationError("output_struct must be a Pydantic model")
            self.output_parser = PydanticOutputParser(pydantic_object=output_struct)

        # Set up tools
        self.tools = tools or []
        
        # Register with garbage collector
        self._gc.register_resource(self)
        
        # Initialize components
        self._setup_prompt_template()
        self._setup_client()
        self._setup_agent()
        
        logger.info(f"Initialized {self._llm_type} LLM with model: {self.config.model_name}")

    def _setup_prompt_template(self) -> None:
        """Setup the prompt template with format instructions if needed"""
        format_instructions = self.output_parser.get_format_instructions() if self.output_parser else ""

        escaped_format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
    
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template=f"{self.system_prompt}\n\n{{query}}\n\n{escaped_format_instructions}"
        )
        
        logger.debug(f"Prompt template initialized: {self.prompt_template.template[:100]}...")

    def _setup_client(self) -> None:
        """Setup the Google GenerativeAI client with validation"""
        try:
            api_key = validate_api_key(os.getenv(self.config.api_key_env_var))
            self.llm = GoogleGenerativeAI(
                api_key=api_key,
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )
            logger.info(f"Google GenerativeAI client initialized with model {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Google GenerativeAI client: {str(e)}")
            raise ConfigurationError(f"LLM client initialization failed: {str(e)}")

    def _setup_agent(self) -> None:
        """Setup the Langchain agent with proper tools and prompting."""
        if not self.llm:
            raise ConfigurationError("LLM client must be initialized before setting up agent")

        # Define the prompt for the agent with correct variable names
        tools_section = """Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question""" if self.tools else """Answer the following questions as best you can.

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Final Answer: the final answer to the original input question"""

        system_message = SystemMessage(content=tools_section)
        human_message = HumanMessage(content="{input}")
        
        # Create the prompt properly for ReAct agent
        react_prompt = ChatPromptTemplate.from_messages([
            system_message,
            human_message,
            # MessagesPlaceholder(variable_name="agent_scratchpad")  # This is the key fix - correct format for agent_scratchpad
        ])

        try:
            # Create the agent with the correct prompt and tools
            self.agent_executor = create_react_agent(
                model=self.llm,
                tools=self.tools,
                checkpointer= MemorySaver()
            )

            
            logger.info(f"Agent initialized with {len(self.tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise ConfigurationError(f"Agent initialization failed: {str(e)}")

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
        Generate responses using the Langchain agent framework with retry logic.
        
        Args:
            prompts: List of prompt strings (only first one is used)
            stop: Optional stop sequences
            run_manager: Optional callback manager
            
        Returns:
            List containing the string response
        """
        if not self.agent_executor:
            raise ConfigurationError("Agent not initialized")
            
        try:
            input_query = prompts[0]
            logger.info(f"Processing query: {input_query[:100]}...")

            with self._gc.track_resource(self.agent_executor):
                response = await self.agent_executor.ainvoke(
                    {"input": input_query}
                )
                
                # Extract the output text from the response
                output_text = response.get('output', '')
                logger.info(f"Generated response: {output_text[:100]}...")
                
                return [output_text]

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            self._gc.cleanup(self.agent_executor)
            raise APIError(f"Failed to generate response: {str(e)}")

    async def generate_response(self, query: Optional[str] = None) -> Union[T, str]:
        """
        Generate a single response, optionally parsing to the output structure.
        
        Args:
            query: Optional query override, uses the initial query if not provided
            
        Returns:
            Either a parsed Pydantic model or the raw string response
        """
        query_text = query or self.user_query
        formatted_prompt = self.prompt_template.format(query=query_text) if self.prompt_template else query_text
        logger.info(f"Formatted prompt: {formatted_prompt[:100]}...")
        
        responses = await self._generate([formatted_prompt])
        response_text = responses[0]
        logger.info(f"Raw response: {response_text[:100]}...")

        if not self.output_parser:
            return response_text
            
            
        try:
            parsed_response = self.output_parser.parse(response_text)
            return cast(T, parsed_response)  # Cast to the generic type
        except Exception as e:
            logger.error(f"Failed to parse output: {str(e)}")
            raise ParseError(f"Failed to parse output: {str(e)}\nRaw output: {response_text[:200]}...")

    # State Management
    def get_state(self, key: str) -> Any:
        """Get a value from the state store"""
        return self.state.get(key)

    def update_state(self, key: str, value: Any) -> None:
        """Update a value in the state store"""
        self.state[key] = value
        logger.debug(f"Updated state key '{key}'")

    def clear_state(self) -> None:
        """Clear all state values"""
        self.state = {}
        logger.debug("State cleared")

    async def add_tool(self, tool: BaseTool) -> None:
        """
        Add a new tool to the agent and reinitialize
        
        Args:
            tool: The tool to add
        """
        self.tools.append(tool)
        self._setup_agent()  # Reinitialize the agent with the new tool
        logger.info(f"Added tool: {tool.name}")

    def __del__(self) -> None:
        """Cleanup when instance is deleted"""
        try:
            self._gc.force_cleanup()
            logger.info(f"Cleaned up LLM instance: {self._llm_type}")
        except:
            pass  # Suppress errors during deletion


# Example usage
# if __name__ == "__main__":
#     # Create a response structure
#     class WeatherResponse(BaseModel):
#         temperature: float
#         condition: str
#         forecast: str
    
#     # Initialize the LLM
#     llm = BreeLLM(
#         system_prompt="You are a helpful weather assistant. Provide accurate weather information.",
#         user_query="What's the weather like in Paris today?",
#         output_struct=WeatherResponse,
#         config=LLMConfig(temperature=0.2, max_tokens=1000)
#     )
    
#     # Run async response generation
#     def get_weather(city:str):
#         """Use API to get weather data"""
        
#     weather_tool =
#     result = asyncio.run(llm.generate_response())
#     print(f"Response: {result}")