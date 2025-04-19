# Import relevant functionality
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class LLMConfig(BaseModel):
    """Configuration settings for the LLM"""
    model_name: str = Field(default="gemini-2.0-flash")
    max_tokens: int = Field(default=4096)
    max_iterations: int = Field(default=9)
    temperature: float = Field(default=0.7)
    api_key_env_var: str = Field(default="GEMINI_API_KEY")
    timeout: int = Field(default=30)
    retry_attempts: int = Field(default=3)

config = LLMConfig()

api_key = os.getenv(config.api_key_env_var)
tavily= os.getenv("TAVILY_API_KEY")

# Add a check to ensure the key was loaded
if not tavily:
    raise ValueError("TAVILY_API_KEY not found. Please set it in your environment or .env file.")
if not api_key:
     raise ValueError(f"{config.api_key_env_var} not found. Please set it in your environment or .env file.")


model = ChatGoogleGenerativeAI(
    api_key=api_key,
    model=config.model_name,
    max_tokens=config.max_tokens,
    temperature=config.temperature,
    timeout=config.timeout
)

# Create the agent
memory = MemorySaver()
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
search = TavilySearchResults(max_results=2, tavily_api_key=tavily)
tools = [search]
# tools_section = """Answer the following questions as best you can. You have access to the following tools:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question""" if tools else """Answer the following questions as best you can.

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Final Answer: the final answer to the original input question"""

# system_message = SystemMessage(content=tools_section)
# human_message = HumanMessage(content="{input}")

# # Create the prompt properly for ReAct agent
# react_prompt = ChatPromptTemplate.from_messages([
#     system_message,
#     human_message,
#     # MessagesPlaceholder(variable_name="agent_scratchpad")  # This is the key fix - correct format for agent_scratchpad
# ])



agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}
# for step in agent_executor.stream(
    
#     config,
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()

# 
output = agent_executor.invoke(
    input= {"messages": [HumanMessage(content="hi im bob! and i live in sf, where can i have fun in sf?")]} ,
    config = config
                      )

print(output["messages"])