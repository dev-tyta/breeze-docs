import os
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, model_validator


class LLMParser(BaseModel):
    