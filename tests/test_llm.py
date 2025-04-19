import unittest
from unittest.mock import patch, AsyncMock
import asyncio
from pydantic import BaseModel, Field

from src.core.llm import BreeLLM
from src.config.config import LLMConfig
from src.exceptions import ParseError, APIError

# Define a sample Pydantic model for testing output parsing
class TestResponse(BaseModel):
    answer: str = Field(...)
    confidence: float = Field(...)

class TestBreeLLM(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.config = LLMConfig(
            model_name="gemini-2.0-flash",
            api_key_env_var="TEST_API_KEY",
            max_tokens=100,
            temperature=0.5,
            timeout=10
        )
        # Set a dummy API key for testing
        import os
        os.environ["TEST_API_KEY"] = "test_key"

    @patch("src.core.llm.GoogleGenerativeAI")
    async def test_initialization(self, MockGoogleGenerativeAI):
        llm = BreeLLM(
            input_prompt="Say hello to {message}",
            query="the world",
            config=self.config
        )
        self.assertIsNotNone(llm.prompt)
        self.assertIsNotNone(llm.llm)
        MockGoogleGenerativeAI.assert_called_once_with(
            api_key="test_key",
            model="gemini-1.5-pro-latest",
            max_tokens=100,
            temperature=0.5,
            timeout=10
        )

    @patch("src.core.llm.GoogleGenerativeAI")
    async def test_generate_response_no_output_parser(self, MockGoogleGenerativeAI):
        mock_response = AsyncMock()
        mock_response.text = "Hello, the world!"
        mock_llm_instance = MockGoogleGenerativeAI.return_value
        mock_llm_instance.invoke.return_value = mock_response

        llm = BreeLLM(
            input_prompt="Say hello to {message}",
            query="the world",
            config=self.config
        )
        response = await llm.generate_response()
        self.assertEqual(response, "Hello, the world!")
        mock_llm_instance.invoke.assert_called_once_with("Say hello to the world")

    @patch("src.core.llm.GoogleGenerativeAI")
    async def test_generate_response_with_output_parser(self, MockGoogleGenerativeAI):
        mock_response = AsyncMock()
        mock_response.text = '{"answer": "The answer is 42", "confidence": 0.9}'
        mock_llm_instance = MockGoogleGenerativeAI.return_value
        mock_llm_instance.invoke.return_value = mock_response

        llm = BreeLLM(
            input_prompt="Answer the question and provide a confidence score.",
            query="What is the meaning of life?",
            output_struct=TestResponse,
            config=self.config
        )
        response = await llm.generate_response()
        self.assertIsInstance(response, TestResponse)
        self.assertEqual(response.answer, "The answer is 42")
        self.assertEqual(response.confidence, 0.9)
        mock_llm_instance.invoke.assert_called_once_with("Answer the question and provide a confidence score.\n\nYou must respond according to the following format: ```json\n{\n \"answer\": string,\n \"confidence\": number\n}\n```")

    @patch("src.core.llm.GoogleGenerativeAI")
    async def test_generate_response_with_output_parser_parse_error(self, MockGoogleGenerativeAI):
        mock_response = AsyncMock()
        mock_response.text = '{"answer": "The answer is 42", "confidence": "not a float"}' # Invalid format
        mock_llm_instance = MockGoogleGenerativeAI.return_value
        mock_llm_instance.invoke.return_value = mock_response

        llm = BreeLLM(
            input_prompt="Answer the question and provide a confidence score.",
            query="What is the meaning of life?",
            output_struct=TestResponse,
            config=self.config
        )
        with self.assertRaises(ParseError):
            await llm.generate_response()

    @patch("src.core.llm.GoogleGenerativeAI")
    async def test_generate_response_api_error(self, MockGoogleGenerativeAI):
        mock_llm_instance = MockGoogleGenerativeAI.return_value
        mock_llm_instance.invoke.side_effect = Exception("API Error")

        llm = BreeLLM(
            input_prompt="Ask a question",
            query="Will this test fail?",
            config=self.config
        )
        with self.assertRaises(APIError):
            await llm.generate_response()

if __name__ == "__main__":
    unittest.main()