import os
from abc import ABC, abstractmethod
from typing import TypeVar

import openai
import structlog
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

load_dotenv()
logger = structlog.get_logger()

ResponseFormatT = TypeVar("ResponseFormatT", bound=BaseModel)


class LLMClient(ABC):
    """
    Abstract base class for Language Model clients.

    This class serves as a template for implementing various LLM client interfaces.
    Future implementations can extend this class with specific LLM APIs.
    """

    ...


class StructuredOutputLLMClient(ABC):
    """
    Abstract base class for LLM clients that support structured output parsing.

    This class defines the interface for LLM clients that can parse responses
    into structured Pydantic models.
    """

    @abstractmethod
    def chat_completion_parsed(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        response_format: type[ResponseFormatT],
    ) -> ResponseFormatT:
        raise NotImplementedError


class OpenAIClient(StructuredOutputLLMClient):
    """
    OpenAI API client implementation supporting structured output parsing.

    This client handles authentication and interaction with OpenAI's chat completion API,
    including parsing responses into structured Pydantic models.

    Attributes:
        client: Configured OpenAI API client instance.
    """

    def __init__(self):
        """
        Initialize the OpenAI client with API credentials from environment variables.
        """
        if not (openai_api_key := os.getenv("OPENAI_API_KEY")):
            raise RuntimeError("Missing required env var: OPENAI_API_KEY")

        if not (openai_org_id := os.getenv("OPENAI_ORG_ID")):
            raise RuntimeError("Missing required env var: OPENAI_API_KEY")

        self.client = openai.OpenAI(
            api_key=openai_api_key,
            organization=openai_org_id,
        )

    def chat_completion_parsed(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        response_format: type[ResponseFormatT],
    ) -> ResponseFormatT:
        """
        Generate a chat completion and parse it into a valid BaseModel as defined by the response_format model.

        :param messages: List of chat messages for the completion.
        :param model: The OpenAI model to use.
        :param response_format: Pydantic model class for parsing the response.
        :return: Parsed completion response.
        """
        completion = self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
        )
        print(f"Completion Message: ", completion.choices[0])
        return completion.choices[0].message.parsed
