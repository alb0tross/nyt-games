from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from src.connections import DailyConnections
from src.connections.utils import create_guess_model
from src.llm_client import StructuredOutputLLMClient


class ConnectionsSolver:
    """
    Solver for the Connections word puzzle game using LLM-based reasoning.

    This class implements the logic for solving Connections puzzles by identifying
    groups of related words using an LLM to make informed guesses.

    Attributes:
        client: A structured output LLM client for making category predictions.
    """

    def __init__(
        self,
        client: StructuredOutputLLMClient,
    ):
        self.client = client

    def _get_category_guess(
        self,
        words: list[str],
        response_model: type[BaseModel],
    ) -> BaseModel:
        """
        Get a category guess from the LLM for the given words.
        """
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=(
                    "You are solving a Connections puzzle. You need to identify groups "
                    "of 4 related words. Each group should have a clear theme "
                    "connecting all words."
                ),
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=f"Find the next most likely group of 4 related words from: {', '.join(words)}",
            ),
        ]

        return self.client.chat_completion_parsed(
            messages=messages, response_format=response_model, model="gpt-4o-mini"
        )

    def solve(self, connections_puzzle: DailyConnections) -> None:
        """
        Solve a DailyConnections puzzle using LLM-based reasoning.

        This method attempts to solve the puzzle by making successive guesses
        for word categories using the LLM client.

        :param connections_puzzle: The puzzle to solve, containing words and solutions.
        :return:
        """
        words = connections_puzzle.words

        CategoryGuessModel = create_guess_model(words)
        print("Schema:", CategoryGuessModel.model_json_schema())

        first_response = self._get_category_guess(words, CategoryGuessModel)
        if hasattr(first_response, 'category_1'):
            print("First category:", first_response.category_1)
        else:
            raise RuntimeError(f"First category not found; got {first_response.model_dump_json()}")
