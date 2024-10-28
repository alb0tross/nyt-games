from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from src.connections import DailyConnections
from src.connections.models import DailyConnectionsSolution
from src.connections.utils import create_guess_model, parse_category_string
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
        """
        Initialize the ConnectionsSolver with a compatible LLM client that can utilize structured outputs using
        Pydantic.
        """
        self.client = client

    def _get_category_guess(
        self,
        words: list[str],
        previous_guesses: list[tuple[str, str, str, str]] | None = None,
    ) -> tuple[str, str, str, str]:
        """
        Get a category guess from the LLM for the given words.

        This method uses the LLM client to predict the next most likely group of 4 related words,
        based on the given words and previous guesses.

        :param words: A list of words to guess.
        :param previous_guesses: A list of tuples of words that have been previously guessed.
        :return: A tuple of 4 words representing the next category guess.
        """
        response_model = create_guess_model(words, previous_guesses)

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
                content="Find the next most likely group of 4 related words.",
            ),
        ]

        response = self.client.chat_completion_parsed(
            messages=messages,
            response_format=response_model,
            model="gpt-4o-mini"
        )

        # Get the current category number based on previous guesses
        category_num = len(previous_guesses) + 1 if previous_guesses else 1
        category_field = f"category_{category_num}"

        if not hasattr(response, category_field):
            raise RuntimeError(
                f"Category {category_num} not found in response: {response.model_dump_json()}"
            )

        category = getattr(response, category_field)
        return parse_category_string(category)

    @staticmethod
    def _check_solution(
        solution: DailyConnectionsSolution,
        puzzle: DailyConnections,
    ) -> int:
        """
        Check the solution against the puzzle's actual solutions.

        This method compares the guessed categories with the actual solution categories
        by matching sets of words, and counts how many categories are correctly identified.

        :param solution: The generated solution to check.
        :param puzzle: The original puzzle with correct solutions.
        :return: Number of correct categories (0-4).
        """
        correct_count = 0
        solution_sets = {
            frozenset(cat.words) for cat in puzzle.solutions
        }

        # Check each category
        for i in range(1, 5):
            category = getattr(solution, f"category_{i}")
            category_set = frozenset(category)

            if category_set in solution_sets:
                correct_count += 1
                # Find the matching solution to get color and theme
                for sol in puzzle.solutions:
                    if frozenset(sol.words) == category_set:
                        print(f"Found correct {sol.color.value} category: {sol.theme}")
                        print(f"Words: {', '.join(category)}")
                        break

        return correct_count

    def solve(self, connections_puzzle: DailyConnections) -> DailyConnectionsSolution:
        """
        Solve a DailyConnections puzzle using an LLM.

        This method attempts to solve the puzzle by making successive guesses
        for word categories using the LLM client. It maintains a list of previous guesses
        to avoid repeating words and ensures that all words are used exactly once.

        :param connections_puzzle: A DailyConnections instance holding the words and solution for the puzzle.
        :return: A DailyConnectionsSolution instance with four unique groups of guesses for each category.
        """
        words = connections_puzzle.words
        previous_guesses: list[tuple[str, str, str, str]] = []

        for guess_num in range(4):
            try:
                guess = self._get_category_guess(words, previous_guesses)
                previous_guesses.append(guess)
                print(f"Category {guess_num + 1}:", ", ".join(guess))

                # Verify words are still available
                remaining_words = set(words)
                for g in previous_guesses:
                    remaining_words -= set(g)

                if len(remaining_words) < 4 and guess_num < 3:
                    raise RuntimeError(
                        f"Not enough remaining words after guess {guess_num + 1}. "
                        f"Remaining: {remaining_words}"
                    )

            except Exception as e:
                raise RuntimeError(
                    f"Failed to make guess {guess_num + 1}: {str(e)}"
                ) from e

        # Convert guesses to solution
        solution = DailyConnectionsSolution(
            category_1=previous_guesses[0],
            category_2=previous_guesses[1],
            category_3=previous_guesses[2],
            category_4=previous_guesses[3],
        )

        # Check and print results
        correct_count = self._check_solution(solution, connections_puzzle)
        print(f"\nFinal Score: {correct_count}/4 correct categories")

        return solution
