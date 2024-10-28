import structlog
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from src.connections import DailyConnections
from src.connections.models import DailyConnectionsSolution
from src.connections.utils import create_guess_model, parse_category_string
from src.llm_client import StructuredOutputLLMClient


logger = structlog.get_logger(__name__)


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
        correct_guesses: list[tuple[str, str, str, str]] | None = None,
        previous_incorrect: list[tuple[str, str, str, str]] | None = None,
    ) -> tuple[str, str, str, str]:
        """
        Get a category guess from the LLM for the given words.

        :param words: A list of words to guess.
        :param correct_guesses: A list of tuples of words that have been correctly guessed.
        :param previous_incorrect: A list of tuples of words that were incorrect for this category.
        :return: A tuple of 4 words representing the next category guess.
        """
        response_model = create_guess_model(words, correct_guesses)

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
        category_num = len(correct_guesses) + 1 if correct_guesses else 1
        category_field = f"category_{category_num}"

        if not hasattr(response, category_field):
            raise RuntimeError(
                f"Category {category_num} not found in response: {response.model_dump_json()}"
            )

        category = getattr(response, category_field)
        return parse_category_string(category)

    @staticmethod
    def _check_guess(
        guess: tuple[str, str, str, str],
        puzzle: DailyConnections,
    ) -> tuple[bool, str | None, str | None]:
        """
        Check if a guess is correct and return the category details if it is.

        :param guess: The guessed words to check
        :param puzzle: The puzzle containing the solutions
        :return: Tuple of (is_correct, color, theme)
        """
        guess_set = frozenset(guess)
        for solution in puzzle.solutions:
            if frozenset(solution.words) == guess_set:
                return True, solution.color, solution.theme
        return False, None, None

    def solve(self, connections_puzzle: DailyConnections) -> DailyConnectionsSolution | None:
        """
        Solve a DailyConnections puzzle using an LLM.

        This method attempts to solve the puzzle by making successive guesses
        for word categories using the LLM client. It maintains a list of previous guesses
        to avoid repeating words and ensures that all words are used exactly once.

        :param connections_puzzle: A DailyConnections instance holding the words and solution for the puzzle.
        :return: A DailyConnectionsSolution instance with four unique groups of guesses for each category.
        """
        words = connections_puzzle.words
        correct_guesses: list[tuple[str, str, str, str]] = []
        wrong_attempts = 0

        current_category_incorrect: list[tuple[str, str, str, str]] = []

        while len(correct_guesses) < 4 and wrong_attempts < 4:
            current_category = len(correct_guesses) + 1

            guess = self._get_category_guess(
                words=words,
                correct_guesses=correct_guesses,
                previous_incorrect=current_category_incorrect
            )

            is_correct, color, theme = self._check_guess(
                guess=guess,
                puzzle=connections_puzzle,
            )

            if is_correct:
                correct_guesses.append(guess)
                remaining_words = set(words)
                for g in correct_guesses:
                    remaining_words -= set(g)
                logger.info(
                    f"Correctly guessed {color} category: {theme}.\n"
                    f"Remaining words: {remaining_words}"
                )
                # Reset incorrect guesses for next category
                current_category_incorrect = []
            else:
                wrong_attempts += 1
                current_category_incorrect.append(guess)
                logger.info(
                    f"Incorrect guess number {wrong_attempts} for category {current_category}: "
                    f"{', '.join(guess)}."
                )

                if wrong_attempts >= 4:
                    logger.info("Failed to solve puzzle: Maximum wrong attempts reached")
                    return None

        solution = DailyConnectionsSolution(
            category_1=correct_guesses[0],
            category_2=correct_guesses[1],
            category_3=correct_guesses[2],
            category_4=correct_guesses[3],
        )

        logger.info(f"Solution:\n{solution.model_dump_json(indent=2)}")
        return solution
