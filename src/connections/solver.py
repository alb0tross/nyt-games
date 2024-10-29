import structlog
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from src.connections import DailyConnections
from src.connections.models import DailyConnectionsSolution
from src.connections.utils import (
    create_guess_model,
    create_guess_word_model,
    create_revision_model,
    parse_category_string,
)
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
        self.action_messages: list[ChatCompletionMessageParam] = []

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
        response_model = create_guess_model(
            words=words,
            correct_guesses=correct_guesses,
            incorrect_guesses=previous_incorrect,
        )

        func_prompt_messages: list[ChatCompletionMessageParam] = [
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

        messages = (
            func_prompt_messages[:1] + self.action_messages + func_prompt_messages[1:]
        )

        response = self.client.chat_completion_parsed(
            messages=messages, response_format=response_model, model="gpt-4o-mini"
        )

        self.action_messages.append(
            ChatCompletionAssistantMessageParam(
                role="assistant", content=response.model_dump_json()
            )
        )

        # Get the current category number based on previous guesses
        category_num = len(correct_guesses) + 1 if correct_guesses else 1
        category_field = f"category_{category_num}"

        if not hasattr(response, category_field):
            raise RuntimeError(
                f"Category {category_num} not found in response: {response.model_dump_json()}"
            )

        category = getattr(response, category_field)
        logger.info(
            f"LLM guessed {category} explaining that: '{getattr(response, 'explanation')}'"
        )

        return parse_category_string(category)

    def _get_category_by_word_guess(
        self,
        words: list[str],
        correct_guesses: list[tuple[str, str, str, str]] | None = None,
    ) -> tuple[str, str, str, str]:
        """
        Get a category guess from the LLM for the given words by selecting one word at a time.

        :param words: A list of words to guess.
        :param correct_guesses: A list of tuples of words that have been correctly guessed.
        :return: A tuple of 4 words representing the next category guess.
        """
        available_words = words.copy()
        selected_words: dict[str, str] = {}
        explanation: str | None = None

        # Base system message that stays constant
        base_system_message = ChatCompletionSystemMessageParam(
            role="system",
            content=(
                "You are solving a Connections puzzle. You need to identify groups "
                "of 4 related words. Each group should have a clear theme "
                "connecting all words. Select one word at a time to build your category."
            ),
        )

        # Select words one at a time
        for word_number in range(1, 5):
            # Create model for current word selection
            response_model = create_guess_word_model(
                words=available_words,
                word_number=word_number,
                correct_guesses=correct_guesses,
                previous_selections=selected_words if selected_words else None,
                previous_explanation=explanation,
            )

            # Create appropriate user message based on progress
            if word_number == 1:
                user_content = (
                    "Select the first word for a new category and explain the theme "
                    "you are building."
                )
            else:
                selected_words_str = ", ".join(selected_words.values())
                user_content = (
                    f"You've selected: {selected_words_str}. "
                    f"Select word #{word_number} that matches your theme: {explanation}"
                )

            messages: list[ChatCompletionMessageParam] = [
                base_system_message,
                *self.action_messages,
                ChatCompletionUserMessageParam(role="user", content=user_content),
            ]

            # Get response for this word
            response = self.client.chat_completion_parsed(
                messages=messages, response_format=response_model, model="gpt-4o"
            )

            # Save the response
            self.action_messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant", content=response.model_dump_json()
                )
            )

            # Extract the selected word
            word_field = f"word_{word_number}"
            selected_word = getattr(response, word_field)

            # Store the word and remove it from available words
            selected_words[word_field] = selected_word
            available_words.remove(selected_word)

            explanation = getattr(response, "explanation")
            logger.info(f"Building category with theme: {explanation}")

            logger.info(f"Selected word {word_number}: {selected_word}")

        # Convert dictionary to tuple maintaining order
        final_words = tuple(selected_words[f"word_{i}"] for i in range(1, 5))

        logger.info(
            "Completed category guess",
            selected_words=final_words,
            explanation=explanation,
        )

        return final_words  # type: ignore

    def _edit_category_guess(
        self,
        words: list[str],
        prior_guess: tuple[str, str, str, str],
        theme: str,
        correct_guesses: list[tuple[str, str, str, str]] | None = None,
    ) -> tuple[str, str, str, str]:
        """
        Get a revised category guess when 3 words were correct.

        :param words: List of all available words.
        :param prior_guess: The previous guess containing 3 correct words.
        :param theme: The theme of the category that was partially guessed.
        :param correct_guesses: List of previously correct guesses
        :return: A new tuple of 4 words with one word replaced
        """
        used_words = set()
        if correct_guesses:
            used_words = {word for guess in correct_guesses for word in guess}

        # Get available words (not used in correct guesses)
        available_words = [w for w in words if w not in used_words]

        response_model = create_revision_model(
            prior_guess=prior_guess, available_words=available_words
        )

        func_prompt_messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=(
                    "You are helping solve a Connections puzzle. The previous guess "
                    f"had 3 correct words. You need to identify which word was incorrect and replace it with "
                    f"a better match."
                ),
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Which word should be replaced and what should replace it?",
            ),
        ]

        messages = (
            func_prompt_messages[:1] + self.action_messages + func_prompt_messages[1:]
        )

        response = self.client.chat_completion_parsed(
            messages=messages, response_format=response_model, model="gpt-4o"
        )

        self.action_messages.append(
            ChatCompletionAssistantMessageParam(
                role="assistant", content=response.model_dump_json()
            )
        )

        word_list = list(prior_guess)
        if not hasattr(response, "prior_guess_word_to_replace") or not hasattr(
            response, "word_to_use_as_replacement"
        ):
            raise RuntimeError("Invalid response when editing a prior incorrect guess.")

        replace_idx = word_list.index(response.prior_guess_word_to_replace)
        word_list[replace_idx] = response.word_to_use_as_replacement

        explanation = getattr(response, "explanation", None)

        logger.info(
            "Editing category guess",
            removed_word=response.prior_guess_word_to_replace,
            replacement_word=response.word_to_use_as_replacement,
            explanation=explanation,
            theme=theme,
            response_model=response_model.model_fields,
            response_json=response.model_dump_json(),
        )

        return word_list[0], word_list[1], word_list[2], word_list[3]

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

    @staticmethod
    def _check_for_partial_match(
        guess: tuple[str, str, str, str],
        puzzle: DailyConnections,
    ) -> tuple[set[str], str, str] | None:
        """
        Check if the guess contains exactly 3 correct words from any solution category.

        :param guess: The guessed words to check
        :param puzzle: The puzzle containing the solution.
        :return: A tuple containing the set of correct words, the category theme, and category color if a partial match
         is found; otherwise, None.
        """
        guess_set = set(guess)
        for solution in puzzle.solutions:
            common_words = guess_set.intersection(set(solution.words))
            if len(common_words) == 3:
                return common_words, solution.theme, solution.color
        return None

    def solve(
        self, connections_puzzle: DailyConnections
    ) -> DailyConnectionsSolution | None:
        """
        Solve a DailyConnections puzzle using an LLM.
        """
        words = connections_puzzle.words
        logger.info(f"Solving daily connection with available words:\n{words}")
        correct_guesses: list[tuple[str, str, str, str]] = []
        wrong_attempts = 0

        current_category_incorrect: list[tuple[str, str, str, str]] = []

        # Track if we're in revision mode
        revising_guess = False
        current_theme: str | None = None
        last_guess: tuple[str, str, str, str] | None = None

        while len(correct_guesses) < 4 and wrong_attempts < 4:
            current_category = len(correct_guesses) + 1

            if revising_guess and last_guess and current_theme:
                # Try to revise the guess with 3 correct words
                guess = self._edit_category_guess(
                    words=words,
                    prior_guess=last_guess,
                    theme=current_theme,
                    correct_guesses=correct_guesses,
                )
            else:
                # Make a regular guess
                # guess = self._get_category_guess(
                #     words=words,
                #     correct_guesses=correct_guesses,
                #     previous_incorrect=current_category_incorrect,
                # )
                guess = self._get_category_by_word_guess(
                    words=words, correct_guesses=correct_guesses
                )

            is_correct, color, theme = self._check_guess(
                guess=guess,
                puzzle=connections_puzzle,
            )

            if is_correct:
                self.action_messages.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=f"You correctly identified the category {theme} using the words: {guess}!",
                    )
                )

                correct_guesses.append(guess)
                remaining_words = set(words)
                for g in correct_guesses:
                    remaining_words -= set(g)
                logger.info(
                    f"Correctly guessed {color} category: {theme}.\n"
                    f"Remaining words: {remaining_words}"
                )
                # Reset states
                current_category_incorrect = []
                revising_guess = False
                current_theme = None
                last_guess = None
            else:
                # Check for partial match (3 correct words)
                partial_match = self._check_for_partial_match(guess, connections_puzzle)

                if partial_match:
                    self.action_messages.append(
                        ChatCompletionUserMessageParam(
                            role="user",
                            content=f"You almost found a category with {guess}! However, one word here doesn't belong"
                            f"and you need to identify which word to replace it with to complete the category.",
                        )
                    )
                    correct_words, match_theme, match_color = partial_match
                    logger.info(
                        "Found partial match with 3 correct words",
                        theme=match_theme,
                        color=match_color,
                        correct_words=list(correct_words),
                    )
                    # Enter revision mode
                    revising_guess = True
                    current_theme = match_theme
                    last_guess = guess
                else:
                    self.action_messages.append(
                        ChatCompletionUserMessageParam(
                            role="user",
                            content=f"Nope, the words: {guess} do not make up a valid category.",
                        )
                    )
                    # Regular wrong guess
                    revising_guess = False
                    current_theme = None
                    last_guess = None

                wrong_attempts += 1
                current_category_incorrect.append(guess)
                logger.warn(
                    f"Incorrect guess number {wrong_attempts} for category {current_category}: "
                    f"{', '.join(guess)}."
                )

                if wrong_attempts >= 4:
                    logger.warn(
                        "Failed to solve puzzle: Maximum wrong attempts reached",
                        messages=self.action_messages,
                    )
                    return None

        # Create solution from correct guesses
        solution = DailyConnectionsSolution(
            category_1=correct_guesses[0],
            category_2=correct_guesses[1],
            category_3=correct_guesses[2],
            category_4=correct_guesses[3],
        )

        logger.info(
            f"Solution:\n{solution.model_dump_json(indent=2)}",
            messages=self.action_messages,
        )
        return solution
