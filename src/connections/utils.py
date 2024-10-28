from itertools import combinations
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, create_model

logger = structlog.get_logger(__name__)


def parse_category_string(category: str) -> tuple[str, str, str, str]:
    """
    Parse a category string into a tuple of four words.

    :param category: Comma-separated string of four words
    :returns: Tuple of four words
    """
    words = category.split(",")
    if len(words) != 4:
        raise ValueError(f"Category must contain exactly 4 words, got: {category}")
    return tuple(words)  # type: ignore


def create_guess_model(
    words: list[str],
    correct_guesses: list[tuple[str, str, str, str]] | None = None,
    incorrect_guesses: list[tuple[str, str, str, str]] | None = None,
) -> type[BaseModel]:
    """
    Create a Pydantic model for guessing categories based on available words and previous guesses.

    :param words: List of all words in the puzzle
    :param correct_guesses: List of correctly guessed word groups
    :param incorrect_guesses: List of incorrect guesses for current category
    :returns: A Pydantic model class with fields for all previous guesses plus the next guess
    """
    logger.info(
        "Creating guess model",
        total_words=len(words),
        correct_guesses=len(correct_guesses) if correct_guesses else 0,
        incorrect_guesses=len(incorrect_guesses) if incorrect_guesses else 0,
    )

    # Filter out words that have been used in correct guesses
    used_words = set()
    if correct_guesses:
        used_words = {word for guess in correct_guesses for word in guess}
        logger.debug(
            "Filtered out words from correct guesses", used_words=sorted(used_words)
        )

    available_words = [w for w in words if w not in used_words]

    # Generate all possible combinations
    possible_groups = list(combinations(sorted(available_words), 4))
    logger.debug(
        "Generated initial combinations",
        total_combinations=len(possible_groups),
        sample_combinations=possible_groups[:3] if possible_groups else [],
    )

    # Remove any combinations that match prior incorrect guesses
    if incorrect_guesses:
        incorrect_sets = {frozenset(guess) for guess in incorrect_guesses}
        logger.debug(
            "Removing incorrect guesses",
            incorrect_guesses=[",".join(guess) for guess in incorrect_guesses],
            incorrect_count=len(incorrect_sets),
        )

        original_count = len(possible_groups)
        possible_groups = [
            group for group in possible_groups if frozenset(group) not in incorrect_sets
        ]
        logger.info(
            "Filtered out incorrect combinations",
            original_count=original_count,
            remaining_count=len(possible_groups),
            removed_count=original_count - len(possible_groups),
        )

    # todo. solution for approximating high value groups over hardcoded limit
    # Limit to 500 combinations after filtering; limit of OpenAI's structured outputs feature
    if len(possible_groups) > 500:
        logger.info(
            "Limiting possible combinations to 500", original_count=len(possible_groups)
        )
        possible_groups = possible_groups[:500]

    enum_values: list[str] = [",".join(group) for group in possible_groups]
    logger.debug(
        "Created enum values",
        value_count=len(enum_values),
        sample_values=enum_values[:3] if enum_values else [],
    )

    fields: dict[str, tuple[Any, Any]] = {}
    properties: dict[str, dict[str, Any]] = {}

    # Add fields for correct guesses as string literals
    if correct_guesses:
        for i, prev_guess in enumerate(correct_guesses, 1):
            guess_str = ",".join(prev_guess)
            fields[f"category_{i}"] = (
                Literal[guess_str],  # type: ignore
                Field(
                    default=guess_str,
                    description=f"Previously guessed category {i}",
                ),
            )
            properties[f"category_{i}"] = {
                "type": "string",
                "description": f"Category {i}",
                "enum": [guess_str],
            }

    next_category_num = len(correct_guesses) + 1 if correct_guesses else 1

    # Add explanation field for Chain of Thought reasoning
    fields["explanation"] = (
        str,
        Field(
            ...,
            description=(
                f"Explain your reasoning for category {next_category_num} guess. "
                "What theme connects these words and why are you choosing them?"
            ),
        ),
    )
    properties["explanation"] = {
        "type": "string",
        "description": (
            f"Explain your reasoning for category {next_category_num} guess. "
            "What theme connects these words and why are you choosing them?"
        ),
    }

    # Add field for the next guess as an enum
    json_schema_extra_for_field: dict[str, Any] = {"enum": enum_values}
    fields[f"category_{next_category_num}"] = (
        str,
        Field(
            ...,
            description=f"Category guess {next_category_num}",
            json_schema_extra=json_schema_extra_for_field,
        ),
    )
    properties[f"category_{next_category_num}"] = {
        "type": "string",
        "description": f"Category {next_category_num}",
        "enum": enum_values,
    }

    json_schema_extra: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": ["explanation"]
        + [f"category_{i}" for i in range(1, next_category_num + 1)],
    }

    model_config = ConfigDict(json_schema_extra=json_schema_extra)

    model = create_model("CategoryGuess", __config__=model_config, **fields)  # type: ignore

    return model


def create_revision_model(
    prior_guess: tuple[str, str, str, str],
    available_words: list[str],
) -> type[BaseModel]:
    """
    Create a model for revising a guess that had 3 correct words.

    :param prior_guess:
    :param available_words:
    :return:
    """
    # Create enum values for the prior guess words
    prior_guess_enum: list[Any] = list(prior_guess)

    # Create enum values for available replacement words
    # Filter out words that were in the prior guess
    replacement_words: list[Any] = [w for w in available_words if w not in prior_guess]

    prior_guess_json_schema_extra: dict[str, Any] = {"enum": prior_guess_enum}
    replacement_words_json_schema_extra: dict[str, Any] = {"enum": replacement_words}

    return create_model(
        "RevisePriorGuess",
        explanation=(
            str,
            Field(
                ...,
                description="An explanation of your logic behind the word replacement you are making. State which"
                "word you are replacing and which word you are replacing it with and explain your reasoning.",
            ),
        ),
        prior_guess_word_to_replace=(
            str,
            Field(
                ...,
                description="Which word from the prior guess should be replaced",
                json_schema_extra=prior_guess_json_schema_extra,
            ),
        ),
        word_to_use_as_replacement=(
            str,
            Field(
                ...,
                description="Which available word should be used as the replacement",
                json_schema_extra=replacement_words_json_schema_extra,
            ),
        ),
        __config__=ConfigDict(
            json_schema_extra={
                "title": "Revise Prior Guess",
                "description": "Model for revising a guess that had 3 correct words",
            }
        ),
    )
