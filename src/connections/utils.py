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
            logger.debug(
                "Added correct guess field", category_number=i, guess=guess_str
            )

    # Add field for the next guess as an enum
    next_category_num = len(correct_guesses) + 1 if correct_guesses else 1
    json_schema_extra_for_field: dict[str, Any] = {"enum": enum_values}
    fields[f"category_{next_category_num}"] = (
        str,
        Field(
            ...,
            description=f"Category guess {next_category_num}",
            json_schema_extra=json_schema_extra_for_field,
        ),
    )
    logger.debug(
        "Added next guess field",
        category_number=next_category_num,
        possible_values_count=len(enum_values),
    )

    properties: dict[str, dict[str, Any]] = {}
    if correct_guesses:
        for i, prev_guess in enumerate(correct_guesses, 1):
            properties[f"category_{i}"] = {
                "type": "string",
                "description": f"Category {i}",
                "enum": [",".join(prev_guess)],
            }

    properties[f"category_{next_category_num}"] = {
        "type": "string",
        "description": f"Category {next_category_num}",
        "enum": enum_values,
    }

    json_schema_extra: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": [f"category_{i}" for i in range(1, next_category_num + 1)],
    }

    model_config = ConfigDict(json_schema_extra=json_schema_extra)

    logger.info(
        "Created model configuration",
        field_count=len(fields),
        category_count=next_category_num,
    )

    model = create_model("CategoryGuess", __config__=model_config, **fields)  # type: ignore

    logger.info(
        "Model creation complete",
        model_name=model.__name__,
        field_count=len(model.model_fields),
    )

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
