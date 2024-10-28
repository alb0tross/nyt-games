from itertools import combinations
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model, ConfigDict


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
    previous_guesses: list[tuple[str, str, str, str]] | None = None
) -> type[BaseModel]:
    """
    Create a Pydantic model for guessing categories based on available words and previous guesses.

    :param words: List of all words in the puzzle
    :param previous_guesses: List of previously guessed word groups
    :returns: A Pydantic model class with fields for all previous guesses plus the next guess
    """
    # Filter out words that have been used in previous guesses
    used_words = set()
    if previous_guesses:
        used_words = {word for guess in previous_guesses for word in guess}

    available_words = [w for w in words if w not in used_words]

    # Generate possible combinations from remaining words (limited to 500)
    possible_groups = list(combinations(sorted(available_words), 4))[:500]
    enum_values: list[str] = [",".join(group) for group in possible_groups]

    # Create field definitions for the model
    fields: dict[str, tuple[Any, Any]] = {}

    # Add fields for previous guesses as string literals of the exact prior guess
    if previous_guesses:
        for i, prev_guess in enumerate(previous_guesses, 1):
            guess_str = ",".join(prev_guess)
            fields[f"category_{i}"] = (
                Literal[guess_str],  # type: ignore
                Field(
                    default=guess_str,
                    description=f"Previously guessed category {i}",
                )
            )

    # Add field for the next guess as an enum
    next_category_num = len(previous_guesses) + 1 if previous_guesses else 1
    json_schema_extra_for_field: dict[str, Any] = {"enum": enum_values}
    fields[f"category_{next_category_num}"] = (
        str,
        Field(
            ...,
            description=f"Category guess {next_category_num}",
            json_schema_extra=json_schema_extra_for_field
        )
    )

    # Create properties dict for json schema
    properties: dict[str, dict[str, Any]] = {}
    if previous_guesses:
        # Add properties for previous guesses
        for i, prev_guess in enumerate(previous_guesses, 1):
            properties[f"category_{i}"] = {
                "type": "string",
                "description": f"Category {i}",
                "enum": [",".join(prev_guess)]
            }

    # Add property for next guess
    properties[f"category_{next_category_num}"] = {
        "type": "string",
        "description": f"Category {next_category_num}",
        "enum": enum_values
    }

    json_schema_extra: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": [f"category_{i}" for i in range(1, next_category_num + 1)]
    }

    model_config = ConfigDict(
        json_schema_extra=json_schema_extra
    )

    # Create the model dynamically with all fields
    return create_model(
        "CategoryGuess",
        __config__=model_config,
        **fields
    )  # type: ignore



