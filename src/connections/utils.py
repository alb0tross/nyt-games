from itertools import combinations
from typing import Any, List

from pydantic import BaseModel, Field


def create_guess_model(
    words: list[str], previous_guesses: list[tuple[str, str, str, str]] | None = None
) -> type[BaseModel]:
    """
    Create a Pydantic model for guessing the next category based on available words.

    This function generates a dynamic Pydantic model that constrains possible guesses
    to valid combinations of the remaining unused words in the puzzle.

    :param words: List of all words in the puzzle.
    :param previous_guesses: Optional list of previously guessed word groups.
    :returns: A Pydantic model class for the next guess with a flattened JSON schema.
    """
    # Filter out words that have been used in previous guesses
    used_words = set()
    if previous_guesses:
        used_words = {word for guess in previous_guesses for word in guess}

    available_words = [w for w in words if w not in used_words]

    # Generate possible combinations from remaining words (limited to 500)
    possible_groups = list(combinations(sorted(available_words), 4))[:500]
    # Convert tuples to strings for the enum
    enum_values: list[Any] = [",".join(group) for group in possible_groups]

    class CategoryGuess(BaseModel):
        category_1: str = Field(
            ...,
            description="A single group of four words that could be a category",
            json_schema_extra={"type": "string", "enum": enum_values},
        )

        model_config = {
            "json_schema_extra": {
                "type": "object",
                "properties": {
                    "category_1": {
                        "type": "string",
                        "description": "A single group of four words that could be a category",
                        "enum": enum_values,
                    }
                },
                "required": ["category_1"],
            }
        }

    return CategoryGuess
