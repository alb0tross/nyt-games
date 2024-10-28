from enum import StrEnum

from pydantic import BaseModel, Field, validator, field_validator
from pydantic_core.core_schema import FieldValidationInfo


class CategoryColor(StrEnum):
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    PURPLE = "purple"


class CategorySolution(BaseModel):
    color: CategoryColor
    theme: str
    words: list[str] = Field(..., min_length=4, max_length=4)

    class Config:
        frozen = True


class DailyConnections(BaseModel):
    date: str
    words: list[str] = Field(..., min_length=16, max_length=16)
    solutions: list[CategorySolution] = Field(..., min_length=4, max_length=4)

    @classmethod
    @field_validator('words')
    def validate_unique_words(cls, v: list[str]) -> list[str]:
        """Validate that all words in the puzzle are unique."""
        if len(set(v)) != 16:
            raise ValueError("All words in 'words' must be unique.")
        return v

    @classmethod
    @field_validator('solutions')
    def validate_solutions(
        cls, v: list[CategorySolution], info: FieldValidationInfo
    ) -> list[CategorySolution]:
        """Validate that all solution words are present in the words list."""
        words = info.data.get('words', [])
        if not words:
            raise ValueError("Words must be validated before solutions.")
        all_solution_words = [word for solution in v for word in solution.words]
        if not all(word in words for word in all_solution_words):
            raise ValueError("All solution words must be present in 'words'.")
        return v

    class Config:
        frozen = True


class DailyConnectionsSolution(BaseModel):
    """Model for returned final category guesses"""

    category_1: tuple[str, str, str, str]
    category_2: tuple[str, str, str, str]
    category_3: tuple[str, str, str, str]
    category_4: tuple[str, str, str, str]
