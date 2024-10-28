from enum import StrEnum

from pydantic import BaseModel, Field


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

    def validate_unique_words(self) -> bool:
        """Validate that all words in the puzzle are unique"""
        return len(set(self.words)) == 16

    def validate_solutions(self) -> bool:
        """Validate that all solution words are present in the words list"""
        all_solution_words = [
            word for solution in self.solutions for word in solution.words
        ]
        return all(word in self.words for word in all_solution_words)


class DailyConnectionsSolution(BaseModel):
    """Model for returned final category guesses"""

    category_1: tuple[str, str, str, str]
    category_2: tuple[str, str, str, str]
    category_3: tuple[str, str, str, str]
    category_4: tuple[str, str, str, str]
