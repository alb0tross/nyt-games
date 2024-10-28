import pydantic
import pytest
from pydantic import BaseModel

from src.connections import create_guess_model


class TestCreateGuessModel:
    @pytest.fixture
    def words(self):
        return [
            "cute",
            "fresh",
            "smart",
            "wise",
            "air",
            "mood",
            "feeling",
            "quality",
            "bar",
            "bel",
            "lux",
            "mole",
            "Mermaid",
            "Prince",
            "Rascals",
            "Tramp",
        ]

    def test_default_state(self, words):
        """Test create_guess_model in default state with no correct or incorrect guesses"""
        model = create_guess_model(
            words=words, correct_guesses=[], incorrect_guesses=[]
        )

        assert issubclass(model, BaseModel)

        schema = pydantic.json_schema.model_json_schema(model)

        # Store enum values separately and verify length
        category_enum = schema["properties"]["category_1"]["enum"]
        assert len(category_enum) == 500

        # Remove enum values for cleaner comparison
        schema["properties"]["category_1"].pop("enum")

        # Check property order (explanation should be first)
        assert list(schema["properties"].keys()) == ["explanation", "category_1"]

        # Verify category_1 is properly typed as enum
        assert schema["properties"]["category_1"]["type"] == "string"
