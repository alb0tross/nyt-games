from src.connections.models import CategoryColor, CategorySolution, DailyConnections
from src.connections.solver import ConnectionsSolver
from src.llm_client import OpenAIClient

example_connections = DailyConnections(
    date="2024-10-28",
    words=[
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
    ],
    solutions=[
        CategorySolution(
            color=CategoryColor.YELLOW,
            theme="sassy",
            words=["cute", "fresh", "smart", "wise"],
        ),
        CategorySolution(
            color=CategoryColor.GREEN,
            theme="ambience",
            words=["air", "mood", "feeling", "quality"],
        ),
        CategorySolution(
            color=CategoryColor.BLUE, theme="units", words=["bar", "bel", "lux", "mole"]
        ),
        CategorySolution(
            color=CategoryColor.PURPLE,
            theme="The Little ____",
            words=["Mermaid", "Prince", "Rascals", "Tramp"],
        ),
    ],
)


def main():
    solver = ConnectionsSolver(client=OpenAIClient())
    solver.solve(example_connections)


if __name__ == "__main__":
    main()
