"""Small local task suite with a deterministic automatic checker."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    quantity: int
    unit_price: int

    @property
    def expected(self):
        return {"total": self.quantity * self.unit_price}

    def check(self, answer: dict) -> tuple[bool, str]:
        passed = answer == self.expected
        return passed, "answer matches expected total" if passed else "incorrect total or shape"


TASKS = tuple(
    Task(f"purchase-{i:02}", f"Find the total cost of {q} items at {p} each.", q, p)
    for i, (q, p) in enumerate(
        [(2, 10), (3, 12), (4, 7), (5, 15), (6, 9), (7, 11), (8, 13), (9, 6)], start=1
    )
)
