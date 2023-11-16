from itertools import combinations
from typing import Sequence, Generator


def create_combinations_from[T](items: Sequence) -> Generator[tuple[T, ...], ..., ...]:
    length = len(items)
    for cell_length in range(1, length + 1):
        for combination in combinations(items, cell_length):
            yield combination
