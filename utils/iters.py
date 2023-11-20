from itertools import combinations
from typing import Sequence, Generator


def create_combinations_from[T](
        items: Sequence,
        /,
        *,
        min_length: int = 1,
        max_length: int = None,
) -> Generator[tuple[T, ...], ..., ...]:
    length = len(items)
    max_length = max_length or length
    for cell_length in range(min_length, max_length + 1):
        for combination in combinations(items, cell_length):
            yield combination
