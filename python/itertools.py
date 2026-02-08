import itertools
import functools
from typing import Generator


def batch(iterable, size: int) -> Generator[tuple]:
    """Yield successive chunks of `size` from any iterable.
    Last batch may be shorter. Must work with generators (no len())."""

    batched = itertools.batched(iterable, n=size)
    for b in batched:
        yield b


def sliding_window(iterable, window_size: int):
    """Yield overlapping tuples of `window_size` from iterable.
    e.g. sliding_window([1,2,3,4,5], 3) -> (1,2,3), (2,3,4), (3,4,5)"""
    slices = itertools.islice(iterable)


@functools.lru_cache(maxsize=128)
def expensive_tokenize(text: str) -> tuple[str, ...]:
    """Simulate an expensive tokenization. Return tuple (hashable for cache)."""
    time.sleep(0.1)  # simulate cost
    return tuple(text.lower().split())


# Bonus: implement your own LRU cache decorator from scratch
def my_lru_cache(maxsize: int = 128):
    def decorator(func): ...

    return decorator
