from dataclasses import dataclass


@dataclass
class LoadResult:
    source: str
    data: dict | None
    error: str | None

    @property
    def success(self) -> bool:
        return self.error is None


def load_json_safe(filepath: str) -> LoadResult:
    """Load a JSON file, returning LoadResult instead of raising."""
    ...


def load_many(filepaths: list[str], fail_fast: bool = False) -> list[LoadResult]:
    """Load multiple files.
    If fail_fast=True, stop at first error and raise.
    If fail_fast=False, collect all results (successes and failures)."""
    ...


def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """Decorator that retries a function on specified exceptions."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs): ...

        return wrapper

    return decorator
