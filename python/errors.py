from dataclasses import dataclass
import json
import time
import functools


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

    try:
        with open(filepath) as f:
            data = json.load(f)
            error = None
    except Exception as e:
        data = None
        error = str(e)

    return LoadResult(source=filepath, data=data, error=error)


def load_many(filepaths: list[str], fail_fast: bool = False) -> list[LoadResult]:
    """Load multiple files.
    If fail_fast=True, stop at first error and raise.
    If fail_fast=False, collect all results (successes and failures)."""
    results = []
    for p in filepaths:
        res = load_json_safe(p)
        if fail_fast:
            if res.error is not None:
                raise Exception
        results.append(res)
    return results


def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """Decorator that retries a function on specified exceptions."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    time.sleep(delay)
            raise Exception

        return wrapper

    return decorator
