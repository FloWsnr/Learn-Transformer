import time
from contextlib import contextmanager


@contextmanager
def timed(label: str = "Block"):
    """Context manager that prints how long the block took."""
    ...


# Usage: with timed("Data loading"): ...


class TempWorkspace:
    """Context manager that creates a temp directory, yields its path,
    and cleans it up on exit — even if an exception occurs."""

    def __init__(self, prefix: str = "workspace_"): ...

    def __enter__(self): ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...
        return False  # don't suppress exceptions


# Usage:
with TempWorkspace("experiment_") as ws:
    print(f"Working in {ws}")
    # create files, do work...
# directory is cleaned up here
