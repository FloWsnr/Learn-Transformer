import time
from contextlib import contextmanager
from pathlib import Path
import shutil


@contextmanager
def timed(label: str = "Block"):
    """Context manager that prints how long the block took."""
    start_time = time.time()
    try:
        yield
    finally:
        print(f"{label} took:", time.time() - start_time)


# Usage: with timed("Data loading"): ...


class TempWorkspace:
    """Context manager that creates a temp directory, yields its path,
    and cleans it up on exit — even if an exception occurs."""

    def __init__(self, prefix: str = "workspace_"):
        self.temp_dir = Path("tmp/" + prefix)

    def __enter__(self):
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.temp_dir)
        return False  # don't suppress exceptions


# Usage:
with TempWorkspace("experiment_") as ws:
    print(f"Working in {ws}")
    # create files, do work...
# directory is cleaned up here
