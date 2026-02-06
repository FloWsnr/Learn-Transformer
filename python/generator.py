from typing import Iterable, Generator


class LogIterable:
    def __init__(self, filepath: str, level: str) -> None:
        self.file = open(filepath, "r")
        self.level = level

    def __next__(self):
        """Get the next item"""
        for line in self.file:
            if self.level in line:
                return line
        self.file.close()
        raise StopIteration

    def __iter__(self):
        return self


# Implement these:
def filter_logs(filepath: str, level: str = "ERROR") -> Generator[str]:
    """Yield log lines matching the given severity level."""
    with open(filepath, mode="r") as f:
        for line in f:
            if level in line:
                yield line


def parse_logs(log_lines: Iterable[str]):
    """Take an iterable of log strings like
    '2024-01-15 10:23:45 ERROR Database connection failed'
    and yield {"timestamp": datetime, "level": str, "message": str}"""
    for log_string in log_lines:
        log = log_string.split()
        data = {"timestamp": log[1], "level": log[2], "message": " ".join(log[3:])}
        yield data


# Should work like:
for entry in parse_logs(filter_logs("python/server.log", "ERROR")):
    print(entry)
