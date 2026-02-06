from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Model:
    name: str
    version: str  # semver like "1.2.3"
    size_gb: float
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def __gt__(self, other: "Model") -> bool:
        """Compare by semantic version."""
        ...


class ModelRegistry:
    def __init__(self): ...

    def register(self, model: Model) -> None:
        """Register a model. Raise if name+version already exists."""
        ...

    def get_latest(self, name: str) -> Optional[Model]:
        """Return the highest version of a model by name."""
        ...

    def search(
        self, tag: Optional[str] = None, min_size: Optional[float] = None
    ) -> list[Model]:
        """Filter models by tag and/or minimum size."""
        ...


# Usage:
registry = ModelRegistry()
registry.register(Model("mistral", "0.1.0", 14.5, ["llm", "text"]))
registry.register(Model("mistral", "0.2.0", 15.1, ["llm", "text"]))
registry.register(Model("pixtral", "1.0.0", 8.2, ["vision"]))
print(registry.get_latest("mistral"))
