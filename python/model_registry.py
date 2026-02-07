from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


def get_version(version: str) -> tuple[int, ...]:
    nums = version.split(".")
    num_version = tuple([int(n) for n in nums])
    return num_version


@dataclass
class Model:
    name: str
    version: str  # semver like "1.2.3"
    size_gb: float
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def __gt__(self, other: "Model") -> bool:
        """Compare by semantic version."""
        self_v = get_version(self.version)
        other_v = get_version(other.version)

        return self_v > other_v

    def __eq__(self, other: "Model") -> bool:
        """Compare by semantic version."""
        if self.name != other.name:
            return False

        self_v = get_version(self.version)
        other_v = get_version(other.version)
        return self_v == other_v


class ModelRegistry:
    def __init__(self):
        self._models: list[Model] = []

    def register(self, model: Model) -> None:
        """Register a model. Raise if name+version already exists."""
        for ex_model in self._models:
            if model == ex_model:
                raise KeyError
        self._models.append(model)

    def get_latest(self, name: str) -> Optional[Model]:
        """Return the highest version of a model by name."""
        highest = None
        for ex_model in self._models:
            if name == ex_model.name:
                if highest is None:
                    highest = ex_model
                elif ex_model > highest:
                    highest = ex_model

        return highest

    def search(
        self, tag: Optional[str] = None, min_size: Optional[float] = None
    ) -> list[Model]:
        """Filter models by tag and/or minimum size."""
        models = self._models
        if tag is not None:
            models = self._search_tag(models, tag)
        if min_size is not None:
            models = self._search_size(models, min_size)
        return models

    @staticmethod
    def _search_tag(models: list[Model], tag: str) -> list[Model]:
        results = []
        for ex_model in models:
            if tag in ex_model.tags:
                results.append(ex_model)
        return results

    @staticmethod
    def _search_size(models: list[Model], min_size: float) -> list[Model]:
        results = []
        for ex_model in models:
            if ex_model.size_gb >= min_size:
                results.append(ex_model)
        return results


# Usage:
registry = ModelRegistry()
registry.register(Model("mistral", "0.1.0", 14.5, ["llm", "text"]))
registry.register(Model("mistral", "0.2.0", 15.1, ["llm", "text"]))
registry.register(Model("pixtral", "1.0.0", 8.2, ["vision"]))
print(registry.get_latest("mistral"))
