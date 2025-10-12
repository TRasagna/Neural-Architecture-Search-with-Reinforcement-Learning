"""Base Search Space for Neural Architecture Search.

Defines the abstract interface and common functionality for all
search spaces (CNN, Transformer, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import json


class BaseSearchSpace(ABC):
    """Abstract base class for neural architecture search spaces."""

    def __init__(self, name: str, max_layers: int = 12):
        self.name = name
        self.max_layers = max_layers
        self.operations = []
        self._constraints = {}

    @abstractmethod
    def get_operations(self) -> List[str]:
        """Return list of available operations."""
        pass

    @abstractmethod
    def validate_architecture(self, architecture: Dict) -> bool:
        """Validate that architecture specification is valid."""
        pass

    @abstractmethod
    def get_search_space_size(self) -> int:
        """Estimate total number of possible architectures."""
        pass

    @abstractmethod
    def sample_random_architecture(self) -> Dict:
        """Sample a random valid architecture."""
        pass

    def add_constraint(self, name: str, constraint_fn):
        """Add constraint function for architecture validation."""
        self._constraints[name] = constraint_fn

    def check_constraints(self, architecture: Dict) -> Tuple[bool, List[str]]:
        """Check all constraints on architecture."""
        violations = []
        for name, constraint_fn in self._constraints.items():
            if not constraint_fn(architecture):
                violations.append(name)
        return len(violations) == 0, violations

    def to_dict(self) -> Dict:
        """Convert search space to dictionary representation."""
        return {
            'name': self.name,
            'max_layers': self.max_layers,
            'operations': self.operations,
            'type': self.__class__.__name__
        }

    def save(self, filepath: str):
        """Save search space configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load search space configuration from file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        # This would need specific implementations in subclasses
        return cls.from_dict(config)
