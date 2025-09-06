from abc import ABC, abstractmethod
from typing import Any


class Algorithm(ABC):
    """Base class for all algorithms.

    Subclasses should implement the :meth:`execute` method.
    """

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Run the algorithm and return the result."""
        raise NotImplementedError
