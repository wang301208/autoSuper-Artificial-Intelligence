"""Template module for algorithm implementations."""

from .base import Algorithm


class TemplateAlgorithm(Algorithm):
    """Example implementation demonstrating naming conventions."""

    def execute(self, data):
        """Return the sum of a list of numbers.

        Args:
            data: List of numbers to process.

        Returns:
            int | float: Sum of the provided numbers.

        Raises:
            TypeError: If ``data`` is not a list.
            ValueError: If any element is not numeric.
        """
        if not isinstance(data, list):
            raise TypeError("data must be a list")
        if not all(isinstance(x, (int, float)) for x in data):
            raise ValueError("all elements must be numeric")
        return sum(data)
