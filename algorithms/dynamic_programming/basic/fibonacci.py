"""Fibonacci number using dynamic programming."""
from ...base import Algorithm


class Fibonacci(Algorithm):
    """Compute the n-th Fibonacci number."""

    def execute(self, n: int) -> int:
        """Return the n-th Fibonacci number.

        Time complexity: O(n)
        Space complexity: O(1)
        """
        if n <= 1:
            return n
        prev, curr = 0, 1
        for _ in range(2, n + 1):
            prev, curr = curr, prev + curr
        return curr
