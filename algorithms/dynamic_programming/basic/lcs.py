"""Longest common subsequence using dynamic programming."""
from ...base import Algorithm
from typing import List


class LongestCommonSubsequence(Algorithm):
    """Compute length of the longest common subsequence."""

    def execute(self, s1: str, s2: str) -> int:
        """Return length of LCS of ``s1`` and ``s2``.

        Time complexity: O(m * n)
        Space complexity: O(m * n)
        """
        m, n = len(s1), len(s2)
        dp: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if s1[i] == s2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
        return dp[m][n]
