"""
Subset Sum solvers.

Given a list of integers `nums` and a target `T`, decide whether some
subset sums exactly to T (and, where noted, recover that subset).

Three approaches, each with an honest complexity label:

1. brute_force        - O(2^n) time.            Always correct, simplest.
2. subset_sum_dp       - O(n * T) time/space.     Exact, but "pseudo-polynomial":
                         fast only when T isn't astronomically large.
3. meet_in_the_middle  - O(2^(n/2) * n) time.     Exact, exponential but with
                         a much smaller exponent than brute force.

None of these run in true polynomial time for all inputs -- Subset Sum
is NP-complete, so that would be a big deal (see the earlier discussion).
"""

from itertools import combinations
from bisect import bisect_left


def brute_force(nums, target):
    """Check every subset directly. O(2^n)."""
    n = len(nums)
    for r in range(n + 1):
        for combo in combinations(nums, r):
            if sum(combo) == target:
                return list(combo)
    return None


def subset_sum_dp(nums, target):
    """
    Exact DP. O(n * target) time and space (only handles non-negative ints
    and non-negative target; that's the standard formulation).

    dp[t] = a subset (as indices) that sums to t, or None if unreachable.
    We rebuild the subset via backpointers.
    """
    if target < 0:
        return None

    n = len(nums)
    # reachable[t] = index of the item used to FIRST reach sum t (for reconstruction)
    reachable = [False] * (target + 1)
    reachable[0] = True
    choice = [[-1] * (target + 1) for _ in range(n)]  # choice[i][t] = True if item i used

    prev = reachable[:]
    for i, x in enumerate(nums):
        curr = prev[:]
        if x <= target:
            for t in range(target, x - 1, -1):
                if prev[t - x] and not curr[t]:
                    curr[t] = True
                    choice[i][t] = 1
        prev = curr

    if not prev[target]:
        return None

    # Reconstruct which items were used
    t = target
    result = []
    for i in range(n - 1, -1, -1):
        if choice[i][t] == 1:
            result.append(nums[i])
            t -= nums[i]
    return result


def meet_in_the_middle(nums, target):
    """
    Split into two halves, enumerate all subset sums of each half,
    then match complementary sums. O(2^(n/2) * n).
    """
    n = len(nums)
    half = n // 2
    left, right = nums[:half], nums[half:]

    def all_sums_with_indices(arr):
        # returns list of (sum, tuple_of_items) for every subset
        out = []
        for r in range(len(arr) + 1):
            for combo in combinations(arr, r):
                out.append((sum(combo), combo))
        return out

    left_sums = all_sums_with_indices(left)
    right_sums = sorted(all_sums_with_indices(right), key=lambda p: p[0])
    right_vals = [s for s, _ in right_sums]

    for lsum, lcombo in left_sums:
        need = target - lsum
        idx = bisect_left(right_vals, need)
        if idx < len(right_vals) and right_vals[idx] == need:
            return list(lcombo) + list(right_sums[idx][1])
    return None


if __name__ == "__main__":
    nums = list(range(1, 210))
    target = 440


    for name, fn in [
        ("DP (pseudo-polynomial)", subset_sum_dp),
    ]:
        result = fn(nums, target)
        status = f"found {result} (sum={sum(result)})" if result else "no subset found"
        print(f"{name:26s}: {status}")

   
