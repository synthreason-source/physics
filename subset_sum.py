"""
Equation-driven subset-sum solver.

This actually WIRES the image's formula into the search, instead of just
quoting it afterward:

    N(k) = 2^n * (1-r)^k          (predicted candidates remaining after k steps)

At each level k of the search, we compute what N(k) *should* be for a given
assumed constant rate r, and cap the beam (how many partial-sum branches we
keep) at that predicted size -- i.e. we literally trust the formula to tell
the search how aggressively it's allowed to prune.

If the formula's constant-r assumption is realistic for the instance, this
finds the answer using close to the promised n^c work. If the assumption is
wrong for the instance, the search prunes away the correct branch and FAILS
to find a real answer -- which is the point.
"""

import math, random


def equation_beam_search(nums, target, r, c=2):
    """
    Beam search where beam width at step k is set directly by the
    image's formula: width(k) = ceil(2^n * (1-r)^k), clipped to >=1
    and to n^c as an absolute cap once the "collapsed" regime is reached.
    """
    n = len(nums)
    cap = max(1, n ** c)

    # beam = list of (partial_sum, path) tuples, all using items[0..k-1] decided
    beam = [(0, [])]

    for k in range(1, n + 1):
        x = nums[k - 1]
        # expand: try include and exclude for every state in the beam
        expanded = []
        for partial, path in beam:
            # exclude
            expanded.append((partial, path))
            # include (only if it doesn't already overshoot)
            if partial + x <= target:
                expanded.append((partial + x, path + [x]))

        # check for a solution before pruning
        for partial, path in expanded:
            if partial == target:
                return path, k

        # ---- THE EQUATION DRIVES THE PRUNING ----
        # log-space to avoid OverflowError when 2**n is astronomically large:
        # log(predicted_width) = n*ln2 + k*ln(1-r)
        log_width = n * math.log(2) + k * math.log(1 - r)
        if log_width > math.log(cap):
            width = cap                      # already bigger than the cap; skip exponentiating
        else:
            width = max(1, min(cap, math.ceil(math.exp(log_width))))
        # keep the `width` candidates closest to target (best-first heuristic)
        expanded.sort(key=lambda item: abs(target - item[0]))
        beam = expanded[:width]

    return None, n

def run(label, nums, target, r, c=2):
    n = len(nums)
    print(f"=== {label}  (n={n}, target={target}, r={r}, c={c}) ===")

    result, steps_used = equation_beam_search(nums, target, r, c)
    print(f"equation-driven beam search: {'found ' + str(result) if result else 'FAILED to find one'}"
          f"  (ran {steps_used} steps, cap n^c={n**c})")

    print()


random.seed(1)
n = 249000

# EASY: earlier we measured real r ~= 0.44 for this kind of instance
easy_nums = [random.randint(1, 20) for _ in range(n)]
easy_target = sum(random.sample(easy_nums, 5))
run("(matches its own measured r)", easy_nums, easy_target, r=0.44)
