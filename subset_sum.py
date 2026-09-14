"""
Equation-driven subset-sum solver -- memory/compute-efficient version.

Two changes from the earlier version, both needed once n gets into the
millions:

1. PATH REPRESENTATION: paths were stored as Python lists and rebuilt with
   `path + [x]` on every "include" step -- that's an O(depth) copy PER
   CANDIDATE PER STEP, so total path-copying cost alone is O(width * n^2)
   in the worst case. Fixed by using an immutable linked-list ("cons cell":
   (value, parent_node)) instead. Extending a path is now O(1): "exclude"
   reuses the same node reference, "include" allocates exactly one new
   node. The full path is only ever materialized once, at the very end,
   by walking the parent chain of the winning candidate.

2. BEAM WIDTH CAP: n**c is not a usable real-world cap once n is large
   (2_490_000**2 ~ 6.2e12 -- you cannot materialize trillions of beam
   candidates). Added an explicit `max_beam_width` that hard-caps memory
   regardless of what the formula's n^c theoretically allows -- this is
   the actual, honest cap; n^c was never enforceable as a memory bound.

Also: dedup by partial sum (many different item combinations often land
on the same partial sum -- no need to keep more than one path per sum),
and use heapq.nsmallest instead of a full sort to pick the width-best
candidates in O(m log width) instead of O(m log m).
"""

import math, random, heapq
from itertools import islice


def _materialize(node):
    """Walk an immutable path linked-list back to a plain Python list."""
    out = []
    while node is not None:
        val, node = node
        out.append(val)
    out.reverse()
    return out


def equation_beam_search(nums, target, r, c=2, max_beam_width=2000):
    """
    Beam search where beam width at step k is driven by the image's
    formula, log-space to avoid overflow, and hard-capped at
    max_beam_width for actual memory sanity.
    """
    n = len(nums)
    theoretical_cap = n ** c  # kept only for reporting; NOT used as the real cap
    cap = min(theoretical_cap, max_beam_width)

    # beam: list of (partial_sum, path_node)   path_node is None or (value, parent)
    beam = [(0, None)]

    for k in range(1, n + 1):
        x = nums[k - 1]

        # dedupe by partial sum as we expand -- keep first path reaching each sum
        seen = {}
        for partial, node in beam:
            # exclude: path unchanged, O(1), no copy
            if partial not in seen:
                seen[partial] = node
            # include
            new_partial = partial + x
            if new_partial <= target and new_partial not in seen:
                seen[new_partial] = (x, node)   # O(1) cons, no list copy

        if target in seen:
            return _materialize(seen[target]), k

        # ---- THE EQUATION DRIVES THE PRUNING (log-space, hard-capped) ----
        log_width = n * math.log(2) + k * math.log(1 - r)
        if log_width > math.log(cap):
            width = cap
        else:
            width = max(1, min(cap, math.ceil(math.exp(log_width))))

        items = seen.items()
        if len(items) <= width:
            beam = list(items)
        else:
            # O(m log width) partial selection instead of O(m log m) full sort
            beam = heapq.nsmallest(width, items, key=lambda kv: abs(target - kv[0]))

    return None, n


def measure_r(nums, sample_size=24, subset_fraction=5, seed=None):
    """
    Automatically estimate r for THIS dataset, instead of the caller
    guessing a number.

    Method: take a random sample of `sample_size` items from nums, build
    a target for that sample the same way real subset-sum targets arise
    (sum of a random subset of the sample), then run REAL branch-and-bound
    (with actual pruning, no beam-width tricks) on just that small sample.
    Its node count vs the sample's own 2^sample_size gives an honest,
    instance-specific r via r = 1 - (nodes / 2^sample_size)^(1/sample_size)
    -- same relationship as before, just automated instead of hand-fed.

    This is a PILOT MEASUREMENT, not a proof: it assumes the sample is
    representative of the full array's value distribution. If nums has
    very non-uniform structure (e.g. clustered vs spread-out regions),
    the measured r may not hold everywhere -- same caveat as always.
    """
    rng = random.Random(seed)
    sample_size = min(sample_size, len(nums))
    sample = rng.sample(nums, sample_size)
    k = max(1, min(subset_fraction, sample_size))
    sample_target = sum(rng.sample(sample, k))

    n = sample_size
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + sample[i]

    nodes = 0
    stack = [(0, 0)]
    while stack:
        i, partial = stack.pop()
        nodes += 1
        if partial == sample_target:
            break
        if i == n or partial > sample_target or partial + suffix[i] < sample_target:
            continue
        stack.append((i + 1, partial))
        stack.append((i + 1, partial + sample[i]))

    total = 2 ** n
    ratio = max(nodes / total, 1e-300)     # guard against log(0)
    r = 1 - math.exp(math.log(ratio) / n)
    r = min(max(r, 1e-6), 1 - 1e-6)        # keep strictly inside (0,1)

    return r, nodes, sample_size, sample_target


def run(label, nums, target, c=2, max_beam_width=2000, r=None, seed=None):
    n = len(nums)

    if r is None:
        r, sample_nodes, sample_n, sample_target = measure_r(nums, seed=seed)
        print(f"=== {label}  (n={n}, target={target}, c={c}, "
              f"max_beam_width={max_beam_width}) ===")
        print(f"auto-measured r: pilot search on {sample_n} sampled items "
              f"(target={sample_target}) visited {sample_nodes} nodes "
              f"-> r = {r:.4f}")
    else:
        print(f"=== {label}  (n={n}, target={target}, r={r} [manual], c={c}, "
              f"max_beam_width={max_beam_width}) ===")

    result, steps_used = equation_beam_search(nums, target, r, c, max_beam_width)
    print(f"equation-driven beam search: "
          f"{'found ' + str(result) if result else 'FAILED to find one'}"
          f"  (ran {steps_used} steps)")
    print()


if __name__ == "__main__":
    random.seed(1)
    n = 100000000

    # EASY-style: small clustered values -- r should auto-measure high
    easy_nums = [random.randint(1, 20) for _ in range(n)]
    easy_target = sum(random.sample(easy_nums, 50))
    run("large n, small clustered values", easy_nums, easy_target,
        max_beam_width=500, seed=1)

    # HARD-style: large spread-out distinct-ish values -- r should auto-measure low
    hard_nums = [random.randint(1, 10_000_000) for _ in range(n)]
    hard_target = sum(random.sample(hard_nums, 50))
    run("large n, large spread-out values", hard_nums, hard_target,
        max_beam_width=500, seed=2)
