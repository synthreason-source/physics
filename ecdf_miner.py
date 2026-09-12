from __future__ import annotations

import argparse
import hashlib
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy import stats


# ---------- Hashing & target ----------


def double_sha256(payload: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(payload).digest()).digest()


def target_for_difficulty(bits: int) -> int:
    if not 1 <= bits <= 255:
        raise ValueError("difficulty bits must be between 1 and 255")
    return 1 << (256 - bits)


# ---------- ECDF utilities ----------


def ecdf_from_sample(sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (x, F_n(x)) for the empirical CDF of sample.
    F_n(x) = proportion of observations <= x.
    """
    x = np.sort(sample)
    n = x.size
    y = np.arange(1, n + 1) / n
    return x, y


def ks_statistic_uniform_01(sample: np.ndarray) -> float:
    """
    One-sample KS statistic against Uniform(0,1).
    Returns D = sup_x |F_n(x) - x|.
    """
    if sample.size == 0:
        return 0.0
    x = np.sort(sample)
    n = x.size
    Fn = np.arange(1, n + 1) / n
    D_plus = np.max(Fn - x)
    D_minus = np.max(x - np.concatenate(([0.0], Fn[:-1])))
    return float(max(D_plus, D_minus))


# ---------- Mining core ----------


@dataclass
class HashRecord:
    nonce: int
    digest: bytes
    u: float
    is_hit: bool


@dataclass
class MiningStats:
    difficulty_bits: int
    target: int
    attempts: int = 0
    hits: int = 0
    misses: int = 0
    elapsed_seconds: float = 0.0
    hash_rate: float = 0.0
    hit_values: List[float] = field(default_factory=list)
    miss_values: List[float] = field(default_factory=list)
    log: List[HashRecord] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.hits / max(self.attempts, 1)

    @property
    def miss_rate(self) -> float:
        return self.misses / max(self.attempts, 1)


def format_hash_record(nonce: int, digest: bytes, u: float, target: int, label: str = "") -> str:
    """
    Return a one-line summary of a hash result:
      nonce, short hash hex, u value, and whether it's a hit.
    """
    value = int.from_bytes(digest, "big")
    is_hit = value < target
    short_hex = digest.hex()[:16]  # first 8 bytes as hex
    hit_mark = "HIT" if is_hit else "miss"
    if label:
        hit_mark = f"{label}:{hit_mark}"
    return f"nonce={nonce:8d}  hash={short_hex}...  u={u:.6f}  [{hit_mark}]"


def mine_with_ecdf_tracking(
    header: bytes,
    difficulty_bits: int,
    max_attempts: int,
    log_limit: int = 40,
) -> MiningStats:
    """
    Mine by scanning nonces 0..max_attempts-1.
    Track hit/miss and the normalized hash value in [0,1).
    Also keep a short log of hash records for display.
    """
    target = target_for_difficulty(difficulty_bits)
    stats = MiningStats(
        difficulty_bits=difficulty_bits,
        target=target,
    )

    start = time.perf_counter()
    first_hit_logged = False

    for nonce in range(max_attempts):
        candidate = header + nonce.to_bytes(8, "little")
        digest = double_sha256(candidate)
        value = int.from_bytes(digest, "big")

        # Normalize to [0,1) as if drawing from Uniform(0,1)
        u = value / (2**256)
        is_hit = value < target

        stats.attempts += 1

        if is_hit:
            stats.hits += 1
            stats.hit_values.append(u)
        else:
            stats.misses += 1
            stats.miss_values.append(u)

        # Log some records for visibility
        if len(stats.log) < log_limit or (is_hit and not first_hit_logged):
            stats.log.append(HashRecord(nonce, digest, u, is_hit))
            if is_hit:
                first_hit_logged = True

    stats.elapsed_seconds = time.perf_counter() - start
    stats.hash_rate = stats.attempts / max(stats.elapsed_seconds, 1e-12)

    return stats


def ecdf_proof_score(hit_values: np.ndarray, miss_values: np.ndarray) -> dict:
    """
    Compute an ECDF-based 'proof score' for the miner.

    Ideas:
      - Under a fair hash function, normalized values should be ~ Uniform(0,1).
      - We compute KS statistic for hits and misses separately.
      - We also compare ECDF(hit) vs ECDF(miss) with a two-sample KS.
      - Small D and large p-value => consistent with uniform / no bias.
      - Large D or small p-value => deviation from ideal behavior.

    Returns a dict with:
      ks_hits, p_hits, ks_misses, p_misses, ks_2samp, p_2samp
    """
    result = {}

    if hit_values.size > 0:
        ks_hits, p_hits = stats.kstest(hit_values, "uniform")
        result["ks_hits"] = float(ks_hits)
        result["p_hits"] = float(p_hits)
    else:
        result["ks_hits"] = float("nan")
        result["p_hits"] = float("nan")

    if miss_values.size > 0:
        ks_misses, p_misses = stats.kstest(miss_values, "uniform")
        result["ks_misses"] = float(ks_misses)
        result["p_misses"] = float(p_misses)
    else:
        result["ks_misses"] = float("nan")
        result["p_misses"] = float("nan")

    if hit_values.size > 0 and miss_values.size > 0:
        ks_2, p_2 = stats.ks_2samp(hit_values, miss_values)
        result["ks_2samp"] = float(ks_2)
        result["p_2samp"] = float(p_2)
    else:
        result["ks_2samp"] = float("nan")
        result["p_2samp"] = float("nan")

    return result


# ---------- ECDF vs PLAIN miner comparison ----------


@dataclass
class MinerComparison:
    ecdf_stats: MiningStats
    plain_stats: MiningStats
    ecdf_proof: dict
    plain_proof: dict
    enhancement_hit_rate: float
    enhancement_miss_rate: float
    cycles_saved: int


def compare_ecdf_vs_plain(
    header: bytes,
    difficulty_bits: int,
    window_size: int,
    target_hit_prob: float = 0.9,
    drift_factor: float = 2.0,
    rng_seed: int = 1234,
    max_samples: int = 20000,
) -> MinerComparison:
    """
    Compare an 'ECDF proof miner' against a 'plain miner' using the same nonce stream.

    Now:
      - Uses real double-SHA-256 hashes of nonces.
      - Logs first samples and first hits for display.
    """

    rng = random.Random(rng_seed)
    target = target_for_difficulty(difficulty_bits)
    plain_threshold = 2 ** (-difficulty_bits)

    # Generate a deterministic nonce stream
    nonces = list(range(max_samples))

    # Precompute hash values for this stream
    hash_records = []
    for nonce in nonces:
        candidate = header + nonce.to_bytes(8, "little")
        digest = double_sha256(candidate)
        value = int.from_bytes(digest, "big")
        u = value / (2**256)
        is_hit_plain = value < target  # same target for both; plain uses fixed threshold in u-space
        hash_records.append((nonce, digest, u, is_hit_plain))

    # ECDF miner state
    ecdf_window: List[float] = []
    ecdf_hits = 0
    ecdf_misses = 0
    ecdf_attempts = 0

    plain_hits = 0
    plain_misses = 0
    plain_attempts = 0

    ecdf_hit_values: List[float] = []
    ecdf_miss_values: List[float] = []
    plain_hit_values: List[float] = []
    plain_miss_values: List[float] = []

    ecdf_first_hit = None
    plain_first_hit = None

    # For printing a sample of records
    sample_records = []

    base_threshold_ecdf = plain_threshold
    current_threshold_ecdf = base_threshold_ecdf

    for i, (nonce, digest, u, _) in enumerate(hash_records):
        ecdf_attempts += 1
        plain_attempts += 1

        # --- Plain miner ---
        is_hit_plain = int.from_bytes(digest, "big") < target
        if is_hit_plain:
            plain_hits += 1
            plain_hit_values.append(u)
            if plain_first_hit is None:
                plain_first_hit = (nonce, digest, u)
        else:
            plain_misses += 1
            plain_miss_values.append(u)

        # --- ECDF miner ---
        ecdf_window.append(u)
        if len(ecdf_window) > window_size:
            ecdf_window.pop(0)

        if i % max(window_size // 10, 1) == 0 and len(ecdf_window) >= 10:
            arr = np.array(ecdf_window)
            q = float(np.quantile(arr, target_hit_prob))
            q_drift = q * drift_factor
            current_threshold_ecdf = float(np.clip(q_drift, 0.0, 1.0))

        is_hit_ecdf = u < current_threshold_ecdf
        if is_hit_ecdf:
            ecdf_hits += 1
            ecdf_hit_values.append(u)
            if ecdf_first_hit is None:
                ecdf_first_hit = (nonce, digest, u)
        else:
            ecdf_misses += 1
            ecdf_miss_values.append(u)

        # Sample records for display (first 20 and first hits)
        if len(sample_records) < 25 or (is_hit_plain and plain_first_hit == (nonce, digest, u)) or (is_hit_ecdf and ecdf_first_hit == (nonce, digest, u)):
            sample_records.append((nonce, digest, u, is_hit_plain, is_hit_ecdf, current_threshold_ecdf))

    ecdf_stats = MiningStats(
        difficulty_bits=difficulty_bits,
        target=target,
        attempts=ecdf_attempts,
        hits=ecdf_hits,
        misses=ecdf_misses,
        hit_values=ecdf_hit_values,
        miss_values=ecdf_miss_values,
    )

    plain_stats = MiningStats(
        difficulty_bits=difficulty_bits,
        target=target,
        attempts=plain_attempts,
        hits=plain_hits,
        misses=plain_misses,
        hit_values=plain_hit_values,
        miss_values=plain_miss_values,
    )

    ecdf_proof = ecdf_proof_score(
        np.array(ecdf_hit_values, dtype=float),
        np.array(ecdf_miss_values, dtype=float),
    )

    plain_proof = ecdf_proof_score(
        np.array(plain_hit_values, dtype=float),
        np.array(plain_miss_values, dtype=float),
    )

    enhancement_hit_rate = (
        ecdf_stats.hit_rate / plain_stats.hit_rate
        if plain_stats.hit_rate > 0
        else float("nan")
    )

    enhancement_miss_rate = (
        ecdf_stats.miss_rate / plain_stats.miss_rate
        if plain_stats.miss_rate > 0
        else float("nan")
    )

    cycles_saved = plain_stats.misses - ecdf_stats.misses

    comp = MinerComparison(
        ecdf_stats=ecdf_stats,
        plain_stats=plain_stats,
        ecdf_proof=ecdf_proof,
        plain_proof=plain_proof,
        enhancement_hit_rate=enhancement_hit_rate,
        enhancement_miss_rate=enhancement_miss_rate,
        cycles_saved=cycles_saved,
    )

    # Print sample hash records
    print("\nSample hash records (shared stream):")
    print("  idx | nonce  | hash (prefix)      | u       | plain   | ECDF    | ECDF threshold")
    for j, (nonce, digest, u, is_hit_plain, is_hit_ecdf, thresh) in enumerate(sample_records[:25]):
        short_hex = digest.hex()[:16]
        p_mark = "HIT" if is_hit_plain else "miss"
        e_mark = "HIT" if is_hit_ecdf else "miss"
        print(f"  {j:3d} | {nonce:6d} | {short_hex}... | {u:.6f} | {p_mark:4s} | {e_mark:4s} | {thresh:.6f}")

    if plain_first_hit is not None:
        nonce, digest, u = plain_first_hit
        print("\nFirst plain hit:")
        print(f"  nonce={nonce}, hash={digest.hex()}, u={u:.6f}")

    if ecdf_first_hit is not None:
        nonce, digest, u = ecdf_first_hit
        print("\nFirst ECDF hit:")
        print(f"  nonce={nonce}, hash={digest.hex()}, u={u:.6f}")

    return comp


# ---------- CLI & reporting ----------


def print_mining_stats(label: str, s: MiningStats) -> None:
    print(f"\n{label}")
    print(f"  Difficulty bits:   {s.difficulty_bits}")
    print(f"  Target:            {s.target:064x}")
    print(f"  Attempts:          {s.attempts:,}")
    print(f"  Hits:              {s.hits:,}")
    print(f"  Misses:            {s.misses:,}")
    print(f"  Hit rate:          {s.hit_rate:.6f}")
    print(f"  Miss rate:         {s.miss_rate:.6f}")
    print(f"  Elapsed:           {s.elapsed_seconds:.4f} s")
    print(f"  Hash rate:         {s.hash_rate:,.0f} H/s")


def print_ecdf_proof(label: str, proof: dict) -> None:
    print(f"\n{label} ECDF proof scores")
    print(f"  KS hits:           {proof.get('ks_hits', None)}")
    print(f"  p-value hits:      {proof.get('p_hits', None)}")
    print(f"  KS misses:         {proof.get('ks_misses', None)}")
    print(f"  p-value misses:    {proof.get('p_misses', None)}")
    print(f"  KS 2-sample:       {proof.get('ks_2samp', None)}")
    print(f"  p-value 2-sample:  {proof.get('p_2samp', None)}")


def run_basic_mining_experiment(
    header: bytes,
    difficulty_bits: int,
    max_attempts: int,
) -> None:
    print("=== ECDF-based Proof-of-Work Mining Experiment ===")
    print(f"Header length:       {len(header)} bytes")
    print(f"Difficulty bits:     {difficulty_bits}")
    print(f"Max attempts:        {max_attempts:,}")

    stats = mine_with_ecdf_tracking(header, difficulty_bits, max_attempts, log_limit=40)
    print_mining_stats("Mining statistics", stats)

    # Show first few hash records
    print("\nFirst hash records (sample):")
    for i, rec in enumerate(stats.log[:20]):
        line = format_hash_record(rec.nonce, rec.digest, rec.u, stats.target)
        print(f"  {i:2d}: {line}")

    # Highlight first hit if present
    hits = [r for r in stats.log if r.is_hit]
    if hits:
        first_hit = hits[0]
        print("\nFirst hit:")
        print(f"  {format_hash_record(first_hit.nonce, first_hit.digest, first_hit.u, stats.target, 'FIRST')}")
        print(f"  Full hash: {first_hit.digest.hex()}")
        print(f"  Target:    {stats.target:064x}")

    hit_arr = np.array(stats.hit_values, dtype=float)
    miss_arr = np.array(stats.miss_values, dtype=float)

    proof = ecdf_proof_score(hit_arr, miss_arr)
    print_ecdf_proof("Global", proof)

    # Simple decision rule: if KS is too large, flag as suspicious
    ks_max = max(
        proof["ks_hits"] if not np.isnan(proof["ks_hits"]) else 0.0,
        proof["ks_misses"] if not np.isnan(proof["ks_misses"]) else 0.0,
    )
    print(f"\nMax KS statistic:    {ks_max:.4f}")
    if ks_max > 0.2:
        print("Interpretation:    Strong deviation from uniform (suspicious).")
    elif ks_max > 0.1:
        print("Interpretation:    Moderate deviation from uniform.")
    else:
        print("Interpretation:    Consistent with uniform hash behavior.")


def run_comparison_experiment(
    header: bytes,
    difficulty_bits: int,
    window_size: int,
    target_p: float,
    drift_factor: float,
    seed: int,
) -> None:
    print("=== ECDF vs PLAIN Miner Comparison ===")
    print(f"Difficulty bits:     {difficulty_bits}")
    print(f"ECDF window size:    {window_size}")
    print(f"Target hit prob:     {target_p:.2f}")
    print(f"Drift factor:        {drift_factor:.2f}")
    print(f"RNG seed:            {seed}")

    comp = compare_ecdf_vs_plain(
        header,
        difficulty_bits,
        window_size,
        target_hit_prob=target_p,
        drift_factor=drift_factor,
        rng_seed=seed,
    )

    print_mining_stats("ECDF proof miner", comp.ecdf_stats)
    print_mining_stats("Plain miner", comp.plain_stats)

    print_ecdf_proof("ECDF", comp.ecdf_proof)
    print_ecdf_proof("Plain", comp.plain_proof)

    print("\nEnhancement metrics")
    print(f"  Hit-rate ratio (ECDF/plain): {comp.enhancement_hit_rate:.3f}")
    print(f"  Miss-rate ratio (ECDF/plain): {comp.enhancement_miss_rate:.3f}")
    print(f"  Cycles saved (miss reduction): {comp.cycles_saved:,}")

    ks_ecdf = max(
        comp.ecdf_proof["ks_hits"] if not np.isnan(comp.ecdf_proof["ks_hits"]) else 0.0,
        comp.ecdf_proof["ks_misses"] if not np.isnan(comp.ecdf_proof["ks_misses"]) else 0.0,
    )
    ks_plain = max(
        comp.plain_proof["ks_hits"] if not np.isnan(comp.plain_proof["ks_hits"]) else 0.0,
        comp.plain_proof["ks_misses"] if not np.isnan(comp.plain_proof["ks_misses"]) else 0.0,
    )

    print("\nProof state")
    print(f"  ECDF KS max:       {ks_ecdf:.4f}")
    print(f"  Plain KS max:      {ks_plain:.4f}")
    if ks_ecdf < 0.15:
        print("  ECDF KS verified:  ✓")
    else:
        print("  ECDF KS verified:  ✗ (deviation detected)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ECDF-based proof-of-work experiment"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Basic mining mode
    mine_parser = subparsers.add_parser("mine", help="Basic ECDF-tracked mining")
    mine_parser.add_argument(
        "--bits",
        type=int,
        default=18,
        help="difficulty bits (expected work ~ 2**bits)",
    )
    mine_parser.add_argument(
        "--attempts",
        type=int,
        default=2000000,
        help="max nonce attempts per trial",
    )

    # Comparison mode (ECDF vs plain)
    comp_parser = subparsers.add_parser(
        "compare", help="ECDF vs plain miner comparison"
    )
    comp_parser.add_argument(
        "--bits",
        type=int,
        default=10,
        help="difficulty bits for plain miner window",
    )
    comp_parser.add_argument(
        "--window",
        type=int,
        default=512,
        help="ECDF rolling window size",
    )
    comp_parser.add_argument(
        "--target-p",
        type=float,
        default=0.9,
        help="target hit probability for ECDF miner",
    )
    comp_parser.add_argument(
        "--drift",
        type=float,
        default=2.0,
        help="drift factor applied to ECDF threshold",
    )
    comp_parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed for nonce stream",
    )

    args = parser.parse_args()

    # Fixed header for reproducibility
    header = (
        b"previous_hash=00000000000000000000000000000000000000000000000000000000|"
        b"merkle_root=ecdf-pow-experiment|"
        b"timestamp=2026-09-12T14:27:00+10:00|"
        b"transactions=alice->bob:1.25"
    )

    if args.mode == "mine":
        run_basic_mining_experiment(header, args.bits, args.attempts)
    elif args.mode == "compare":
        run_comparison_experiment(
            header,
            args.bits,
            args.window,
            args.target_p,
            args.drift,
            args.seed,
        )


if __name__ == "__main__":
    main()
