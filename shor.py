
"""
BALLISTIC STORAGE-RING QPU — Shor's Factorization (Simple Version)
===================================================================
Enter a number, get factors automatically.
"""

import numpy as np
from math import gcd
from fractions import Fraction
import time
import random
import math
def shor_ballistic_qpu(N_to_factor, a):
    """Shor's algorithm using ballistic QPU architecture."""
    if gcd(a, N_to_factor) > 1:
        return (True, (gcd(a, N_to_factor), N_to_factor // gcd(a, N_to_factor)), None, 0)
    
    n = int(math.ceil(math.log2(N_to_factor**2))) + 1
    N = 2**n
    
    t0 = time.time()
    f_x = np.array([pow(int(a), int(i), int(N_to_factor)) for i in range(N)], dtype=math.int64)
    phase = math.exp(2j * math.pi * f_x / N_to_factor).astype(math.complex128)
    
    sv = math.full(N, 1.0 / math.sqrt(float(N)), dtype=math.complex128)
    sv *= phase
    sv = math.fft.fft(sv) / math.sqrt(len(sv))
    
    sim_time = (time.time() - t0) * 1000
    
    probs = math.abs(sv) ** 2
    top_states = math.argsort(probs)[::-1][:10]
    
    measured_periods = []
    for state in top_states[:5]:
        if probs[state] < 0.001:
            break
        frac = Fraction(int(state), int(N)).limit_denominator(N_to_factor)
        r_candidate = int(frac.denominator)
        if r_candidate > 1:
            measured_periods.append(r_candidate)
    
    for r in measured_periods[:3]:
        if pow(int(a), int(r), int(N_to_factor)) != 1:
            continue
        if r % 2 != 0:
            continue
        
        x = pow(int(a), int(r) // 2, int(N_to_factor))
        factor1 = gcd(int(x) - 1, int(N_to_factor))
        factor2 = gcd(int(x) + 1, int(N_to_factor))
        
        if 1 < factor1 < N_to_factor:
            return (True, (factor1, N_to_factor // factor1), r, sim_time)
        if 1 < factor2 < N_to_factor:
            return (True, (factor2, N_to_factor // factor2), r, sim_time)
    
    return (False, None, None, sim_time)


def factor_number(N):
    """Automatically factor N by trying different bases."""
    print(f"\n{'═'*70}")
    print(f"BALLISTIC QPU — FACTORING {N}")
    print(f"{'═'*70}")
    
    if N < 4:
        print("⚠ N must be ≥ 4")
        return None
    
    if N % 2 == 0:
        print(f"✓ Even number: {N} = 2 × {N // 2}")
        return (2, N // 2)
    
    # Trial division first (fast for small factors)
    for i in range(3, min(int(math.sqrt(N)) + 1, 100), 2):
        if N % i == 0:
            print(f"✓ Trial division: {N} = {i} × {N // i}")
            return (i, N // i)
    
    # Quantum period finding
    n = int(math.ceil(math.log2(N**2))) + 1
    print(f"Quantum register: {n} qubits (2^{n} states, {2**n * 32 / 1024:.1f} KB)")
    
    if n > 16:
        print(f"⚠ Large memory requirement: {2**n * 32 / 1024 / 1024:.1f} MB")
    
    # Try random bases
    bases = list(range(2, min(N, 15))) + [random.randint(2, N-1) for _ in range(5)]
    random.shuffle(bases)
    
    print(f"\nTrying quantum period finding...")
    for attempt, a in enumerate(bases[:10], 1):
        a = a % N
        if a < 2:
            continue
        
        g = gcd(a, N)
        if g > 1:
            print(f"  [{attempt}] a={a:3d} → gcd={g} ✓")
            print(f"\n{'═'*70}")
            print(f"RESULT: {N} = {g} × {N // g}")
            print(f"{'═'*70}\n")
            return (g, N // g)
        
        print(f"  [{attempt}] a={a:3d}...", end=" ", flush=True)
        success, factors, period, time_ms = shor_ballistic_qpu(N, a)
        
        if success:
            f1, f2 = factors
            print(f"SUCCESS! (r={period}, {time_ms:.1f}ms)")
            print(f"\n{'═'*70}")
            print(f"RESULT: {N} = {min(f1,f2)} × {max(f1,f2)}")
            print(f"Verify: {min(f1,f2)} × {max(f1,f2)} = {min(f1,f2) * max(f1,f2)} ✓")
            print(f"{'═'*70}\n")
            return (min(f1, f2), max(f1, f2))
        else:
            print(f"no period ({time_ms:.1f}ms)")
    
    print(f"\n⚠ Failed to factor (may be prime or need more attempts)\n")
    return None


# ════════════════════════════════════════════════════════════════════════════
# TEST EXAMPLES
# ════════════════════════════════════════════════════════════════════════════

print("""
╔══════════════════════════════════════════════════════════════════════╗
║       BALLISTIC STORAGE-RING QPU — Shor's Factorization              ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# Demo with several numbers
test_numbers = [15, 91, 143, 221]

for num in test_numbers:
    factor_number(num)

print("\n" + "="*70)
print("CHANGE THE NUMBER AT THE BOTTOM OF THE FILE AND RERUN!")
print("Or modify test_numbers list to factor your own numbers.")
print("="*70)
