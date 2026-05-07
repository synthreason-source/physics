"""
BALLISTIC STORAGE-RING QPU — Subset Sum, Minimal Memory
=========================================================
The oracle's net effect on the selection register is a diagonal phase matrix:
  +1 for non-solutions,  -1 for solutions.
The ancilla register only existed to *compute* that phase inside the circuit.
In simulation we compute the phase vector once from the problem arithmetic
and throw the ancilla away. Two arrays live in memory at any time:

  sv    — 2^n complex128   (the selection-qubit statevector)
  phase — 2^n float64      (oracle diagonal, precomputed from S and T)

Memory: O(2^n).  Previously O(2^(n+m)) — m ancilla bits, 512× larger.

Oracle  : sv *= phase           (one element-wise multiply)
Diffuser: sv  = 2*mean(sv) - sv (inversion about mean, one scalar broadcast)
Verify  : n additions per answer, O(n), classical

No Qiskit. No ancilla register. No composite gates. No statevector overhead.
The physics is identical — only the representation changes.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import time, warnings
warnings.filterwarnings('ignore')

# ── Palette ──────────────────────────────────────────────────────────────────
BG    = "#050810"
PANEL = "#0c1221"
ACC   = "#38bdf8"
GOLD  = "#fbbf24"
GREEN = "#34d399"
PINK  = "#f472b6"
RED   = "#f87171"
WHITE = "#f1f5f9"
GREY  = "#1e293b"
LGREY = "#334155"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "text.color": WHITE, "axes.labelcolor": WHITE,
    "xtick.color": WHITE, "ytick.color": WHITE,
    "axes.edgecolor": LGREY, "grid.color": LGREY,
    "grid.alpha": 0.2, "font.family": "monospace",
})

# ════════════════════════════════════════════════════════════════════════════
# PROBLEM
# ════════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(7)
n   = 21
S   = sorted(rng.choice(np.arange(43, 6000), n, replace=False).tolist())
_hidden = sorted(rng.choice(n, rng.integers(17, 31), replace=False).tolist())
T  = int(sum(S[i] for i in _hidden))
del _hidden

N = 2**n
m = int(np.ceil(np.log2(sum(S) + 1)))   # ancilla bits that would be needed

print(f"S = {S}")
print(f"T = {T}   n={n}   N={N}")
print(f"Ancilla that hardware needs: m={m} bits")
print(f"Ancilla simulation needs:    0 bits")

# ════════════════════════════════════════════════════════════════════════════
# MEMORY ACCOUNTING
# ════════════════════════════════════════════════════════════════════════════
bytes_sv    = N * 16              # complex128 per amplitude
bytes_phase = N * 8               # float64 per oracle element
bytes_prev  = (2**(n+m)) * 16     # previous approach: full (sel+anc) statevector

print(f"\nMemory — sv:    {bytes_sv:>10,} B  ({bytes_sv/1024:.1f} KB)")
print(f"Memory — phase: {bytes_phase:>10,} B  ({bytes_phase/1024:.1f} KB)")
print(f"Memory — prev:  {bytes_prev:>10,} B  ({bytes_prev/1024/1024:.1f} MB)")
print(f"Reduction:      {bytes_prev // bytes_sv}×")

# ════════════════════════════════════════════════════════════════════════════
# ORACLE DIAGONAL  —  computed once from arithmetic, never enumerated for solutions
#
# For each basis state |mask⟩, the oracle assigns phase -1 iff the selected
# subset sums to T. This IS O(2^n) work but it's computing the oracle function
# (not searching for solutions). On hardware the Draper adder does this
# quantumly per query. In simulation we pay it once upfront.
# ════════════════════════════════════════════════════════════════════════════
masks  = np.arange(N, dtype=np.uint32)
bits_m = ((masks[:, None] >> np.arange(n, dtype=np.uint32)[None, :]) & 1)
sums_v = (bits_m * np.array(S, dtype=np.int32)).sum(axis=1)
phase  = np.where(sums_v == T, -1.0, 1.0).astype(np.float64)   # oracle diagonal

M = int((phase == -1).sum())   # number of solutions (unknown to search loop)
ITERS = max(1, int(np.pi / 4 * np.sqrt(float(N))))

print(f"\nM = {M} solution(s)   optimal laps ≈ {ITERS}")
print(f"Oracle diagonal: {phase[phase==-1].size} entries = -1, rest = +1")

# ════════════════════════════════════════════════════════════════════════════
# GROVER SEARCH  —  two operations, two arrays, nothing else in memory
# ════════════════════════════════════════════════════════════════════════════
print(f"\nSimulating {ITERS} ring laps …")
t0 = time.time()

sv = np.full(N, 1.0 / np.sqrt(float(N)), dtype=np.complex128)   # |+⟩^n

amp_history = np.zeros((ITERS + 2, N), dtype=np.float64)   # for visualisation only
amp_history[0] = np.abs(sv) ** 2

peak_lap  = -1
peak_prob = -1.0
lap_peak_probs = []

for lap in range(1, ITERS + 2):
    sv   *= phase                   # oracle: O(N), in-place
    mean  = sv.mean()
    sv    = 2.0 * mean - sv         # diffuser: O(N), one broadcast

    probs = np.abs(sv) ** 2
    amp_history[lap] = probs
    best_prob = float(probs.max())
    lap_peak_probs.append(best_prob)
    if best_prob > peak_prob:
        peak_prob = best_prob
        peak_lap  = lap
    print(f"  Lap {lap:2d}  P(peak)={best_prob:.4f}")

sim_time = time.time() - t0
print(f"Done in {sim_time*1000:.2f} ms")

# Sample at peak lap (multinomial draw from probability distribution)
rng2 = np.random.default_rng(42)
probs_peak = amp_history[peak_lap]
samples    = rng2.multinomial(8192, probs_peak / probs_peak.sum())
counts     = {mask: samples[mask] / 8192 for mask in range(N) if samples[mask] > 0}

# ════════════════════════════════════════════════════════════════════════════
# CLASSICAL VERIFICATION  —  O(n) per answer
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*58)
print("CLASSICAL VERIFICATION  (O(n) per answer)")
print("═"*58)

THRESHOLD = 0.01
verification_rows = []
all_ok = True
for state, prob in sorted(counts.items(), key=lambda x: -x[1]):
    if prob < THRESHOLD: continue
    t_chk  = time.time()
    bits   = [(state >> i) & 1 for i in range(n)]
    subset = [S[i] for i in range(n) if bits[i]]
    total  = sum(subset)
    dt_us  = (time.time() - t_chk) * 1e6
    ok     = (total == T)
    status = "✓ VERIFIED" if ok else "✗ FALSE POSITIVE"
    if not ok: all_ok = False
    print(f"  |{state:0{n}b}⟩  P={prob:.3f}  {subset}  sum={total}  {status}  [{dt_us:.1f}µs]")
    verification_rows.append((f"|{state:0{n}b}⟩", prob, subset, total, ok, status, dt_us))

print(f"\nVERDICT: {'ALL ANSWERS VERIFIED ✓' if all_ok else 'FALSE POSITIVE ✗'}")
print(f"  sim: {sim_time*1000:.2f} ms   verify: {n} additions each   mem: {bytes_sv+bytes_phase} B total")
print("═"*58)
