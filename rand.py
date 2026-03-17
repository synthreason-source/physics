
import numpy as np
from scipy.stats import entropy as scipy_entropy
import zlib

# ─── Config ───────────────────────────────────────────────────────────────────
N       = 1024
N_QUARTER = N // 4
BINS    = 64
TRIALS  = 200
np.random.seed(42)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def shannon_entropy(signal, bins=BINS):
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]; hist /= hist.sum()
    return scipy_entropy(hist, base=2)

def kolmogorov_proxy(signal):
    """Compression ratio as proxy for Kolmogorov complexity."""
    raw   = (signal * 1000).astype(np.int16).tobytes()
    comp  = zlib.compress(raw, level=9)
    return len(comp) / len(raw)  # <1 = compressible = low entropy

def mutual_information(x, y, bins=32):
    """Estimated mutual information via joint histogram."""
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
    c_xy = c_xy / c_xy.sum()
    c_x  = c_xy.sum(axis=1); c_y = c_xy.sum(axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if c_xy[i,j] > 0 and c_x[i] > 0 and c_y[j] > 0:
                mi += c_xy[i,j] * np.log2(c_xy[i,j] / (c_x[i] * c_y[j]))
    return mi

def phase_space_density(signal, bins=32):
    """Delay-embedding density spread — higher = more diffuse (more random)."""
    x, y = signal[:-1], signal[1:]
    H, _, _ = np.histogram2d(x, y, bins=bins)
    H = H / H.sum()
    H = H[H > 0]
    return scipy_entropy(H, base=2)  # high = diffuse cloud

def upsample(sub, target_len):
    x_sub  = np.linspace(0, 1, len(sub))
    x_full = np.linspace(0, 1, target_len)
    return np.interp(x_full, x_sub, sub)

def normalise(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else x

# ─── Build systems ────────────────────────────────────────────────────────────
t = np.linspace(0, 4 * np.pi, N)
det_signal = normalise(np.sin(t) + 0.3 * np.sin(3*t) + 0.1 * np.sin(5*t))

rand_full    = np.random.uniform(0, 1, N)
rand_quarter = np.random.uniform(0, 1, N_QUARTER)
rand_q_up    = upsample(rand_quarter, N)  # best-case upsampled reconstruction

# ─── Single-system metrics ────────────────────────────────────────────────────
print("=" * 58)
print("  SYSTEM ENTROPY PROFILE")
print("=" * 58)
metrics = {
    "Deterministic  (N)"    : det_signal,
    "Random Full    (N)"    : rand_full,
    "Random Quarter (N/4)"  : rand_quarter,
    "Random Q Upsmp (N/4→N)": rand_q_up,
}
for name, sig in metrics.items():
    H   = shannon_entropy(sig)
    K   = kolmogorov_proxy(sig)
    psd = phase_space_density(sig)
    print(f"  {name}")
    print(f"    Shannon H       : {H:.3f} bits")
    print(f"    Kolmogorov proxy: {K:.3f}  (lower=more ordered)")
    print(f"    Phase-space PSD : {psd:.3f} bits (lower=tighter attractor)")
    print()

# ─── Cross-system capacity tests ─────────────────────────────────────────────
print("=" * 58)
print("  CAPACITY / CORRELATION TESTS")
print("=" * 58)

corr_full  = np.corrcoef(det_signal, rand_full)[0,1]
corr_qup   = np.corrcoef(det_signal, rand_q_up)[0,1]
mi_full    = mutual_information(det_signal, rand_full)
mi_qup     = mutual_information(det_signal, rand_q_up)

print(f"  Pearson correlation  | Det vs Rand(N)  : {corr_full:+.4f}")
print(f"  Pearson correlation  | Det vs Rand(N/4): {corr_qup:+.4f}")
print(f"  Mutual Information   | Det vs Rand(N)  : {mi_full:.4f} bits")
print(f"  Mutual Information   | Det vs Rand(N/4): {mi_qup:.4f} bits")
print()

# ─── Monte Carlo: can ANY N/4 random draw ever hold the deterministic set? ───
print("=" * 58)
print(f"  MONTE CARLO ({TRIALS} trials) — best correlation N/4 can achieve")
print("=" * 58)
best_corr = -99; best_mi = -99; corrs = []
for _ in range(TRIALS):
    sub = np.random.uniform(0, 1, N_QUARTER)
    up  = upsample(sub, N)
    c   = np.corrcoef(det_signal, up)[0,1]
    corrs.append(c)
    if c > best_corr:
        best_corr = c
        best_sub  = sub.copy()
    mi = mutual_information(det_signal, up)
    if mi > best_mi:
        best_mi = mi

corrs = np.array(corrs)
print(f"  Best correlation     : {best_corr:+.4f}")
print(f"  Mean correlation     : {corrs.mean():+.4f}")
print(f"  Std correlation      : {corrs.std():.4f}")
print(f"  Best mutual info     : {best_mi:.4f} bits")
print(f"  % trials |r| < 0.1  : {(np.abs(corrs) < 0.1).mean()*100:.1f}%")
print()

# ─── Constraint dimensionality test ──────────────────────────────────────────
print("=" * 58)
print("  CONSTRAINT DIMENSIONALITY (PCA effective rank)")
print("=" * 58)
def effective_rank(signal, embed_dim=8):
    """Hankel delay-embedding + SVD to count active dimensions."""
    L = len(signal) - embed_dim + 1
    H_mat = np.array([signal[i:i+L] for i in range(embed_dim)])
    _, S, _ = np.linalg.svd(H_mat, full_matrices=False)
    S = S / S.sum()
    return np.exp(-np.sum(S * np.log(S + 1e-12)))  # entropy-based eff rank

er_det  = effective_rank(det_signal)
er_rand = effective_rank(rand_full)
er_qup  = effective_rank(rand_q_up)
print(f"  Effective rank | Deterministic  : {er_det:.2f}")
print(f"  Effective rank | Random Full    : {er_rand:.2f}")
print(f"  Effective rank | Random Quarter : {er_qup:.2f}")
print()

# ─── Verdict ─────────────────────────────────────────────────────────────────
print("=" * 58)
print("  VERDICT")
print("=" * 58)
if best_corr < 0.3:
    print("  [FAIL] N/4 random substrate CANNOT hold deterministic set.")
    print("  Even best-of-200 trials yields negligible correlation.")
    print("  Structural incompatibility confirmed — not a size issue alone.")
else:
    print("  [WARN] Marginal embedding possible — review structure.")
print("=" * 58)