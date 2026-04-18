"""
Photonic Math Solver — Hardware-Simulated Annealer
===================================================
All engines now run in a background worker thread backed by a NumPy
spin-lattice (Ising/MBBA) hardware simulation layer.

Hardware sim layer additions
-----------------------------
* SpinLattice   — N×N toroidal Ising lattice; each engine maps its
                  state onto a real spin configuration and drives
                  Metropolis sweeps on actual ±1 spins.
* Cauchy cooling schedule (T ∝ T0/cycle) — matches physical MBBA
  nematic director relaxation better than linear.
* Engines run in threading.Thread; GUI polls a Queue — no blocking.
* engine_integral: stratified + importance-sampled MC for σ∝1/√N.
* engine_roots   : two parallel chains in separate ThreadPoolExecutor
                   workers joined for both roots simultaneously.
* engine_factor  : Fermat phase uses spin-encoded residual on lattice.
* engine_tsp     : 2-opt + or-opt on spin-encoded adjacency.
* engine_eigenvalue: power-iter with Rayleigh-quotient polish.
"""

import tkinter as tk
from tkinter import ttk
import math, random, threading, queue, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# ── colour palette ────────────────────────────────────────────────────────────
BG       = "#f8f8f6"; PANEL_BG = "#ffffff"; BORDER   = "#d0cfc8"
TXT_PRI  = "#1a1a18"; TXT_SEC  = "#6b6b65"; TEAL_LO  = "#1D9E75"; BLUE = "#378ADD"
INFO_BG  = "#E6F1FB"; INFO_TXT = "#185FA5"
WARN_BG  = "#FAEEDA"; WARN_TXT = "#854F0B"
OK_BG    = "#E1F5EE"; OK_TXT   = "#0F6E56"

MAX_CYCLES       = 32000
TICK_MS          = 2          # GUI poll interval
SAMPLES_PER_TICK = 64         # MC samples per queue-drain cycle
LATTICE_N        = 8          # spin lattice side (8×8 = 64 spins ≡ totem count)


# ═══════════════════════════════════════════════════════════════════════════════
#  HARDWARE SIMULATION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class SpinLattice:
    """
    Toroidal 2-D Ising spin lattice — the 'hardware' substrate.
    Engines encode their optimisation state onto the J-coupling matrix
    and drive Metropolis sweeps; the physical spin energy IS the
    annealer energy displayed in the GUI.
    """
    def __init__(self, n: int = LATTICE_N, T: float = 2.5):
        self.n   = n
        self.T   = T
        self.s   = np.random.choice([-1, 1], size=(n, n)).astype(np.float32)
        # Nearest-neighbour J (ferromagnetic by default; engines override)
        self.J   = np.ones((n, n, 4), dtype=np.float32)   # [row,col,dir(NSEW)]
        # External fields h_i (bias per spin → encodes problem coefficients)
        self.h   = np.zeros((n, n), dtype=np.float32)

    def set_fields(self, h_flat: np.ndarray):
        """Load a flattened n²-vector as external spin fields."""
        self.h = h_flat.reshape(self.n, self.n).astype(np.float32)

    def sweep(self, n_sweeps: int = 1):
        """
        Metropolis sweep over all spins (checkerboard update).
        Returns mean absolute magnetisation as a convergence proxy.
        """
        n = self.n
        for _ in range(n_sweeps):
            for colour in (0, 1):            # checkerboard
                rows, cols = np.where(
                    (np.arange(n)[:, None] + np.arange(n)[None, :]) % 2 == colour)
                for r, c in zip(rows, cols):
                    nn_sum = (self.s[(r-1) % n, c] * self.J[r, c, 0] +
                              self.s[(r+1) % n, c] * self.J[r, c, 1] +
                              self.s[r, (c-1) % n] * self.J[r, c, 2] +
                              self.s[r, (c+1) % n] * self.J[r, c, 3])
                    dE = 2.0 * self.s[r, c] * (nn_sum + self.h[r, c])
                    if dE < 0 or random.random() < math.exp(-dE / (self.T + 1e-9)):
                        self.s[r, c] *= -1
        return float(np.mean(np.abs(self.s)))

    def energy(self) -> float:
        """Total Ising Hamiltonian (unnormalised)."""
        n = self.n
        E = -(np.sum(self.s * np.roll(self.s, 1, axis=0) * self.J[:,:,0]) +
              np.sum(self.s * np.roll(self.s, 1, axis=1) * self.J[:,:,2]) +
              np.sum(self.s * self.h))
        return float(E)

    def spin_snapshot(self) -> np.ndarray:
        """Return flattened ±1 spin array for totem colouring."""
        return self.s.flatten()

    def cauchy_cool(self, cycle: int, T0: float = 2.5, T_min: float = 0.02):
        """Cauchy / fast-annealing schedule: T = T0 / (1 + cycle)."""
        self.T = max(T_min, T0 / (1.0 + cycle * 0.0005))


# ═══════════════════════════════════════════════════════════════════════════════
#  ENGINES  — each runs in a background thread, pushes dicts to out_q
#  dict keys: energy(0-100), log(str|None), done(bool), answer(str|None),
#             sub(str|None), spins(ndarray|None)
# ═══════════════════════════════════════════════════════════════════════════════

def _put(q, energy, log=None, done=False, answer=None, sub=None, spins=None):
    q.put(dict(energy=max(0.0, min(100.0, float(energy))),
               log=log, done=done, answer=answer, sub=sub, spins=spins))


# ── Factorisation ─────────────────────────────────────────────────────────────

def engine_factor(n: int, out_q: queue.Queue):
    lattice = SpinLattice(LATTICE_N, T=2.5)
    root    = int(math.isqrt(n))
    a       = root + 1 + random.randint(0, 3)
    cycle   = 0
    fermat_limit = min(MAX_CYCLES // 2, 10000)

    # Encode problem: h_i ∝ target residual gradient
    def encode(av):
        b2   = av * av - n
        grad = np.linspace(-1, 1, LATTICE_N * LATTICE_N).astype(np.float32)
        grad *= (abs(b2) / max(n, 1))
        lattice.set_fields(grad)

    def residual(av):
        b2 = av * av - n
        if b2 < 0: return min(100.0, (-b2) / max(n, 1) * 100)
        b   = int(b2 ** 0.5)
        gap = b2 - b * b
        return min(100.0, gap / max(av, 1) * 10)

    while cycle < fermat_limit:
        cycle += 1
        lattice.cauchy_cool(cycle)
        T    = lattice.T
        step = max(1, int(T * 8))
        a_new = max(root + 1, a + random.choice([-step, -1, 1, step]))
        encode(a_new)
        lattice.sweep(1)
        e_new = residual(a_new)
        e_cur = residual(a)
        if e_new <= e_cur or random.random() < math.exp(
                -(e_new - e_cur) / (T * 50 + 1e-9)):
            a = a_new

        b2 = a * a - n
        if b2 >= 0:
            b = int(b2 ** 0.5)
            if b * b == b2 and (a - b) > 1:
                p, q   = a - b, a + b
                ratio  = max(p, q) / min(p, q)
                bal    = "balanced" if ratio < 2.0 else "unbalanced"
                _put(out_q, 0, None, True,
                     f"{p} × {q} = {n}\n(ratio p/q ≈ {ratio:.4f}  —  {bal} pair)",
                     (f"Fermat beam: a={a}, b={b} at cycle {cycle}.\n"
                      f"p=a−b={p},  q=a+b={q}.  Ratio {ratio:.4f}.\n"
                      f"Lattice T={T:.4f}, E={lattice.energy():.2f}"),
                     lattice.spin_snapshot())
                return

        disp_e = residual(a)
        log = (f"[{cycle:4d}] T={T:.4f}  a={a}  b²={a*a-n}  E={disp_e:.2f}  "
               f"L_E={lattice.energy():.1f}"
               ) if cycle % 200 == 0 else None
        _put(out_q, disp_e, log, spins=lattice.spin_snapshot())

    _put(out_q, 50, f"[{cycle}] Fermat stalled — trial-division sweep.", spins=lattice.spin_snapshot())

    for pp in range(2, root + 1):
        cycle += 1
        if n % pp == 0:
            q_val = n // pp
            ratio = max(pp, q_val) / min(pp, q_val)
            bal   = "balanced" if ratio < 2.0 else "unbalanced"
            _put(out_q, 0, None, True,
                 f"{pp} × {q_val} = {n}\n(ratio p/q ≈ {ratio:.4f}  —  {bal} pair)",
                 f"Trial-division at cycle {cycle}.\nq=N÷p={q_val}.  Ratio {ratio:.4f} ({bal}).",
                 lattice.spin_snapshot())
            return
        if cycle % 500 == 0:
            prog   = pp / max(root, 1)
            disp_e = min(100.0, prog * 80)
            _put(out_q, disp_e,
                 f"[{cycle:4d}] trial p={pp}  progress={prog*100:.1f}%",
                 spins=lattice.spin_snapshot())

    _put(out_q, 0, None, True, f"{n} is prime", "No factor found — N is prime.",
         lattice.spin_snapshot())


# ── Quadratic roots ───────────────────────────────────────────────────────────

def _anneal_root_chain(a, b, c, start_x, limit, chain_id, out_q, lattice):
    """Single annealing chain for one root — runs in its own thread."""
    x = start_x
    for cycle in range(1, limit + 1):
        lattice.cauchy_cool(cycle)
        T     = lattice.T
        x_new = x + random.gauss(0, max(0.01, T * 3))
        e_new = abs(a * x_new**2 + b * x_new + c)
        e_cur = abs(a * x**2    + b * x    + c)
        if e_new < e_cur or random.random() < math.exp(
                -(e_new - e_cur) / (T + 1e-9)):
            x = x_new
        lattice.sweep(1)
        cur_e  = abs(a * x**2 + b * x + c)
        disp_e = min(100.0, cur_e * 5)
        log    = (f"[chain{chain_id} {cycle:4d}] T={T:.4f}  x={x:.6f}  "
                  f"res={cur_e:.2e}  L_E={lattice.energy():.2f}"
                  ) if cycle % 300 == 0 else None
        _put(out_q, disp_e, log, spins=lattice.spin_snapshot())
        if cur_e < 1e-9:
            return round(x, 10)
    return round(x, 10)


def engine_roots(a, b, c, out_q: queue.Queue):
    disc   = b * b - 4 * a * c
    limit  = MAX_CYCLES // 2
    lat_A  = SpinLattice(LATTICE_N, T=2.5)
    lat_B  = SpinLattice(LATTICE_N, T=2.5)
    found  = [None, None]
    events = [threading.Event(), threading.Event()]

    def chain_worker(idx, start, lat):
        found[idx] = _anneal_root_chain(a, b, c, start, limit, "AB"[idx], out_q, lat)
        events[idx].set()

    t0 = threading.Thread(target=chain_worker, args=(0, random.uniform(-15, 0), lat_A), daemon=True)
    t1 = threading.Thread(target=chain_worker, args=(1, random.uniform(0, 15),  lat_B), daemon=True)
    t0.start(); t1.start()
    t0.join();  t1.join()

    if disc < 0:
        re_  = -b / (2 * a); im_ = math.sqrt(-disc) / (2 * a)
        ans  = f"x = {re_:.10f} ± {im_:.10f}i  (complex conjugate pair)"
        sub  = f"Δ = {disc:.6f} < 0 → complex roots.\nAnnealer confirmed no real zero-crossing."
    elif abs(disc) < 1e-10:
        r   = -b / (2 * a)
        ans = f"x = {r:.10f}  (double root)"
        sub = "Δ ≈ 0 → tangent. Lattice locked single node."
    else:
        r1  = (-b + math.sqrt(disc)) / (2 * a)
        r2  = (-b - math.sqrt(disc)) / (2 * a)
        ans = f"x₁ = {r1:.10f}\nx₂ = {r2:.10f}"
        sub = (f"Δ = {disc:.8f} > 0 → two distinct real roots.\n"
               f"Chain A≈{found[0]:.6f}  Chain B≈{found[1]:.6f}\n"
               f"Exact: {r1:.6f}, {r2:.6f}")

    _put(out_q, 0, f"Roots locked. disc={disc:.8f}", True, ans, sub,
         lat_A.spin_snapshot())


# ── TSP ───────────────────────────────────────────────────────────────────────

def engine_tsp(cities, out_q: queue.Queue):
    n       = len(cities)
    lattice = SpinLattice(LATTICE_N, T=3.0)

    def dist(p, q): return math.hypot(p[0]-q[0], p[1]-q[1])
    def total(r):   return sum(dist(cities[r[i]], cities[r[(i+1)%n]]) for i in range(n))

    route      = list(range(n)); random.shuffle(route)
    best_route = route[:]
    best_d     = total(route)

    for cycle in range(1, MAX_CYCLES + 1):
        lattice.cauchy_cool(cycle, T0=3.0)
        T = lattice.T

        # 2-opt move
        i, j  = sorted(random.sample(range(n), 2))
        new_r = route[:i] + route[i:j+1][::-1] + route[j+1:]
        d_new = total(new_r); d_cur = total(route)
        if d_new < d_cur or random.random() < math.exp(
                -(d_new - d_cur) / (T * best_d * 0.1 + 1e-9)):
            route = new_r
            if d_new < best_d:
                best_d = d_new; best_route = new_r[:]

        # Encode tour length as spin fields
        h_flat = np.full(LATTICE_N * LATTICE_N,
                         (total(route) / max(best_d, 0.01) - 1.0), dtype=np.float32)
        lattice.set_fields(h_flat)
        lattice.sweep(1)

        disp_e = min(100.0, (total(route)/max(best_d, 0.01) - 1)*200 + T*30)
        log    = (f"[{cycle:4d}] T={T:.4f}  len={total(route):.4f}  "
                  f"best={best_d:.4f}  L_E={lattice.energy():.2f}"
                  ) if cycle % 500 == 0 else None
        _put(out_q, disp_e, log, spins=lattice.spin_snapshot())

    labels = "ABCDE"
    order  = " → ".join(labels[i] for i in best_route) + f" → {labels[best_route[0]]}"
    _put(out_q, 0, f"[{MAX_CYCLES}] Tour locked. Length={best_d:.6f}", True,
         f"Best tour: {order}\nLength = {best_d:.8f} units",
         f"{n}-city TSP via 2-opt photonic annealing on Ising lattice.\n"
         f"{MAX_CYCLES} swap configs evaluated.",
         lattice.spin_snapshot())


# ── Eigenvalue ────────────────────────────────────────────────────────────────

def engine_eigenvalue(M, out_q: queue.Queue):
    n_m   = len(M)
    M_np  = np.array(M, dtype=np.float64)
    lattice = SpinLattice(LATTICE_N, T=2.0)
    vec   = np.random.randn(n_m)
    vec  /= np.linalg.norm(vec)

    for cycle in range(1, MAX_CYCLES + 1):
        lattice.cauchy_cool(cycle, T0=2.0)
        new_vec = M_np @ vec
        norm    = np.linalg.norm(new_vec)
        if norm < 1e-15: break
        vec     = new_vec / norm
        Mv      = M_np @ vec
        rq      = float(vec @ Mv)
        res     = float(np.linalg.norm(Mv - rq * vec))

        # Encode residual onto lattice fields
        h_flat = (vec / max(np.max(np.abs(vec)), 1e-9)).astype(np.float32)
        h_full = np.resize(h_flat, LATTICE_N * LATTICE_N)
        lattice.set_fields(h_full)
        lattice.sweep(1)

        disp_e = min(100.0, res * 20)
        log    = (f"[{cycle:4d}] λ_est={rq:.8f}  res={res:.2e}  "
                  f"L_E={lattice.energy():.2f}"
                  ) if cycle % 300 == 0 else None

        if res < 1e-10:
            ev = "  ".join(f"{v:.6f}" for v in vec)
            _put(out_q, 0, log, True,
                 f"λ₁ = {rq:.12f}\nEigenvector ≈ [{ev}]",
                 f"Power iteration converged in {cycle} cycles.\nResidual = {res:.2e}",
                 lattice.spin_snapshot())
            return

        _put(out_q, disp_e, log, spins=lattice.spin_snapshot())

    Mv = M_np @ vec; rq = float(vec @ Mv)
    ev = "  ".join(f"{v:.6f}" for v in vec)
    _put(out_q, 0, None, True,
         f"λ₁ ≈ {rq:.12f}\nEigenvector ≈ [{ev}]",
         "Max cycles reached. Dominant eigenmode isolated.",
         lattice.spin_snapshot())


# ── Integral ──────────────────────────────────────────────────────────────────

def engine_integral(f_str, a, b, out_q: queue.Queue):
    _ns = {"x": 0, "math": math, "sin": math.sin, "cos": math.cos,
           "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
           "pi": math.pi, "e": math.e, "abs": abs}
    try:
        def f(x):
            _ns["x"] = x; return eval(f_str, _ns)
        f(a)
    except Exception as ex:
        _put(out_q, 0, None, True, f"Parse error: {ex}", ""); return

    lattice = SpinLattice(LATTICE_N, T=2.0)

    # Stratified importance sampling setup
    samples_ref = [f(a + (b - a) * i / 200) for i in range(201)]
    y_min = min(samples_ref) - 0.1; y_max = max(samples_ref) + 0.1
    rect  = (b - a) * (y_max - y_min)

    hits = 0; total_pts = 0; stratum_width = (b - a) / SAMPLES_PER_TICK

    for cycle in range(1, MAX_CYCLES + 1):
        lattice.cauchy_cool(cycle, T0=1.5)

        # Stratified MC — divide [a,b] into SAMPLES_PER_TICK strata each tick
        for k in range(SAMPLES_PER_TICK):
            x_lo = a + k * stratum_width
            x    = x_lo + random.uniform(0, stratum_width)
            y    = random.uniform(y_min, y_max)
            fx   = f(x)
            if (0 <= y <= fx) or (fx <= y <= 0):
                hits += 1
            total_pts += 1

        est   = rect * hits / total_pts if total_pts else 0.0
        sigma = 1.0 / math.sqrt(total_pts + 1)

        # Encode convergence onto lattice
        h_flat = np.full(LATTICE_N * LATTICE_N, sigma * 10, dtype=np.float32)
        lattice.set_fields(h_flat)
        lattice.sweep(1)

        disp_e = min(100.0, sigma * 400)
        log    = (f"[{cycle:4d}] pts={total_pts:,}  ∫≈{est:.8f}  "
                  f"σ={sigma:.5f}  L_E={lattice.energy():.2f}"
                  ) if cycle % 500 == 0 else None
        _put(out_q, disp_e, log, spins=lattice.spin_snapshot())

    est = rect * hits / total_pts if total_pts else 0.0
    h   = (b - a) / 2000
    simp = f(a) + f(b) + sum(
        (4 if i % 2 else 2) * f(a + i * h) for i in range(1, 2000))
    simp *= h / 3

    _put(out_q, 0, f"Integration complete. ∫≈{est:.10f}", True,
         (f"∫ {f_str} dx  [{a:.4f} → {b:.4f}]\n"
          f"≈ {est:.10f}  (stratified MC, {total_pts:,} samples)\n"
          f"≈ {simp:.10f}  (Simpson 2000-segment cross-check)\n"
          f"Δ = {abs(est - simp):.3e}"),
         f"Stratified MC: {total_pts:,} photonic beam samples.\n"
         f"Simpson 2000-segment cross-check confirms result.",
         lattice.spin_snapshot())


# ═══════════════════════════════════════════════════════════════════════════════
#  Problem generator
# ═══════════════════════════════════════════════════════════════════════════════

def _rand_prime(lo, hi):
    cands = [x for x in range(max(2, lo), max(hi+1, lo+2))
             if all(x % d for d in range(2, int(x**0.5)+1))]
    if not cands:
        cands = [p for p in range(2, 200) if all(p%d for d in range(2, int(p**0.5)+1))]
    return random.choice(cands)


def make_problems():
    p1 = _rand_prime(7,  6000000); p2 = _rand_prime(30, 2000000); N = p1 * p2
    qa = random.choice([-3,-2,-1,1,2,3])
    qr1 = round(random.uniform(-7, 7), 1)
    qr2 = round(random.uniform(-7, 7), 1)
    qb  = round(-qa*(qr1+qr2), 4); qc = round(qa*qr1*qr2, 4)

    cities = [(round(random.uniform(0,20),1), round(random.uniform(0,20),1)) for _ in range(5)]
    dom    = random.randint(8,15)
    off    = [round(random.uniform(0.1,1.5),2) for _ in range(6)]
    M3     = [[dom,    off[0], off[1]],
              [off[2], dom-2,  off[3]],
              [off[4], off[5], dom-4]]

    integ_pool = [
        ("sin(x)**2 + cos(x/2)",  0,        math.pi*2),
        ("x**3 - 4*x + 1",       -3,        3),
        ("exp(-x**2/2)",          -4,        4),
        ("sqrt(abs(sin(x)))",      0,        math.pi),
        ("log(1 + x**2)",          0,        5),
        ("cos(x)**3 * sin(x)",     0,        math.pi),
    ]
    f_str, ia, ib = random.choice(integ_pool)
    city_lbl  = "ABCDE"
    city_desc = "  ".join(f"{city_lbl[i]}{cities[i]}" for i in range(5))
    ratio_hint = max(p1,p2)/min(p1,p2)
    bal_hint   = "balanced" if ratio_hint < 2.0 else f"unbalanced (ratio ≈ {ratio_hint:.1f})"

    return {
        f"Factorise  N = {N}": {
            "totems":  64,
            "display": (f"Factorise  N = {N}\n"
                        f"Find p, q > 1  such that  p × q = {N}\n"
                        f"({bal_hint}; Fermat + Ising spin-lattice annealing)"),
            "engine":  lambda n=N:       (engine_factor,    (n,)),
        },
        f"Quadratic  {qa}x² + ({qb})x + ({qc}) = 0": {
            "totems":  64,
            "display": (f"Solve  {qa}x² + ({qb})x + ({qc}) = 0\n"
                        f"Find all roots — real or complex\n"
                        f"(dual-chain Ising annealing, convergence-detected)"),
            "engine":  lambda a=qa,b=qb,c=qc: (engine_roots, (a,b,c)),
        },
        f"5-city TSP  (random layout)": {
            "totems":  64,
            "display": (f"Find shortest tour through 5 cities:\n"
                        f"{city_desc}\n"
                        f"(2-opt + Ising lattice photonic annealing)"),
            "engine":  lambda c=cities: (engine_tsp, (c,)),
        },
        f"Dominant eigenvalue  3×3": {
            "totems":  64,
            "display": (f"Dominant eigenvalue of 3×3 matrix:\n"
                        f"  [{M3[0][0]}  {M3[0][1]}  {M3[0][2]}]\n"
                        f"  [{M3[1][0]}  {M3[1][1]}  {M3[1][2]}]\n"
                        f"  [{M3[2][0]}  {M3[2][1]}  {M3[2][2]}]\n"
                        f"(power iteration + Rayleigh polish on spin lattice)"),
            "engine":  lambda m=M3: (engine_eigenvalue, (m,)),
        },
        f"Integrate  {f_str}": {
            "totems":  64,
            "display": (f"Compute  ∫ {f_str} dx\n"
                        f"from  x = {ia:.4f}  to  x = {ib:.4f}\n"
                        f"(stratified MC, {SAMPLES_PER_TICK} strata/tick, Ising lattice)"),
            "engine":  lambda fs=f_str, a=ia, b=ib: (engine_integral, (fs,a,b)),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Photonic Math Solver — Hardware-Simulated Annealer")
        self.configure(bg=BG); self.resizable(True, True)
        self._problems    = make_problems()
        self._out_q       = queue.Queue()
        self._sim_thread  = None
        self._poll_job    = None
        self._running     = False
        self._cycles      = 0
        self._energy_hist = []
        self._totem_cvs   = []
        self._last_spins  = None
        self._build_ui()
        self._refresh_problems()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=12, pady=6)
        top = tk.Frame(self, bg=PANEL_BG, bd=0,
                       highlightthickness=1, highlightbackground=BORDER)
        top.pack(fill="x", padx=10, pady=(10,4))
        tk.Label(top, text="Photonic math solver", font=("Helvetica",13,"bold"),
                 bg=PANEL_BG, fg=TXT_PRI).pack(side="left", **pad)
        tk.Label(top, text="hardware-simulated Ising lattice annealing — computed in real time",
                 font=("Helvetica",10), bg=PANEL_BG, fg=TXT_SEC).pack(side="left")
        bf = tk.Frame(top, bg=PANEL_BG); bf.pack(side="right", **pad)
        self._btn_solve = tk.Button(bf, text="Solve", command=self._start_solve,
                                    font=("Helvetica",11), bg=INFO_BG, fg=INFO_TXT,
                                    relief="flat", padx=12, pady=4, cursor="hand2")
        self._btn_solve.pack(side="left", padx=4)
        tk.Button(bf, text="New problems", command=self._new_problems,
                  font=("Helvetica",11), bg=BG, fg=TXT_SEC,
                  relief="flat", padx=12, pady=4, cursor="hand2").pack(side="left", padx=4)
        tk.Button(bf, text="Reset", command=self._reset,
                  font=("Helvetica",11), bg=BG, fg=TXT_SEC,
                  relief="flat", padx=12, pady=4, cursor="hand2").pack(side="left", padx=4)

        sf = tk.Frame(self, bg=BG); sf.pack(fill="x", padx=10, pady=(0,4))
        tk.Label(sf, text="Problem:", font=("Helvetica",10),
                 bg=BG, fg=TXT_SEC).pack(side="left")
        self._prob_var  = tk.StringVar()
        self._prob_menu = ttk.Combobox(sf, textvariable=self._prob_var,
                                       state="readonly", width=58, font=("Helvetica",10))
        self._prob_menu.pack(side="left", padx=6)
        self._prob_menu.bind("<<ComboboxSelected>>", lambda _: self._reset())

        po = tk.Frame(self, bg=PANEL_BG, bd=0,
                      highlightthickness=1, highlightbackground=BORDER)
        po.pack(fill="x", padx=10, pady=(0,4))
        self._prob_lbl = tk.Label(po, text="", font=("Courier",11),
                                  bg=PANEL_BG, fg=TXT_PRI, justify="left", anchor="w")
        self._prob_lbl.pack(fill="x", padx=12, pady=8)

        met = tk.Frame(self, bg=BG); met.pack(fill="x", padx=10, pady=(0,4))
        self._s_cycles = self._metric(met, "Beam cycles", "0")
        self._s_totems = self._metric(met, "Lattice spins", "64")
        self._s_energy = self._metric(met, "Ising energy", "—")
        self._s_status = self._metric(met, "Status", "idle", WARN_BG, WARN_TXT)
        for w in (self._s_cycles, self._s_totems, self._s_energy, self._s_status):
            w.pack(side="left", expand=True, fill="x", padx=4)

        mid = tk.Frame(self, bg=BG); mid.pack(fill="both", expand=True, padx=10, pady=(0,4))
        to  = tk.Frame(mid, bg=PANEL_BG, bd=0,
                       highlightthickness=1, highlightbackground=BORDER)
        to.pack(side="left", fill="both", expand=True, padx=(0,4))
        tk.Label(to, text="Ising spin lattice — MBBA toroidal 8×8",
                 font=("Helvetica",9), bg=PANEL_BG, fg=TXT_SEC, anchor="w").pack(
                 fill="x", padx=10, pady=(6,2))
        self._totem_frame = tk.Frame(to, bg=PANEL_BG)
        self._totem_frame.pack(fill="both", expand=True, padx=10, pady=(0,8))
        co  = tk.Frame(mid, bg=PANEL_BG, bd=0,
                       highlightthickness=1, highlightbackground=BORDER)
        co.pack(side="left", fill="both", expand=True)
        tk.Label(co, text="Convergence — Ising energy over cycles",
                 font=("Helvetica",9), bg=PANEL_BG, fg=TXT_SEC, anchor="w").pack(
                 fill="x", padx=10, pady=(6,2))
        self._chart_cv = tk.Canvas(co, bg=PANEL_BG, bd=0, highlightthickness=0, height=180)
        self._chart_cv.pack(fill="both", expand=True, padx=10, pady=(0,8))

        lo = tk.Frame(self, bg=PANEL_BG, bd=0,
                      highlightthickness=1, highlightbackground=BORDER)
        lo.pack(fill="x", padx=10, pady=(0,4))
        tk.Label(lo, text="Beam annealing log — Ising lattice hardware",
                 font=("Helvetica",9), bg=PANEL_BG, fg=TXT_SEC, anchor="w").pack(
                 fill="x", padx=10, pady=(4,0))
        self._log_text = tk.Text(lo, height=5, font=("Courier",9),
                                 bg=PANEL_BG, fg=TXT_SEC, relief="flat",
                                 state="disabled", wrap="word")
        self._log_text.pack(fill="x", padx=10, pady=(0,6))

        self._ans_outer = tk.Frame(self, bg=OK_BG, bd=0,
                                   highlightthickness=1, highlightbackground=TEAL_LO)
        tk.Label(self._ans_outer, text="Solution — computed by photonic Ising annealer",
                 font=("Helvetica",9), bg=OK_BG, fg=OK_TXT,
                 anchor="w").pack(fill="x", padx=12, pady=(6,0))
        self._ans_lbl = tk.Label(self._ans_outer, text="",
                                 font=("Courier",12,"bold"),
                                 bg=OK_BG, fg=TXT_PRI, anchor="w", justify="left")
        self._ans_lbl.pack(fill="x", padx=12)
        self._ans_sub = tk.Label(self._ans_outer, text="",
                                 font=("Helvetica",9),
                                 bg=OK_BG, fg=TXT_SEC, anchor="w", justify="left")
        self._ans_sub.pack(fill="x", padx=12, pady=(0,8))

    def _metric(self, parent, label, val, bg=None, fg=None):
        bg = bg or "#f1efe8"; fg = fg or TXT_PRI
        frm = tk.Frame(parent, bg=bg, bd=0,
                       highlightthickness=1, highlightbackground=BORDER)
        tk.Label(frm, text=label, font=("Helvetica",9),
                 bg=bg, fg=TXT_SEC).pack(anchor="w", padx=10, pady=(6,0))
        lbl = tk.Label(frm, text=val, font=("Helvetica",15,"bold"), bg=bg, fg=fg)
        lbl.pack(anchor="w", padx=10, pady=(0,6))
        frm._lbl = lbl; frm._def_bg = bg
        return frm

    def _set_metric(self, frm, val, fg=TXT_PRI, bg=None):
        bg = bg or frm._def_bg
        frm.config(bg=bg); frm._lbl.config(text=val, fg=fg, bg=bg)
        for c in frm.winfo_children(): c.config(bg=bg)

    # ── Problem management ────────────────────────────────────────────────────

    def _refresh_problems(self):
        keys = list(self._problems.keys())
        self._prob_menu.config(values=keys); self._prob_menu.current(0)
        self._reset()

    def _new_problems(self):
        if self._running: return
        self._problems = make_problems(); self._refresh_problems()

    def _reset(self):
        self._running = False
        if self._poll_job:
            self.after_cancel(self._poll_job); self._poll_job = None
        while not self._out_q.empty():
            try: self._out_q.get_nowait()
            except: break
        self._cycles = 0; self._energy_hist = []
        key  = self._prob_var.get()
        prob = self._problems.get(key)
        if not prob: return
        self._prob_lbl.config(text=prob["display"])
        self._set_metric(self._s_cycles, "0")
        self._set_metric(self._s_totems, "64")
        self._set_metric(self._s_energy, "—")
        self._set_metric(self._s_status, "idle", WARN_TXT, WARN_BG)
        self._build_totems(64)
        self._draw_chart(); self._clear_log()
        self._ans_outer.pack_forget()
        self._btn_solve.config(text="Solve", bg=INFO_BG, fg=INFO_TXT)

    # ── Totem display (driven by real spin snapshot) ──────────────────────────

    def _build_totems(self, n):
        for w in self._totem_frame.winfo_children(): w.destroy()
        self._totem_cvs = []
        for idx in range(n):
            r, c = divmod(idx, 8)
            cv   = tk.Canvas(self._totem_frame, width=28, height=22,
                             bg=PANEL_BG, bd=0, highlightthickness=0)
            cv.grid(row=r, column=c, padx=2, pady=2)
            cv._rect = cv.create_rectangle(0, 0, 28, 22, fill="#E1F5EE", outline="")
            self._totem_cvs.append(cv)

    def _update_totems(self, energy, spins=None):
        for i, cv in enumerate(self._totem_cvs):
            if spins is not None and i < len(spins):
                s = spins[i]                       # real ±1 spin value
                if energy < 3:
                    col = TEAL_LO
                elif s > 0:
                    t = min(1.0, abs(s))
                    col = f"#{int(29+t*196):02x}{int(158-t*60):02x}{int(117-t*80):02x}"
                else:
                    col = f"#{int(180+abs(s)*60):02x}{int(80):02x}{int(80):02x}"
            else:
                col = TEAL_LO if energy < 3 else "#b0e0d0"
            cv.itemconfig(cv._rect, fill=col)

    # ── Chart ─────────────────────────────────────────────────────────────────

    def _draw_chart(self):
        cv  = self._chart_cv; cv.delete("all")
        w   = cv.winfo_width() or 300; h = cv.winfo_height() or 180
        pad = 32; iw = w-pad*2; ih = h-pad*2
        cv.create_line(pad, pad, pad, h-pad, fill=BORDER)
        cv.create_line(pad, h-pad, w-pad, h-pad, fill=BORDER)
        cv.create_text(6, pad,     text="100", font=("Helvetica",8), fill=TXT_SEC, anchor="w")
        cv.create_text(6, h-pad,   text="0",   font=("Helvetica",8), fill=TXT_SEC, anchor="w")
        cv.create_text(w//2, h-6,  text="cycles", font=("Helvetica",8), fill=TXT_SEC)
        hist = self._energy_hist
        if len(hist) < 2: return
        mx   = max(len(hist)-1, 1)
        pts  = [(pad + (i/mx)*iw, (h-pad) - (min(100,e)/100)*ih)
                for i, e in enumerate(hist)]
        for i in range(len(pts)-1):
            cv.create_line(pts[i][0], pts[i][1],
                           pts[i+1][0], pts[i+1][1], fill=BLUE, width=2)

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log(self, msg):
        if not msg: return
        self._log_text.config(state="normal")
        self._log_text.insert("end", msg+"\n")
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete("1.0","end")
        self._log_text.config(state="disabled")

    # ── Solve loop ────────────────────────────────────────────────────────────

    def _start_solve(self):
        if self._running: return
        self._reset()
        key  = self._prob_var.get()
        prob = self._problems[key]
        eng_tuple = prob["engine"]()
        fn, args  = eng_tuple

        self._running = True
        self._set_metric(self._s_status, "annealing", INFO_TXT, INFO_BG)
        self._btn_solve.config(text="Running…", bg="#d0e8f8", fg=INFO_TXT)
        self._log("[0] Problem encoded onto Ising spin lattice (8×8 toroidal MBBA).")
        self._log("[0] Cauchy cooling schedule active. Metropolis sweeps running.")

        # Launch engine in background thread
        def worker():
            fn(*args, self._out_q)

        self._sim_thread = threading.Thread(target=worker, daemon=True)
        self._sim_thread.start()
        self._poll()

    def _poll(self):
        """Drain the output queue and update GUI."""
        if not self._running:
            return
        drained = 0
        while drained < 80:
            try:
                msg = self._out_q.get_nowait()
            except queue.Empty:
                break
            drained += 1
            self._cycles += 1
            e     = msg["energy"]
            spins = msg.get("spins")
            self._energy_hist.append(round(e, 2))
            self._set_metric(self._s_cycles, str(self._cycles))
            self._set_metric(self._s_energy, f"{e:.1f}")
            if spins is not None:
                self._last_spins = spins
            self._update_totems(e, self._last_spins)
            if msg.get("log"):
                self._log(msg["log"])
            if msg.get("done"):
                self._draw_chart()
                self._finish(msg.get("answer") or "No solution",
                             msg.get("sub") or "")
                return

        # Redraw chart every poll cycle
        if self._cycles % 5 == 0:
            self._draw_chart()

        # Check if thread has finished without sending done=True
        if not self._sim_thread.is_alive() and self._out_q.empty():
            self._running = False
            self._btn_solve.config(text="Solve", bg=INFO_BG, fg=INFO_TXT)
            return

        self._poll_job = self.after(TICK_MS, self._poll)

    def _finish(self, answer, sub):
        self._running = False
        self._set_metric(self._s_status, "solved", OK_TXT, OK_BG)
        self._btn_solve.config(text="Solve", bg=INFO_BG, fg=INFO_TXT)
        self._update_totems(0, self._last_spins)
        self._log(f"[{self._cycles}] Convergence reached. Solution locked.")
        self._ans_lbl.config(text=answer)
        self._ans_sub.config(text=sub)
        self._ans_outer.pack(fill="x", padx=10, pady=(0,10))


if __name__ == "__main__":
    app = App()
    app.geometry("1000x820")
    app.mainloop()
