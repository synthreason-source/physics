"""
512-Channel Optoelectronic-Thermoelectronic Layered Grid
Hardware Simulator — Subset Sum Solver  v4.0  [ANNEALING EDITION]
==========================================================
Architecture upgrade: replaces exhaustive parallel sweep with
Simulated Annealing — a stochastic search inspired by metallurgical
annealing. Instead of 512 channels firing in fixed order, a single
"probe" walks the subset space probabilistically, accepting bad moves
with probability exp(-ΔE / T) so it can escape local minima.

All four physical layers are retained as real-time readouts of the
current candidate state at each annealing step.

LAYER 0 — Mach-Zehnder Optical Modulator
  Phase shift φ = 2π · |sum(subset) - T| / scale
  I_out = cos²(φ/2)
  LOCK when I_out > 0.97

LAYER 1 — Peltier Thermoelectric Gate
  ΔT = T_max · |sum(subset) - T| / scale
  Gate OPEN when ΔT < 1 K

LAYER 2 — RF Interference Mesh
  A = |sin(π · |sum - T| / scale)|
  CLEAR when A < 0.05

LAYER 3 — Digital Residue Comparator
  Latches HIGH (3.3 V) when sum(subset) == T exactly.

ANNEALING:
  Temperature schedule: T(k) = T0 * α^k  (exponential cooling)
  Neighbour moves: flip a bit, swap set↔unset bit, or flip two bits
  Acceptance: ΔE ≤ 0 always accepted; ΔE > 0 accepted w/ prob exp(-ΔE/T)
"""

import math, time, sys, os, random, threading
from colorama import init, Fore, Back, Style
init(autoreset=True)

# ── palette ───────────────────────────────────────────────────────────────────
R  = Style.RESET_ALL
B  = Style.BRIGHT
D  = Style.DIM
CY = Fore.CYAN
YL = Fore.YELLOW
GR = Fore.GREEN;  BGR = Back.GREEN
RD = Fore.RED;    BRD = Back.RED
MG = Fore.MAGENTA
BL = Fore.BLUE
WH = Fore.WHITE
DM = Fore.BLACK + Style.BRIGHT
OG = Fore.YELLOW + Style.BRIGHT   # "orange" approximation

def c(*args):
    codes = args[1:]
    return "".join(codes) + str(args[0]) + R

def cls():
    os.system("cls" if sys.platform == "win32" else "clear")

try:
    TW = os.get_terminal_size().columns
except Exception:
    TW = 100

def centre(text, width=TW):
    pad = max(0, (width - len(text)) // 2)
    return " " * pad + text

def hline(ch="─", width=TW, col=D):
    print(c(ch * width, col))

# ── physical layer constants ──────────────────────────────────────────────────
MZ_LOCK_THRESHOLD  = 0.97
T_AMBIENT          = 295.0
T_MAX_DELTA        = 80.0
T_GATE_OPEN        = 296.0
SEEBECK_UV_PER_K   = 200.0
RF_NULL_THRESHOLD  = 0.05

LAYERS = [
    "MZ Optical Modulator",
    "Peltier Thermo Gate",
    "RF Interference Mesh",
    "Residue Comparator",
]

# ── annealing defaults ────────────────────────────────────────────────────────
DEFAULT_T0        = 100.0     # initial temperature
DEFAULT_ALPHA     = 0.97      # cooling rate (exponential schedule)
DEFAULT_MAX_STEPS = 8000      # max annealing iterations
DISPLAY_INTERVAL  = 200       # print status every N steps
GRID_COLS         = 32
GRID_ROWS         = 16        # 512 cells total in visit map

# ── helpers ───────────────────────────────────────────────────────────────────
def _norm(diff, scale):
    if scale == 0:
        return 0.0 if diff == 0 else 1.0
    return min(abs(diff) / scale, 1.0)

def mask_sum(elements, mask):
    return sum(v for i, v in enumerate(elements) if mask & (1 << i))

def mask_to_subset(elements, mask):
    return tuple(v for i, v in enumerate(elements) if mask & (1 << i))

def energy(elements, mask, target):
    return abs(mask_sum(elements, mask) - target)

# ── physical layers ───────────────────────────────────────────────────────────
def layer_mz_optical(diff, scale):
    dist = _norm(diff, scale)
    phi  = 2 * math.pi * dist
    I    = math.cos(phi / 2) ** 2
    ok   = I > MZ_LOCK_THRESHOLD
    return ok, f"φ={math.degrees(phi):7.2f}°  I={I:.4f}  {'LOCK' if ok else 'DRIFT'}"

def layer_peltier(diff, scale):
    dist = _norm(diff, scale)
    dT   = T_MAX_DELTA * dist
    T    = T_AMBIENT + dT
    Vs   = SEEBECK_UV_PER_K * dT
    ok   = T < T_GATE_OPEN
    return ok, f"ΔT={dT:5.2f}K  T={T:6.2f}K  Vs={Vs:6.1f}µV  {'OPEN' if ok else 'CLOSED'}"

def layer_rf_mesh(diff, scale):
    dist = _norm(diff, scale)
    A    = abs(math.sin(math.pi * dist))
    ok   = A < RF_NULL_THRESHOLD
    return ok, f"A={A:.4f}  {'CLEAR' if ok else 'ACTIVE'}"

def layer_residue(diff):
    ok   = diff == 0
    rail = 3.3 if ok else 0.0
    return ok, f"sum−T={diff:<+8}  V_out={rail:.1f}V  {'HIGH ✓' if ok else 'LOW'}"

def evaluate_layers(elements, mask, target, scale):
    diff = mask_sum(elements, mask) - target
    return [
        layer_mz_optical(diff, scale),
        layer_peltier(diff, scale),
        layer_rf_mesh(diff, scale),
        layer_residue(diff),
    ]

# ── layer panel ───────────────────────────────────────────────────────────────
def render_layer_panel(elements, mask, layers, T_anneal, step, label=""):
    subset  = mask_to_subset(elements, mask)
    s       = mask_sum(elements, mask)
    n       = len(elements)
    lw      = max(len(f"L{i} {LAYERS[i]}") for i in range(4))
    iw      = max(len(info) for _, info in layers)
    w       = max(lw + iw + 20, 70)

    def hdr(l, r, f="─"): print(c(l + f*(w-2) + r, CY))

    hdr("┌", "┐")
    title = f" Step={step}  T={T_anneal:7.3f}  mask=0b{mask:0{n}b}  sum={s} {label}"[:w-4]
    print(c("│ " + title + " "*(w-3-len(title)) + "│", CY+B))
    sub_s  = f" subset={list(subset)}"[:w-4]
    print(c("│ " + sub_s + " "*(w-3-len(sub_s)) + "│", CY))
    hdr("├", "┤")

    for i, (ok, info) in enumerate(layers):
        lbl  = f"L{i} {LAYERS[i]}"
        icon = "▶ PASS" if ok else "✖ FAIL"
        col  = GR+B if ok else RD+B
        row  = "│ " + c(f"{lbl:<{lw}}", WH) + " " + c(icon, col) + "  " + c(info, D)
        # strip ANSI for width calc
        import re
        vis = len(re.sub(r'\x1b\[[0-9;]*m', '', row))
        print(row + " " * max(w - vis - 1, 0) + "│")

    hdr("└", "┘")

# ── visit map (heat map of searched masks) ────────────────────────────────────
def render_visit_map(elements, visited_masks, solution_masks, current_mask, target, scale):
    n       = len(elements)
    total   = 1 << n
    buckets = GRID_COLS * GRID_ROWS      # 512 buckets
    bsize   = max(1, total // buckets)

    print(c("  ANNEALING SEARCH MAP  (each cell = one subset bucket)", D))
    print(c("  █=solution  ▓=hot  ▒=warm  ░=visited  ··=unvisited  ◉=current probe\n", D))

    for row in range(GRID_ROWS):
        line = "  "
        for col in range(GRID_COLS):
            idx  = row * GRID_COLS + col
            base = idx * bsize
            # representative mask for this bucket
            rep  = min(base, total - 1)
            if rep == current_mask:
                line += c("◉◉", CY+B)
            elif any(m in solution_masks for m in range(base, min(base+bsize, total))):
                line += BGR + DM + "██" + R
            elif rep in visited_masks:
                e    = energy(elements, rep, target)
                d    = _norm(e, scale)
                if   d < 0.05:  line += c("▓▓", YL+B)
                elif d < 0.2:   line += c("▒▒", YL)
                elif d < 0.5:   line += c("░░", CY)
                else:            line += c("··", D)
            else:
                line += c("··", D + Fore.BLACK)
        print(line)
    print()

# ── energy sparkline ──────────────────────────────────────────────────────────
SPARK = "▁▂▃▄▅▆▇█"

def sparkline(values, width=60):
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn or 1
    chars = [SPARK[int((v - mn) / rng * (len(SPARK)-1))] for v in values[-width:]]
    return "".join(chars)

# ── neighbour moves ───────────────────────────────────────────────────────────
def random_neighbour(mask, n):
    r = random.random()
    if r < 0.40:
        # flip one random bit
        bit = random.randrange(n)
        return (mask ^ (1 << bit)) & ((1 << n) - 1)
    elif r < 0.70:
        # swap a set bit for an unset bit (preserves |subset| size)
        set_bits   = [i for i in range(n) if mask & (1 << i)]
        unset_bits = [i for i in range(n) if not (mask & (1 << i))]
        if set_bits and unset_bits:
            rem = random.choice(set_bits)
            add = random.choice(unset_bits)
            return (mask & ~(1 << rem)) | (1 << add)
        return (mask ^ (1 << random.randrange(n))) & ((1 << n) - 1)
    else:
        # flip two random bits
        b1 = random.randrange(n)
        b2 = (b1 + random.randrange(1, n)) % n
        return (mask ^ (1 << b1) ^ (1 << b2)) & ((1 << n) - 1)

# ── temperature schedule ──────────────────────────────────────────────────────
def temperature(step, T0, alpha, schedule="exponential"):
    if schedule == "exponential":
        return T0 * (alpha ** step)
    elif schedule == "linear":
        return max(0.01, T0 * (1 - step / DEFAULT_MAX_STEPS))
    elif schedule == "logarithmic":
        return T0 / (1 + math.log(1 + step))
    return T0 * (alpha ** step)

# ── main annealer ─────────────────────────────────────────────────────────────
def solve(elements, target,
          T0=DEFAULT_T0, alpha=DEFAULT_ALPHA,
          max_steps=DEFAULT_MAX_STEPS,
          schedule="exponential"):

    cls()
    hline("═", col=CY+B)
    print(c(centre("512-CHANNEL OPTOELECTRONIC-THERMOELECTRONIC"), CY+B))
    print(c(centre("LAYERED GRID HARDWARE SIMULATOR  v4.0  [ANNEALING EDITION]"), CY+B))
    print(c(centre("── SUBSET SUM SOLVER — SIMULATED ANNEALING ──"), CY))
    hline("═", col=CY+B)
    print()

    n = len(elements)
    if n == 0:
        print(c("  ✗  Element set is empty.", RD+B)); return
    if n > 430:
        print(c(f"  ✗  Too many elements ({n}). Annealer supports up to 30.", RD+B)); return

    total  = 1 << n
    pos    = sum(x for x in elements if x > 0)
    neg    = sum(x for x in elements if x < 0)
    scale  = max(pos - neg, 1)

    print(c("  SET",    B+WH) + f"   S = {c(str(list(elements)), B+CY)}")
    print(c("  TARGET", B+WH) + f" T = {c(target, B+YL)}")
    print(c(f"  |S| = {n}", D) + c(f"   →  search space: 2^{n} = {total} subsets", D))
    print()
    print(c("  Annealing parameters:", B+WH))
    print(f"    {c('T₀',        B+CY)} = {c(T0, YL)}")
    print(f"    {c('α',         B+CY)} = {c(alpha, YL)}  ({schedule} schedule)")
    print(f"    {c('max steps', B+CY)} = {c(max_steps, YL)}")
    print(f"    {c('neighbour', B+CY)} = bit-flip / swap / double-flip")
    print()
    print(c("  Physical layers:", B+WH))
    descs = [
        "Phase lock when I > 0.97",
        "Gate open when T < 296 K",
        "RF null when A < 0.05",
        "Exact match — 3.3 V latch",
    ]
    for i, (name, desc) in enumerate(zip(LAYERS, descs)):
        print(f"    {c(f'L{i}', B+CY)} {c(name, WH):<24} {c(desc, D)}")
    print()
    time.sleep(0.8)

    # ── initialise ────────────────────────────────────────────────────────────
    current_mask  = random.randrange(total)
    current_e     = energy(elements, current_mask, target)
    best_mask     = current_mask
    best_e        = current_e

    visited       = set()
    solutions     = []          # list of masks
    energy_hist   = []
    temp_hist     = []

    accepted_total   = 0
    rejected_total   = 0
    improvements     = 0

    start_time = time.time()

    # ── annealing loop ─────────────────────────────────────────────────────────
    for step in range(max_steps):
        T_now     = temperature(step, T0, alpha, schedule)
        neighbour = random_neighbour(current_mask, n)
        e_new     = energy(elements, neighbour, target)
        delta_e   = e_new - current_e

        # acceptance criterion
        if delta_e < 0 or (T_now > 1e-9 and random.random() < math.exp(-delta_e / T_now)):
            current_mask = neighbour
            current_e    = e_new
            accepted_total += 1
            if delta_e < 0:
                improvements += 1
        else:
            rejected_total += 1

        visited.add(current_mask)

        if current_e < best_e:
            best_e    = current_e
            best_mask = current_mask

        # exact solution found
        if current_e == 0 and current_mask not in solutions:
            solutions.append(current_mask)

        energy_hist.append(current_e)
        temp_hist.append(T_now)

        # ── periodic display ──────────────────────────────────────────────────
        if step % DISPLAY_INTERVAL == 0 or step == max_steps - 1:
            cls()
            hline("═", col=CY+B)
            print(c(centre(f"ANNEALING — step {step}/{max_steps}"), B+YL))
            hline("═", col=CY+B)
            print()

            # stats row
            accept_rate = accepted_total / max(step+1, 1) * 100
            print(
                c(f"  T={T_now:8.3f}", OG if T_now > 20 else (YL if T_now > 5 else CY)) +
                c(f"  |ΔE|best={best_e}", GR+B if best_e == 0 else YL) +
                c(f"  accept={accept_rate:.1f}%", D) +
                c(f"  solutions={len(solutions)}", GR+B if solutions else D) +
                c(f"  visited={len(visited)}", D)
            )
            print()

            # energy sparkline
            sw = min(TW - 10, 80)
            hist_w = energy_hist[-sw:]
            spark  = sparkline(hist_w, sw)
            print(c("  Energy ↑", D) + c(spark, RD) + c("  (last steps)", D))

            # temperature sparkline
            spark_t = sparkline(temp_hist[-sw:], sw)
            print(c("  Temp   ↑", D) + c(spark_t, OG) + c("  (cooling curve)", D))
            print()

            # visit map
            render_visit_map(elements, visited, set(solutions), current_mask, target, scale)

            # layer panel for current state
            layers = evaluate_layers(elements, current_mask, target, scale)
            all_pass = all(ok for ok, _ in layers)
            lbl = c("[ALL PASS — EXACT SOLUTION ✓]", BGR+DM+B) if all_pass else c(f"[best energy={best_e}]", D)
            render_layer_panel(elements, current_mask, layers, T_now, step, label=lbl)
            print()

            # solutions so far
            if solutions:
                print(c(f"  ◆ Solutions found so far: {len(solutions)}", GR+B))
                for m in solutions[:5]:
                    print(f"    {c(str(list(mask_to_subset(elements, m))), B+GR)}  = {c(target, B+GR)}")
                if len(solutions) > 5:
                    print(c(f"    ... and {len(solutions)-5} more", D))
                print()

            if step < max_steps - 1:
                # brief pause between display snapshots
                elapsed = time.time() - start_time
                eta = elapsed / (step+1) * (max_steps - step - 1)
                print(c(f"  elapsed {elapsed:.1f}s  ETA {eta:.0f}s  (Ctrl-C to stop early)", D))
                time.sleep(0.05)

    # ── final result ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    cls()
    hline("═", col=CY+B)
    print(c(centre("ANNEALING COMPLETE — FINAL RESULT"), B+WH))
    hline("═", col=CY+B)
    print()
    print(c(f"  Set    : {list(elements)}", D))
    print(c(f"  Target : {target}", D))
    print()
    print(c(f"  Steps run    : {max_steps}", D))
    print(c(f"  Elapsed      : {elapsed:.2f}s", D))
    print(c(f"  Visited      : {len(visited)} unique subsets ({len(visited)/total*100:.2f}% of space)", D))
    print(c(f"  Accepted     : {accepted_total}  Rejected: {rejected_total}  Improvements: {improvements}", D))
    print(c(f"  Best energy  : {best_e}", GR+B if best_e == 0 else YL))
    print()

    # de-duplicate solutions
    unique = list(dict.fromkeys(solutions))

    if unique:
        print(c(f"  {len(unique)} exact solution(s) found:", B+GR))
        print()
        for rank, m in enumerate(unique, 1):
            sub  = mask_to_subset(elements, m)
            bits = f"0b{m:0{n}b}"
            print(f"    {c(rank, B+CY)}.  {c(str(list(sub)), B+GR)}  =  {c(target, B+GR)}")
            print(c(f"        mask={bits}  |subset|={bin(m).count('1')}", D))
        print()

        # show layer panel for best solution
        layers = evaluate_layers(elements, unique[0], target, scale)
        print(c("  LAYER ANALYSIS — solution channel:", B+WH))
        render_layer_panel(elements, unique[0], layers, 0.0, max_steps, label="[VERIFIED SOLUTION]")
        print()
        print(c("  STATUS", B+WH) + "  " + c(" SOLVED ✓ ", BGR+DM+B))

    else:
        print(c(f"  No exact solution found in {max_steps} steps.", B+MG))
        print(c(f"  Best candidate (energy={best_e}):", D))
        sub = mask_to_subset(elements, best_mask)
        s   = mask_sum(elements, best_mask)
        print(f"    {c(str(list(sub)), B+YL)}  sums to {c(s, B+YL)}  (off by {c(abs(s-target), RD+B)})")
        print()
        layers = evaluate_layers(elements, best_mask, target, scale)
        print(c("  LAYER ANALYSIS — best candidate:", B+WH))
        render_layer_panel(elements, best_mask, layers, 0.0, max_steps, label=f"[BEST, energy={best_e}]")
        print()
        print(c("  STATUS", B+WH) + "  " + c(" NO EXACT SOLUTION FOUND ", BRD+WH+B))
        print(c("  Tip: increase max_steps or T₀, or try a slower cooling rate (α closer to 1).", D))

    print()
    hline("═", col=CY+B)
    print()

# ── entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(c("512-Channel Grid Simulator  v4.0  [Annealing Edition]", CY+B))
    print(c("Stochastic subset-sum search via simulated annealing.\n", D))

    set_size = int(input("Set size (recommended 8–20): "))
    num_to_sum = int(input("Hidden subset size: "))

    elements = sorted(random.sample(range(1, set_size * 15 + 1), set_size))
    hidden   = random.sample(elements, num_to_sum)
    target   = sum(hidden)

    print(c(f"\n  [hidden subset = {hidden}  →  target = {target}]\n", D+MG))

    print(c("Annealing schedule:", D))
    print("  1. exponential  (T0 * α^k)       — recommended")
    print("  2. linear       (T0 * (1-k/N))")
    print("  3. logarithmic  (T0 / (1+ln(k)))")
    choice = input("Schedule [1/2/3, default=1]: ").strip()
    schedules = {"1": "exponential", "2": "linear", "3": "logarithmic"}
    schedule  = schedules.get(choice, "exponential")

    T0    = float(input("Initial temperature T₀ [default 100]: ").strip() or "100")
    alpha = float(input("Cooling rate α       [default 0.97]: ").strip() or "0.97")
    steps = int(input("Max steps            [default 8000]: ").strip() or "8000")

    solve(tuple(elements), target,
          T0=T0, alpha=alpha,
          max_steps=steps,
          schedule=schedule)