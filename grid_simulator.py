"""
512-Channel Optoelectronic-Thermoelectronic Layered Grid
Hardware Simulator — Factorisation Solver  v2.0
==========================================================
Physical architecture — each channel tests one candidate divisor k
against target N. All 512 channels fire in parallel per sweep.

LAYER 0 — Mach-Zehnder Optical Modulator
  A coherent laser (λ=1550 nm) is split, one arm phase-shifted by
  φ = 2π·(N mod k)/k radians (the normalised residue as a phase).
  At the output coupler the two arms interfere:
    I_out = I_0 · cos²(φ/2)
  Constructive (I_out → I_0) when φ=0, i.e. rem=0 (exact factor).
  LOCK threshold: I_out/I_0 > 0.97  (φ < ~14°).

LAYER 1 — Peltier Thermoelectric Gate
  A Peltier cell drives a ΔT proportional to the normalised residue:
    ΔT = T_max · (rem / k)        T_max = 80 K
  Junction temperature: T = T_ambient + ΔT  (T_ambient = 295 K)
  Gate OPEN when T < T_threshold = 296 K  (rem/k < 1/80 ≈ 1.25%)
  Seebeck voltage V_S = S·ΔT  displayed (S = 200 µV/K, bismuth telluride).

LAYER 2 — RF Interference Mesh (microwave standing-wave cancellation)
  A microwave tone at f = f_0·(rem/k) is injected into a resonant cavity.
  The cavity Q-factor produces a standing wave amplitude:
    A = |sin(π·rem/k)|
  When rem=0 → A=0 → full destructive cancellation → mesh CLEAR (factor).
  CLEAR threshold: A < 0.05  (within ~3° of null).

LAYER 3 — Digital Residue Comparator (output latch)
  A modular arithmetic unit computes rem = N mod k in hardware.
  Output register latches HIGH (3.3 V) when rem == 0.
  All other layers must agree for a CONFIRMED FACTOR output.
"""

import math, time, sys, threading, os, re as _re
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

def _vis(s):
    return len(_re.sub(r'\x1b\[[0-9;]*m', '', s))

# ── constants ─────────────────────────────────────────────────────────────────
CHANNELS  = 512
GRID_COLS = 32
GRID_ROWS = CHANNELS // GRID_COLS   # 16

LAYERS = [
    "MZ Optical Modulator",
    "Peltier Thermo Gate",
    "RF Interference Mesh",
    "Residue Comparator",
]

# ── physical parameters ───────────────────────────────────────────────────────
MZ_LOCK_THRESHOLD  = 0.97   # I_out/I_0 must exceed this  (φ < ~14°)
T_AMBIENT          = 295.0  # K  — room temperature cold side
T_MAX_DELTA        = 80.0   # K  — max Peltier ΔT at full residue
T_GATE_OPEN        = 296.0  # K  — gate opens below this (rem/k < 1/80)
SEEBECK_UV_PER_K   = 200.0  # µV/K  — Bi₂Te₃ Seebeck coefficient
RF_NULL_THRESHOLD  = 0.05   # standing-wave amplitude below this = CLEAR

# ── layer physics ─────────────────────────────────────────────────────────────

def layer_mz_optical(n, k):
    """
    Mach-Zehnder interferometer.
    Phase shift φ = 2π · (N mod k) / k
    Normalised output intensity I = cos²(φ/2)
    LOCK when I > MZ_LOCK_THRESHOLD  ↔  rem/k < ~2.4%
    """
    rem  = n % k
    phi  = 2 * math.pi * rem / k          # radians
    I    = math.cos(phi / 2) ** 2         # normalised intensity 0..1
    lock = I > MZ_LOCK_THRESHOLD
    phi_deg = math.degrees(phi)
    return lock, f"φ={phi_deg:7.3f}°  I={I:.4f}  {'LOCK' if lock else 'DRIFT'}"


def layer_peltier_thermo(n, k):
    """
    Peltier thermoelectric gate.
    ΔT = T_MAX_DELTA · (rem / k)
    T_junction = T_AMBIENT + ΔT
    V_Seebeck = SEEBECK_UV_PER_K · ΔT  (µV)
    Gate OPEN when T_junction < T_GATE_OPEN  (rem/k < 1/80)
    """
    rem  = n % k
    dT   = T_MAX_DELTA * (rem / k)
    T    = T_AMBIENT + dT
    Vs   = SEEBECK_UV_PER_K * dT          # µV
    gate = T < T_GATE_OPEN
    return gate, f"ΔT={dT:5.2f}K  T={T:6.2f}K  Vs={Vs:6.1f}µV  {'OPEN' if gate else 'CLOSED'}"


def layer_rf_mesh(n, k):
    """
    RF standing-wave interference mesh.
    Amplitude A = |sin(π · rem / k)|
    CLEAR (factor signal) when A < RF_NULL_THRESHOLD
    rem=0 → A=0 (perfect null) → CLEAR
    """
    rem  = n % k
    A    = abs(math.sin(math.pi * rem / k))
    clear = A < RF_NULL_THRESHOLD
    return clear, f"A={A:.4f}  {'CLEAR' if clear else 'ACTIVE'}"


def layer_residue_comparator(n, k):
    """
    Digital modular arithmetic comparator.
    Computes rem = N mod k in a carry-save adder array.
    Output latch: 3.3 V when rem == 0, else 0 V.
    """
    rem   = n % k
    latch = rem == 0
    rail  = 3.3 if latch else 0.0
    return latch, f"rem={rem:<8} V_out={rail:.1f}V  {'HIGH ✓' if latch else 'LOW'}"


LAYER_FNS = [
    layer_mz_optical,
    layer_peltier_thermo,
    layer_rf_mesh,
    layer_residue_comparator,
]

# ── channel sweep ─────────────────────────────────────────────────────────────
def sweep_channels(n, offset=0):
    results = []
    for ch in range(CHANNELS):
        k = offset + ch + 2
        if k > n:
            results.append((k, False, []))
            continue
        layer_data = [fn(n, k) for fn in LAYER_FNS]
        hit = (n % k == 0)
        results.append((k, hit, layer_data))
    return results

# ── grid renderer ─────────────────────────────────────────────────────────────
CELL_HIT  = BGR + DM + "██" + R
CELL_MISS = c("░░", D)
CELL_IDLE = c("··", D + Fore.BLACK)

def render_grid(results):
    print()
    ruler = "     "
    for col in range(0, GRID_COLS, 4):
        ruler += c(f"{col:<8}", D)
    print(ruler)
    for row in range(GRID_ROWS):
        line = c(f" {row*GRID_COLS:>3} │", D)
        for col in range(GRID_COLS):
            idx = row * GRID_COLS + col
            if idx >= len(results):
                line += CELL_IDLE
            else:
                k, hit, _ = results[idx]
                line += CELL_HIT if hit else CELL_MISS
        line += c(f"│ ch {row*GRID_COLS}–{row*GRID_COLS+GRID_COLS-1}", D)
        print(line)
    print()

# ── layer panel ───────────────────────────────────────────────────────────────
def render_layer_panel(k, layers):
    label_w = max(len(f"L{i} {LAYERS[i]}") for i in range(len(layers)))
    info_w  = max(len(info) for _, info in layers)
    w       = max(2 + 1 + label_w + 1 + 6 + 2 + info_w + 1 + 2, 56)

    def hdr(l, r, f="─"): print(c(l + f*(w-2) + r, CY))

    hdr("┌", "┐")
    title = f" Channel k={k} — Layer Analysis "
    pad   = (w - 2 - len(title)) // 2
    print(c("│" + " "*pad + title + " "*(w-2-pad-len(title)) + "│", CY+B))
    hdr("├", "┤")

    for i, (ok, info) in enumerate(layers):
        label    = f"L{i} {LAYERS[i]}"
        icon_str = "▶ PASS" if ok else "✖ FAIL"
        icon_col = GR+B    if ok else RD+B
        row = ("│ "
               + c(f"{label:<{label_w}}", WH)
               + " "
               + c(icon_str, icon_col)
               + "  "
               + c(info, D))
        pad_r = w - _vis(row) - 1
        print(row + " " * max(pad_r, 0) + "│")

    hdr("└", "┘")

# ── thermal map ───────────────────────────────────────────────────────────────
def render_thermal_map(n, results):
    print(c("  PELTIER THERMOELECTRIC — Junction Temperature Map", B+YL))
    print(c("  Hot=high ΔT (large rem), Cold=low ΔT (small rem), Green=factor\n", D))
    for row in range(GRID_ROWS):
        line = "  "
        for col in range(GRID_COLS):
            idx = row * GRID_COLS + col
            if idx >= len(results):
                line += c("·", D); continue
            k, hit, _ = results[idx]
            if hit:
                line += c("█", GR+B)
            else:
                rem   = n % k if k <= n else k
                ratio = rem / k if k else 1.0
                if   ratio < 0.013: line += c("▓", YL+B)   # ΔT < 1 K  (gate-open zone)
                elif ratio < 0.1:   line += c("▒", YL)
                elif ratio < 0.4:   line += c("░", CY)
                else:               line += c("·", BL+D)
        print(line)
    print()
    print(c("  Legend: ", D)
        + c("█", GR+B) + c(" Factor(ΔT=0)  ", D)
        + c("▓", YL+B) + c(" Gate-open zone(ΔT<1K)  ", D)
        + c("▒", YL)   + c(" Warm  ", D)
        + c("░", CY)   + c(" Cool  ", D)
        + c("·", BL+D) + c(" Cold(large rem)", D))
    print()

# ── optical phase map ─────────────────────────────────────────────────────────
PHASE_CHARS = " ·∘○◎●◉"

def render_optical_map(n, results):
    print(c("  MACH-ZEHNDER OPTICAL — Phase Map  (φ = 2π·rem/k)", B+CY))
    print(c("  Bright=constructive (small φ), dim=destructive, green=LOCK\n", D))
    for row in range(GRID_ROWS):
        line = "  "
        for col in range(GRID_COLS):
            idx = row * GRID_COLS + col
            if idx >= len(results):
                line += " "; continue
            k, hit, _ = results[idx]
            if hit:
                line += c("◉", GR+B)
            else:
                rem   = n % k if k else 0
                phi   = 2 * math.pi * rem / k if k else math.pi
                I     = math.cos(phi / 2) ** 2   # 0..1
                ch    = PHASE_CHARS[int(I * (len(PHASE_CHARS) - 1))]
                col_c = CY+B if I > 0.9 else (CY if I > 0.6 else (YL if I > 0.3 else D))
                line += c(ch, col_c)
        print(line)
    print()

# ── animated sweep ────────────────────────────────────────────────────────────
_done, _result = False, None

def _worker(n, offset):
    global _done, _result
    _result = sweep_channels(n, offset)
    _done   = True

def animated_sweep(n, offset=0):
    global _done, _result
    _done, _result = False, None
    threading.Thread(target=_worker, args=(n, offset), daemon=True).start()
    frames = "▁▂▃▄▅▆▇█▇▆▅▄▃▂▁"
    bar_w  = min(44, TW - 24)
    i = 0
    while not _done:
        prog  = i % bar_w
        bar   = c("█" * prog, GR) + c("▒" * (bar_w - prog), D)
        frame = frames[i % len(frames)]
        sys.stdout.write(
            f"\r  {c(frame, CY+B)}  Firing channels {offset}–{offset+CHANNELS-1}  [{bar}]  "
        )
        sys.stdout.flush()
        time.sleep(0.04)
        i += 1
    sys.stdout.write("\r" + " " * (TW - 1) + "\r")
    sys.stdout.flush()
    return _result

# ── main ──────────────────────────────────────────────────────────────────────
def factorise(n):
    cls()
    hline("═", col=CY+B)
    print(c(centre("512-CHANNEL OPTOELECTRONIC-THERMOELECTRONIC"), CY+B))
    print(c(centre("LAYERED GRID HARDWARE SIMULATOR  v2.0"), CY+B))
    hline("═", col=CY+B)
    print()

    if n < 2:
        print(c("  ✗  N must be ≥ 2", RD+B)); return

    sqn   = math.isqrt(n)
    limit = sqn + 1
    total = limit - 2
    nsw   = max(1, math.ceil(total / CHANNELS))

    print(c("  TARGET", B+WH) + f"  N = {c(n, B+CY)}")
    print(c(f"  √N  = {sqn}", D) + c(f"  →  testing candidates 2 … {sqn}", D))
    print()
    print(c(f"  Candidates : {c(total, YL+B)}", D))
    print(c(f"  Channels   : {c(CHANNELS, YL+B)}", D))
    print(c(f"  Sweeps     : {c(nsw, YL+B)}", D))
    print()
    print(c("  Physical layers:", B+WH))
    descs = [
        "Mach-Zehnder modulator  λ=1550 nm  lock threshold I>0.97",
        "Peltier junction  ΔT=80K·(rem/k)  gate open below T=296 K",
        "RF standing-wave cavity  null threshold A<0.05",
        "Carry-save modular comparator  3.3 V latch",
    ]
    for i, (lname, desc) in enumerate(zip(LAYERS, descs)):
        print(f"    {c(f'L{i}', B+CY)} {c(lname, WH):<24} {c(desc, D)}")
    print()
    time.sleep(0.8)

    factors_found = []

    for sw in range(nsw):
        offset = sw * CHANNELS
        hline()
        print(c(f"  SWEEP {sw+1}/{nsw}  —  candidates {offset+2}–{offset+CHANNELS+1}", B+YL))
        hline()

        results = animated_sweep(n, offset)

        # layer pass-rate bars
        print(c("  Layer pass rates:", B+WH))
        for i, lname in enumerate(LAYERS):
            active = sum(1 for _, _, ld in results if ld and ld[i][0])
            total_ch = sum(1 for _, _, ld in results if ld)
            bw = 24
            filled = int(bw * active / max(total_ch, 1))
            bar = c("█"*filled, GR) + c("░"*(bw-filled), D)
            print(f"    {c(f'L{i}',B+CY)} {c(lname,WH):<24} [{bar}] {c(active,GR)}/{c(total_ch,D)}")
        print()

        print(c("  CHANNEL GRID  ██=factor  ░░=no match", D))
        render_grid(results)

        hits = [(k, n//k) for k, hit, _ in results
                if hit and 2 <= k <= sqn]
        factors_found.extend(hits)

        if hits:
            print(c(f"  ◆ {len(hits)} factor(s) detected!", GR+B))
            for a, b in hits:
                print(f"    {c(n,B)} = {c(a,B+GR)} × {c(b,B+GR)}")
            print()

        render_thermal_map(n, results)
        render_optical_map(n, results)

        # layer panel — show factor channel, or nearest miss
        if hits:
            k = hits[0][0]
            _, _, ld = next(r for r in results if r[0] == k)
            print(c("  LAYER ANALYSIS — factor channel:", B+WH))
            render_layer_panel(k, ld)
            print()
        else:
            alive = [(k, n%k, ld) for k, hit, ld in results
                     if ld and 2 <= k <= n and not hit]
            if alive:
                alive.sort(key=lambda x: x[1])
                k, rem, ld = alive[0]
                print(c(f"  LAYER ANALYSIS — nearest miss  (rem={rem}, k={k}):", D))
                render_layer_panel(k, ld)
                print()

        if factors_found:
            break

        if sw < nsw - 1:
            input(c("  [ Enter for next sweep ] ", D))
            cls()

    # result
    hline("═", col=CY+B)
    print(c(centre("FACTORISATION RESULT"), B+WH))
    hline("═", col=CY+B)
    print()
    if factors_found:
        seen = set()
        for a, b in factors_found:
            if a not in seen:
                print(c(f"    {n}  =  {a}  ×  {b}", B+GR))
                seen.add(a)
        print()
        print(c("  STATUS", B+WH) + "  " + c(" FACTORED ✓ ", BGR+DM+B))
    else:
        print(c(f"    {n}  is  PRIME", B+MG))
        print()
        print(c("  STATUS", B+WH) + "  " + c(" PRIME — no factors exist ", BRD+WH+B))
    print()
    hline("═", col=CY+B)
    print()

# ── entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            N = int(sys.argv[1])
        except ValueError:
            print(c("Usage: python grid_simulator.py <integer>", RD)); sys.exit(1)
    else:
        print(c("512-Channel Grid Simulator  v2.0", CY+B))
        try:
            N = int(input(c("  Enter N: ", YL)).strip())
        except ValueError:
            print(c("Invalid integer.", RD)); sys.exit(1)

    factorise(N)
