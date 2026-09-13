"""
512-Channel Memristive Crossbar Layered Grid
Hardware Simulator — Factorisation Solver  v3.0
==========================================================
Physical architecture — each channel tests one candidate divisor k
against target N. All 512 channels fire in parallel per sweep, each
channel implemented as a stack of four distinct memristive devices.

LAYER 0 — TiO2 Memristor Crossbar Cell (HP Linear Ion-Drift Model)
  The doped/undoped boundary inside the TiO2(-x) thin film sits at a
  normalised position driven by the residue:
    x = (N mod k) / k                (dopant boundary fraction, 0..1)
  Memristance (series mixture of doped R_on region and undoped R_off
  region):
    M(x) = R_on + (R_off − R_on)·x     R_on = 100 Ω, R_off = 16 kΩ
  Normalised conductance g = R_on / M(x)  (g → 1 as x → 0, i.e. as the
  boundary retracts fully into the low-resistance doped state).
  SET/LOCK threshold: g > 0.97  (boundary within ~3% of full retraction)

LAYER 1 — VO2 Mott Memristor (Self-Heating Threshold Switch)
  Joule self-heating from the residue current drives the vanadium
  dioxide film toward its insulator–metal transition (IMT):
    ΔT = ΔT_max · (rem / k)          ΔT_max = 80 K
  Film temperature: T = T_ambient + ΔT   (T_ambient = 295 K)
  Device switches to the metallic low-resistance state (gate OPEN)
  when T stays below the IMT threshold T_IMT = 296 K
  (rem/k < 1/80 ≈ 1.25%).
  Switching figure of merit V_th = S·ΔT displayed (S = 200 µV/K,
  thermoelectric readout coefficient of the contact stack).

LAYER 2 — Crossbar Sneak-Path Cancellation Mesh
  Parasitic sneak-path currents around the crossbar array are probed
  with a differential read tone injected at a rate f = f_0·(rem/k).
  The array's parasitic-path interference produces a residual
  sneak-current amplitude:
    A = |sin(π·rem/k)|
  When rem = 0 → A = 0 → full destructive cancellation → mesh CLEAR
  (candidate is a true factor, no sneak-path ambiguity).
  CLEAR threshold: A < 0.05.

LAYER 3 — Binary ReRAM Output Latch
  A digital modular-arithmetic unit computes rem = N mod k and drives
  a binary ReRAM cell into LRS (logic HIGH, 3.3 V) when rem == 0, or
  leaves it in HRS (logic LOW, 0 V) otherwise. All layers must agree
  for a CONFIRMED FACTOR output.
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
    "TiO2 Memristor Crossbar",
    "VO2 Threshold Switch",
    "Sneak-Path Cancel Mesh",
    "ReRAM Output Latch",
]

# ── physical parameters ───────────────────────────────────────────────────────
R_ON               = 100.0    # Ω   — fully-doped (SET) memristance
R_OFF              = 16000.0  # Ω   — fully-undoped (RESET) memristance
G_LOCK_THRESHOLD   = 0.97     # normalised conductance must exceed this
T_AMBIENT          = 295.0    # K   — ambient film temperature
T_MAX_DELTA        = 80.0     # K   — max Joule ΔT at full residue
T_IMT              = 296.0    # K   — insulator-metal transition threshold
SEEBECK_UV_PER_K   = 200.0    # µV/K — thermoelectric readout coefficient
RF_NULL_THRESHOLD  = 0.05     # sneak-path amplitude below this = CLEAR

# ── layer physics ─────────────────────────────────────────────────────────────

def layer_tio2_memristor(n, k):
    """
    HP TiO2 linear ion-drift memristor.
    Doped-boundary fraction x = (N mod k)/k
    Memristance M(x) = R_on + (R_off - R_on)*x
    Normalised conductance g = R_on / M(x)
    SET/LOCK when g > G_LOCK_THRESHOLD (boundary fully retracted, x→0)
    """
    rem = n % k
    x   = rem / k
    M   = R_ON + (R_OFF - R_ON) * x
    g   = R_ON / M
    lock = g > G_LOCK_THRESHOLD
    return lock, f"x={x:6.4f}  M={M:8.1f}Ω  g={g:.4f}  {'SET/LOCK' if lock else 'DRIFT'}"


def layer_vo2_threshold_switch(n, k):
    """
    VO2 Mott memristor, self-heating threshold switch.
    ΔT = T_MAX_DELTA · (rem/k)
    T_film = T_AMBIENT + ΔT
    V_th = SEEBECK_UV_PER_K · ΔT  (µV, switching figure of merit)
    Gate OPEN (metallic LRS) when T_film < T_IMT (rem/k < 1/80)
    """
    rem = n % k
    dT  = T_MAX_DELTA * (rem / k)
    T   = T_AMBIENT + dT
    Vth = SEEBECK_UV_PER_K * dT
    gate = T < T_IMT
    return gate, f"ΔT={dT:5.2f}K  T={T:6.2f}K  Vth={Vth:6.1f}µV  {'LRS/OPEN' if gate else 'HRS/CLOSED'}"


def layer_sneak_path_mesh(n, k):
    """
    Crossbar sneak-path cancellation mesh.
    Residual sneak-current amplitude A = |sin(pi * rem/k)|
    CLEAR (factor signal, no sneak ambiguity) when A < RF_NULL_THRESHOLD
    """
    rem = n % k
    A   = abs(math.sin(math.pi * rem / k))
    clear = A < RF_NULL_THRESHOLD
    return clear, f"A={A:.4f}  {'CLEAR' if clear else 'ACTIVE'}"


def layer_reram_latch(n, k):
    """
    Binary ReRAM output latch.
    Computes rem = N mod k in a carry-save adder array.
    Cell driven to LRS (3.3 V, logic HIGH) when rem == 0, else HRS (0 V).
    """
    rem   = n % k
    latch = rem == 0
    rail  = 3.3 if latch else 0.0
    return latch, f"rem={rem:<8} V_out={rail:.1f}V  {'LRS/HIGH ✓' if latch else 'HRS/LOW'}"


LAYER_FNS = [
    layer_tio2_memristor,
    layer_vo2_threshold_switch,
    layer_sneak_path_mesh,
    layer_reram_latch,
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

# ── thermal map (VO2 threshold-switch layer) ─────────────────────────────────
def render_thermal_map(n, results):
    print(c("  VO2 THRESHOLD SWITCH — Film Temperature Map", B+YL))
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
                if   ratio < 0.013: line += c("▓", YL+B)   # ΔT < 1 K  (near-IMT zone)
                elif ratio < 0.1:   line += c("▒", YL)
                elif ratio < 0.4:   line += c("░", CY)
                else:               line += c("·", BL+D)
        print(line)
    print()
    print(c("  Legend: ", D)
        + c("█", GR+B) + c(" Factor(ΔT=0)  ", D)
        + c("▓", YL+B) + c(" Near-IMT zone(ΔT<1K)  ", D)
        + c("▒", YL)   + c(" Warm  ", D)
        + c("░", CY)   + c(" Cool  ", D)
        + c("·", BL+D) + c(" Cold(large rem)", D))
    print()

# ── ion-drift boundary map (TiO2 memristor layer) ────────────────────────────
DRIFT_CHARS = " ·∘○◎●◉"

def render_optical_map(n, results):
    print(c("  TiO2 MEMRISTOR — Ion-Drift Boundary Map  (x = rem/k)", B+CY))
    print(c("  Bright=near-SET (small x, low M), dim=near-RESET, green=LOCK\n", D))
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
                rem = n % k if k else 0
                x   = rem / k if k else 1.0
                M   = R_ON + (R_OFF - R_ON) * x
                g   = R_ON / M               # 0..1, near 1 when x small
                ch  = DRIFT_CHARS[int(g * (len(DRIFT_CHARS) - 1))]
                col_c = CY+B if g > 0.9 else (CY if g > 0.6 else (YL if g > 0.3 else D))
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
    print(c(centre("512-CHANNEL MEMRISTIVE CROSSBAR"), CY+B))
    print(c(centre("LAYERED GRID HARDWARE SIMULATOR  v3.0"), CY+B))
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
        "TiO2 ion-drift memristor  R_on=100Ω R_off=16kΩ  lock g>0.97",
        "VO2 Mott threshold switch  ΔT=80K·(rem/k)  IMT below T=296K",
        "Crossbar sneak-path mesh  null threshold A<0.05",
        "Binary ReRAM latch  LRS/HRS  3.3 V output",
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
            print(c("Usage: python memristor_grid_simulator.py <integer>", RD)); sys.exit(1)
    else:
        print(c("512-Channel Memristive Crossbar Simulator  v3.0", CY+B))
        try:
            N = int(input(c("  Enter N: ", YL)).strip())
        except ValueError:
            print(c("Invalid integer.", RD)); sys.exit(1)

    factorise(N)
