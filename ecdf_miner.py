"""
ecdf_miner.py
=============
SHA-256 PoW miner driven by Gap ECDF inversion, with genuine efficiency analysis.

WHAT THE ECDF GIVES YOU
-----------------------
The Gap ECDF fits Exp(mean_gap).  Inverting the CDF:

    window(p) = -mean_gap * ln(1 - p)

Starting from any valid nonce anchor, a scan of window(p) nonces finds the
next valid nonce with probability p.  This is exact -- no approximation.

THE REAL EFFICIENCY GAIN: BUDGET CERTAINTY
------------------------------------------
Hash efficiency (hashes-per-hit) is identical for all strategies:
  E[hashes/hit] = mean_gap  (provable: E[min(gap,w)] / p = mean_gap*p / p)

The gain is CERTAINTY OVER YOUR BUDGET:

  1. Marginal certainty per hash  dP/dh = exp(-h/mean_gap) / mean_gap
     Highest at h=0, decays exponentially.  The ECDF curve IS this function.

  2. Budget-to-confidence mapping  p(B) = 1 - exp(-B/mean_gap)
     The only formula that correctly sizes a mining budget for a target probability.
     Naive scanning has no such closed form -- you must simulate.

  3. Miss-rate control  At conf=p, exactly (1-p) of windows miss.
     ECDF lets you set this parameter explicitly; naive cannot.

  4. Certainty efficiency  dp/dh is steepest early (near h=0).
     Spending 1 * mean_gap hashes buys 63.2% confidence.
     Spending 3 * mean_gap buys 95.0%.  The last 5% costs 2x more hashes.

Phases
------
  1. Calibration -- linear scan, fit Exp, KS test
  2. Mining      -- ECDF-windowed search at each confidence level
  3. HTML chart  -- ECDF + CDF inverse + marginal certainty (dP/dh) + session stats

Usage
-----
  pip install numpy scipy
  python ecdf_miner.py
  python ecdf_miner.py --diff 16 --calib 1_500_000 --targets 20
  python ecdf_miner.py --diff 18 --calib 3_000_000 --conf 0.5 0.9 0.99 --targets 10
  python ecdf_miner.py --header MY_BLOCK --diff 14 --calib 500_000 --targets 30
"""

import argparse, hashlib, json, math, struct, time
import numpy as np
from scipy.stats import kstest

# ── CLI ────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--diff",    type=int,               default=16)
ap.add_argument("--calib",   type=int,               default=1_500_000)
ap.add_argument("--targets", type=int,               default=20)
ap.add_argument("--conf",    type=float, nargs="+",  default=[0.5, 0.9, 0.99])
ap.add_argument("--header",  type=str,               default="PROTO_BLOCK_v1")
ap.add_argument("--out",     type=str,               default="ecdf_miner.html")
args = ap.parse_args()

HEADER = args.header.encode()
DIFF   = args.diff
TARGET = 1 << (256 - DIFF)
CONFS  = sorted(set(max(0.01, min(0.999, c)) for c in args.conf))

# ── Hash ───────────────────────────────────────────────────────────────────────
def dsha(nonce: int) -> int:
    d = HEADER + struct.pack("<Q", nonce)
    return int.from_bytes(hashlib.sha256(hashlib.sha256(d).digest()).digest(), "big")

def is_valid(nonce: int) -> bool:
    return dsha(nonce) < TARGET

def hex_hash(nonce: int) -> str:
    d = HEADER + struct.pack("<Q", nonce)
    return hashlib.sha256(hashlib.sha256(d).digest()).hexdigest()

# ── Helpers ────────────────────────────────────────────────────────────────────
def cdf_inv(p: float, mu: float) -> int:
    return max(1, int(-mu * math.log(1.0 - p)))

def ecdf_pts(arr: np.ndarray, max_pts: int = 500):
    s = np.sort(arr); n = len(s)
    step = max(1, n // max_pts)
    idx  = list(range(0, n, step))
    if idx[-1] != n - 1: idx.append(n - 1)
    xs = [0.0] + [round(float(s[i]), 4) for i in idx]
    ys = [0.0] + [round((i + 1) / n, 4) for i in idx]
    return xs, ys

# ── Phase 1: Calibration ───────────────────────────────────────────────────────
print("=" * 68)
print("  ECDF-Driven SHA-256 PoW Miner  --  Budget Certainty Analysis")
print("=" * 68)
print(f"  Header : {args.header}")
print(f"  Diff   : {DIFF} bits  (1 in {2**DIFF:,})")
print(f"  Calib  : {args.calib:,} nonces")
print(f"  Conf   : {[f'{c*100:.0f}%' for c in CONFS]}")
print()
print("  [1/3] Calibration -- linear scan ...")

calib = []
t0 = time.perf_counter()
for n in range(args.calib):
    if is_valid(n):
        calib.append(n)
t_calib = time.perf_counter() - t0

print(f"        {args.calib:,} nonces in {t_calib:.2f}s  ({args.calib/t_calib/1e6:.2f} MH/s)")
print(f"        Found {len(calib)} valid nonces")

if len(calib) < 5:
    print(f"\n  x  Only {len(calib)} hits (expected ~{args.calib/2**DIFF:.1f}).")
    print(f"     Raise --calib or lower --diff.")
    raise SystemExit(1)

gaps      = np.diff(calib).astype(float)
mean_gap  = float(gaps.mean())
std_gap   = float(gaps.std())
cv        = std_gap / mean_gap
norm_gaps = gaps / mean_gap
ks_stat, ks_p = kstest(np.sort(norm_gaps), "expon")

print()
print(f"  Gap fit (n={len(gaps)} gaps):")
print(f"    mean  = {mean_gap:>12,.1f}   theoretical = {2**DIFF:,}")
print(f"    std   = {std_gap:>12,.1f}")
print(f"    CV    = {cv:>12.4f}   Poisson -> 1.0")
print(f"    KS    = {ks_stat:>12.4f}   p = {ks_p:.4f}  "
      f"-> {'PASS' if ks_p > 0.05 else 'FAIL'}")
print()

# ── Efficiency analysis (computed, not simulated) ───────────────────────────
# Budget certainty: p(B) = 1 - exp(-B/mean_gap)
# Marginal certainty: dp/dB = exp(-B/mean_gap) / mean_gap
# Certainty at CONFS:
print("  Budget certainty derived from ECDF:")
print(f"  {'Confidence':>12}  {'Budget (hashes)':>16}  {'Ratio to mean_gap':>18}  {'Marginal dp/dh':>16}")
for c in CONFS:
    w    = cdf_inv(c, mean_gap)
    dpdh = math.exp(-w / mean_gap) / mean_gap   # marginal at that point
    print(f"  {c*100:>11.1f}%  {w:>16,}  {w/mean_gap:>18.3f}x  {dpdh*1e6:>14.4f} /MH")

# Standard certainty milestones from Exp fit:
print()
print("  Certainty milestones (from Exp fit):")
for k, label in [(0.693, "0.5"), (1.0, "0.632"), (2.303, "0.9"), (4.605, "0.99")]:
    b = int(k * mean_gap)
    p = 1 - math.exp(-b / mean_gap)
    print(f"    {k:.3f} x mean_gap = {b:>10,} hashes -> {p*100:.2f}% confidence")
print()

# ── Phase 2: Mining ────────────────────────────────────────────────────────────
print(f"  [2/3] ECDF-windowed mining -- {args.targets} targets per conf ...")
print()

sessions = {}

for conf in CONFS:
    window   = cdf_inv(conf, mean_gap)
    label    = f"{conf*100:.0f}%"
    p_theory = 1 - math.exp(-window / mean_gap)
    print(f"  -- conf={label}  window={window:,}  P(hit)={p_theory:.4f} --")

    found   = []; hashes = 0; windows = 0; anchor = calib[-1]
    t1 = time.perf_counter()

    while len(found) < args.targets:
        windows += 1
        hit = None
        for n in range(anchor + 1, anchor + window + 1):
            hashes += 1
            if is_valid(n):
                hit = n; break
        if hit:
            found.append((hit, hex_hash(hit)))
            anchor = hit
            print(f"    [{len(found):>3}/{args.targets}]  nonce={hit:>14,}  "
                  f"hash={hex_hash(hit)[:18]}...  w={windows:,}  h={hashes:,}")
        else:
            anchor += window

    t_mine = time.perf_counter() - t1

    # Per-window efficiency: for each window, classify hit/miss and hashes spent
    # Reconstruct from found nonces
    win_hashes_hit  = []   # hashes spent in hit-windows
    win_hashes_miss = []   # hashes spent in miss-windows (always = window)
    hit_nonces = [f[0] for f in found]
    pos = calib[-1]
    hi  = 0
    for _ in range(windows):
        if hi < len(hit_nonces) and hit_nonces[hi] <= pos + window:
            gap = hit_nonces[hi] - pos
            win_hashes_hit.append(gap)
            pos = hit_nonces[hi]
            hi += 1
        else:
            win_hashes_miss.append(window)
            pos += window

    n_hits  = len(win_hashes_hit)
    n_miss  = len(win_hashes_miss)
    e_h_hit = float(np.mean(win_hashes_hit)) if win_hashes_hit else 0.0
    # theoretical E[gap | gap <= window] = mean_gap*(1 - (1+w/mean_gap)*exp(-w/mean_gap)) / (1-exp(-w/mean_gap))
    w = window
    mu = mean_gap
    e_gap_given_hit_theory = mu * (1 - (1 + w/mu)*math.exp(-w/mu)) / (1 - math.exp(-w/mu))

    sessions[conf] = {
        "label":                label,
        "window":               window,
        "conf":                 conf,
        "found":                found,
        "hashes":               hashes,
        "windows":              windows,
        "n_hits":               n_hits,
        "n_miss":               n_miss,
        "hit_rate":             n_hits / windows,
        "expected_hr":          p_theory,
        "hashes_per_f":         hashes / len(found) if found else 0,
        "e_h_hit_obs":          e_h_hit,
        "e_h_hit_theory":       e_gap_given_hit_theory,
        "time":                 t_mine,
    }
    print(f"    -> found={len(found)} | windows={windows} | hashes={hashes:,} | "
          f"hit_rate={n_hits/windows:.3f} (exp {p_theory:.3f}) | {t_mine:.2f}s")
    print(f"       E[h/hit-window]={e_h_hit:,.1f}  theory={e_gap_given_hit_theory:,.1f}")
    print()

# ── Phase 3: HTML chart ────────────────────────────────────────────────────────
print(f"  [3/3] Writing chart -> {args.out}")

PALETTE    = ["#4EFFA0", "#5B7FFF", "#FFCB47", "#FF5565", "#C084FC"]
CC         = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(CONFS)}

# ECDF datasets
theory_x = [round(x, 3) for x in np.linspace(0, 5, 300).tolist()]
theory_y = [round(1 - math.exp(-x), 6) for x in theory_x]
cal_x, cal_y = ecdf_pts(norm_gaps)

ecdf_datasets = [
    {"label": "Exp(1) theory", "data": [{"x":x,"y":y} for x,y in zip(theory_x,theory_y)],
     "borderColor":"#545A72","borderWidth":1.5,"borderDash":[7,5],
     "pointRadius":0,"fill":False,"tension":0,"order":0},
    {"label": f"Calibration ECDF (n={len(gaps):,}, KS p={ks_p:.3f})",
     "data": [{"x":x,"y":y} for x,y in zip(cal_x,cal_y)],
     "borderColor":"#C8CEDE","borderWidth":2,"pointRadius":0,
     "fill":False,"tension":0,"stepped":"after","order":1},
]
for i, conf in enumerate(CONFS):
    s = sessions[conf]
    if len(s["found"]) >= 2:
        fn = np.array([f[0] for f in s["found"]], dtype=np.float64)
        fg = np.diff(fn) / mean_gap
        fx, fy = ecdf_pts(fg)
    else:
        fx, fy = [0.0, 1.0], [0.0, 1.0]
    ecdf_datasets.append({
        "label": f"conf={s['label']} (n={len(s['found'])} hits)",
        "data": [{"x":x,"y":y} for x,y in zip(fx,fy)],
        "borderColor": CC[conf], "borderWidth":1.8,
        "pointRadius":0,"fill":False,"tension":0,"stepped":"after","order":i+2,
    })

# CDF inverse dataset
inv_ps = [round(p, 3) for p in np.linspace(0.01, 0.999, 300).tolist()]
inv_ws = [round(cdf_inv(p, mean_gap) / mean_gap, 4) for p in inv_ps]
conf_markers = [{"x":c,"y":round(cdf_inv(c,mean_gap)/mean_gap,4),
                 "label":f"{c*100:.0f}%","color":CC[c]} for c in CONFS]

# Marginal certainty (dp/dh) curve: shows how fast each additional hash
# increases confidence -- this IS the efficiency gain visualised
h_axis  = [round(h, 3) for h in np.linspace(0, 5, 300).tolist()]  # in units of mean_gap
dpdh_y  = [round(math.exp(-h), 6) for h in h_axis]    # dp/d(h/mu) = exp(-h/mu) normalised
cumconf = [round(1 - math.exp(-h), 6) for h in h_axis]

# Efficiency gain per window: E[gap|hit] / window -- fraction of budget actually needed
eff_confs  = [round(p, 3) for p in np.linspace(0.01, 0.999, 300).tolist()]
eff_gains  = []
for p in eff_confs:
    w  = -mean_gap * math.log(1 - p)
    mu = mean_gap
    e_gap_hit = mu * (1 - (1 + w/mu)*math.exp(-w/mu)) / (1 - math.exp(-w/mu))
    # fraction of window budget used when hit occurs
    eff_gains.append(round(e_gap_hit / w, 4))

ecdf_json    = json.dumps(ecdf_datasets)
inv_ps_json  = json.dumps(inv_ps)
inv_ws_json  = json.dumps(inv_ws)
cmk_json     = json.dumps(conf_markers)
h_json       = json.dumps(h_axis)
dpdh_json    = json.dumps(dpdh_y)
cc_json      = json.dumps(cumconf)
eff_c_json   = json.dumps(eff_confs)
eff_g_json   = json.dumps(eff_gains)
mg_js        = round(mean_gap, 1)

# Session efficiency bar data for Chart.js
bar_labels   = json.dumps([sessions[c]["label"] for c in CONFS])
bar_obs_hr   = json.dumps([round(sessions[c]["hit_rate"], 4) for c in CONFS])
bar_exp_hr   = json.dumps([round(sessions[c]["expected_hr"], 4) for c in CONFS])
bar_e_h_obs  = json.dumps([round(sessions[c]["e_h_hit_obs"] / mean_gap, 4) for c in CONFS])
bar_e_h_th   = json.dumps([round(sessions[c]["e_h_hit_theory"] / mean_gap, 4) for c in CONFS])
bar_colors   = json.dumps([CC[c] for c in CONFS])

# Table helpers
def session_rows():
    rows = ""
    for conf in CONFS:
        s   = sessions[conf]
        col = CC[conf]
        # efficiency gain = how much LESS than window(p) the hit windows used on average
        eff = (1 - s["e_h_hit_obs"] / s["window"]) * 100 if s["window"] else 0
        rows += (
            f"<tr>"
            f"<td><span class='dot' style='background:{col}'></span></td>"
            f"<td>{s['label']}</td>"
            f"<td>{s['window']:,}</td>"
            f"<td>{s['windows']:,}</td>"
            f"<td>{s['n_hits']}/{s['windows']}</td>"
            f"<td>{s['hit_rate']:.4f} <span class='dim'>({s['expected_hr']:.4f})</span></td>"
            f"<td>{s['e_h_hit_obs']:,.1f} <span class='dim'>({s['e_h_hit_theory']:,.1f})</span></td>"
            f"<td class='gain'>-{eff:.1f}%</td>"
            f"<td>{s['hashes']:,}</td>"
            f"<td>{s['time']:.2f}s</td>"
            f"</tr>\n"
        )
    return rows

def legend_html():
    out  = "<span class='li'><span class='ls dash'></span>Exp(1) theory</span>\n"
    out += "<span class='li'><span class='ls' style='background:#C8CEDE'></span>Calibration ECDF</span>\n"
    for c in CONFS:
        s = sessions[c]
        out += (f"<span class='li'><span class='ls' style='background:{CC[c]}'></span>"
                f"conf={s['label']} ECDF</span>\n")
    return out

def found_rows():
    rows = ""
    for conf in CONFS:
        s = sessions[conf]
        for nonce, h in s["found"]:
            rows += (
                f"<tr>"
                f"<td><span class='dot' style='background:{CC[conf]}'></span>{s['label']}</td>"
                f"<td class='mono'>{nonce:,}</td>"
                f"<td class='mono hash'>{h}</td>"
                f"</tr>\n"
            )
    return rows

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ECDF Miner -- diff={DIFF}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#07080A;--surface:#0D0F13;--panel:#111519;
  --border:#191D27;--border2:#202535;
  --text:#BFC8DC;--muted:#4A526A;--dim:#252A3A;
  --accent:#4EFFA0;--blue:#5B7FFF;--warn:#FFCB47;--danger:#FF5565;
  --mono:'JetBrains Mono',monospace;--head:'Syne',sans-serif;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:var(--mono);
     padding:36px 30px 80px;min-height:100vh}}
body::before{{content:'';position:fixed;inset:0;pointer-events:none;
  background-image:linear-gradient(var(--border) 1px,transparent 1px),
                   linear-gradient(90deg,var(--border) 1px,transparent 1px);
  background-size:40px 40px;opacity:.28;z-index:0}}
.w{{position:relative;z-index:1;max-width:1160px;margin:0 auto}}

h1{{font-family:var(--head);font-size:27px;font-weight:800;color:#fff;letter-spacing:-.5px}}
h1 em{{font-style:normal;color:var(--accent)}}
.sub{{font-size:11px;color:var(--muted);margin-top:5px;letter-spacing:.05em}}
.hdr{{display:flex;align-items:flex-start;justify-content:space-between;
      gap:16px;flex-wrap:wrap;margin-bottom:28px}}
.badges{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:4px}}
.badge{{display:inline-flex;align-items:center;gap:6px;padding:5px 13px;
        border-radius:99px;font-size:11px;font-weight:600;
        border:1px solid var(--border2);background:var(--surface)}}
.badge.g{{color:var(--accent);border-color:var(--accent);background:rgba(78,255,160,.07)}}
.badge.b{{color:var(--blue);border-color:var(--blue);background:rgba(91,127,255,.08)}}
.badge.y{{color:var(--warn);border-color:var(--warn);background:rgba(255,203,71,.07)}}
.dot2{{width:6px;height:6px;border-radius:50%;background:currentColor}}

/* STAT CARDS */
.sg{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:18px}}
@media(max-width:860px){{.sg{{grid-template-columns:repeat(3,1fr)}}}}
.sc{{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:13px 15px}}
.sc .l{{font-size:10px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}}
.sc .v{{font-size:18px;font-weight:700;color:#fff;margin-top:3px;font-family:var(--head)}}
.sc .h{{font-size:10px;color:var(--dim);margin-top:2px}}
.sc.ac{{border-color:var(--accent);background:rgba(78,255,160,.04)}}.sc.ac .v{{color:var(--accent)}}
.sc.bl{{border-color:var(--blue);background:rgba(91,127,255,.05)}}.sc.bl .v{{color:var(--blue)}}
.sc.yl .v{{color:var(--warn)}}

/* PANELS */
.box{{background:var(--panel);border:1px solid var(--border);border-radius:14px;
      padding:20px 22px;margin-bottom:14px}}
.bt{{font-size:10px;letter-spacing:.1em;text-transform:uppercase;
     color:var(--muted);margin-bottom:16px;display:flex;align-items:center;gap:8px}}
.bt::before{{content:'';width:3px;height:13px;background:var(--accent);border-radius:2px}}

/* EXPLAIN */
.explain{{background:var(--panel);border:1px solid var(--border);border-radius:14px;
          padding:20px 24px;margin-bottom:14px;line-height:1.9;font-size:12px;color:var(--muted)}}
.explain h2{{font-family:var(--head);font-size:14px;font-weight:700;color:var(--text);margin-bottom:12px}}
.formula{{background:var(--surface);border:1px solid var(--border2);border-radius:8px;
          padding:11px 15px;margin:10px 0;font-size:12.5px;font-weight:600;color:#fff;letter-spacing:.03em}}
.insight{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}}
@media(max-width:680px){{.insight{{grid-template-columns:1fr}}}}
.ins{{background:var(--surface);border:1px solid var(--border2);border-radius:10px;padding:12px 14px}}
.ins .il{{font-size:10px;color:var(--muted);letter-spacing:.07em;text-transform:uppercase;margin-bottom:5px}}
.ins .iv{{font-size:13px;color:var(--text);font-weight:500}}

/* CHART GRID */
.two{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
.three{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}}
@media(max-width:900px){{.two,.three{{grid-template-columns:1fr}}}}
.ca{{position:relative;height:280px}}
.ca.tall{{height:320px}}

.legend{{display:flex;flex-wrap:wrap;gap:7px 18px;margin-top:11px;font-size:11px;color:var(--muted)}}
.li{{display:flex;align-items:center;gap:6px}}
.ls{{width:18px;height:2.5px;border-radius:2px}}
.ls.dash{{background:repeating-linear-gradient(90deg,#545A72 0 5px,transparent 5px 10px)}}

/* TABLE */
table{{width:100%;border-collapse:collapse;font-size:11.5px}}
th{{text-align:left;color:var(--muted);font-weight:400;font-size:10px;
    letter-spacing:.06em;text-transform:uppercase;
    padding:0 10px 8px 0;border-bottom:1px solid var(--border)}}
td{{padding:7px 10px 7px 0;border-bottom:1px solid var(--dim);color:var(--muted)}}
tr:last-child td{{border:none}}
.dot{{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px}}
.mono{{font-family:var(--mono)}}
.hash{{font-size:10px;color:var(--dim);word-break:break-all}}
.dim{{color:var(--dim)}}
.gain{{color:var(--accent);font-weight:600}}

::-webkit-scrollbar{{width:4px;height:4px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:var(--border2);border-radius:4px}}
</style>
</head>
<body>
<div class="w">

<!-- HEADER -->
<div class="hdr">
  <div>
    <h1>ECDF <em>Miner</em> &mdash; Budget Certainty</h1>
    <p class="sub">SHA-256 PoW &middot; header: {args.header} &middot; diff={DIFF} bits &middot; 1 in {2**DIFF:,} &middot; KS p={ks_p:.4f} &middot; mean gap = {mean_gap:,.0f}</p>
  </div>
  <div class="badges">
    <span class="badge g"><span class="dot2"></span>CDF inversion active</span>
    <span class="badge b"><span class="dot2"></span>E[h/hit] = {mean_gap:,.0f} always</span>
    <span class="badge y"><span class="dot2"></span>gain = budget certainty</span>
  </div>
</div>

<!-- STAT CARDS -->
<div class="sg">
  <div class="sc ac"><div class="l">Mean gap</div><div class="v">{mean_gap:,.0f}</div><div class="h">fitted Exp(&lambda;)</div></div>
  <div class="sc bl"><div class="l">CV</div><div class="v">{cv:.4f}</div><div class="h">Poisson &rarr; 1.00</div></div>
  <div class="sc">
    <div class="l">KS p-value</div>
    <div class="v" style="font-size:14px;margin-top:5px;color:{'var(--accent)' if ks_p>0.05 else 'var(--danger)'}">{ks_p:.4f}</div>
    <div class="h">{'PASS' if ks_p>0.05 else 'FAIL'} &mdash; Exp fit</div>
  </div>
  <div class="sc"><div class="l">50% budget</div><div class="v">{cdf_inv(0.5,mean_gap):,}</div><div class="h">0.693 &times; mean_gap</div></div>
  <div class="sc"><div class="l">90% budget</div><div class="v">{cdf_inv(0.9,mean_gap):,}</div><div class="h">2.303 &times; mean_gap</div></div>
  <div class="sc yl"><div class="l">99% budget</div><div class="v">{cdf_inv(0.99,mean_gap):,}</div><div class="h">4.605 &times; mean_gap</div></div>
</div>

<!-- EXPLAIN -->
<div class="explain">
  <h2>The efficiency gain: budget certainty, not fewer hashes per hit</h2>
  <strong style="color:var(--text)">Provable fact:</strong> E[hashes/hit] = mean_gap for <em>every</em> strategy &mdash;
  because E[min(gap,&nbsp;window)] / p&nbsp;=&nbsp;(mean_gap&nbsp;&times;&nbsp;p)&nbsp;/&nbsp;p&nbsp;=&nbsp;mean_gap always.
  The ECDF gain is <strong style="color:var(--accent)">knowing exactly how large a budget to commit for a given confidence</strong>:

  <div class="formula">
    p(B) = 1 &minus; exp(&minus;B / mean_gap) &nbsp;&nbsp;&nbsp;
    window(p) = &minus;mean_gap &times; ln(1 &minus; p) &nbsp;&nbsp;&nbsp;
    dp/dB = exp(&minus;B/mean_gap) / mean_gap
  </div>

  <div class="insight">
    <div class="ins">
      <div class="il">Budget sizing</div>
      <div class="iv">Commit exactly window(p) hashes. P(hit) = p by construction. No simulation needed.</div>
    </div>
    <div class="ins">
      <div class="il">Diminishing returns</div>
      <div class="iv">dp/dB is steepest at B=0 and decays as exp(&minus;B/&mu;). The last 9% of confidence (90%&rarr;99%) costs 2&times; more hashes than the first 90%.</div>
    </div>
    <div class="ins">
      <div class="il">Hit-window efficiency</div>
      <div class="iv">When a hit occurs inside a window, only E[gap&nbsp;|&nbsp;gap&nbsp;&le;&nbsp;window] hashes are used &mdash; always less than window. The surplus budget is freed immediately.</div>
    </div>
  </div>
</div>

<!-- TOP CHARTS ROW -->
<div class="two">
  <div class="box">
    <div class="bt">Gap ECDF &mdash; calibration vs sessions vs Exp(1)</div>
    <div class="ca tall"><canvas id="ecdfChart"></canvas></div>
    <div class="legend">{legend_html()}</div>
  </div>
  <div class="box">
    <div class="bt">Budget certainty &mdash; p(B) and dp/dB (marginal gain per hash)</div>
    <div class="ca tall"><canvas id="certaintyChart"></canvas></div>
    <div class="legend">
      <span class="li"><span class="ls" style="background:#4EFFA0"></span>p(B) = 1&minus;exp(&minus;B/&mu;)</span>
      <span class="li"><span class="ls" style="background:#5B7FFF;opacity:.7"></span>dp/dB (marginal, right axis)</span>
    </div>
  </div>
</div>

<!-- BOTTOM CHARTS ROW -->
<div class="two">
  <div class="box">
    <div class="bt">CDF inverse &mdash; confidence &rarr; scan window (&divide; mean_gap)</div>
    <div class="ca"><canvas id="invChart"></canvas></div>
  </div>
  <div class="box">
    <div class="bt">Hit-window efficiency &mdash; E[gap|hit] / window vs confidence</div>
    <div class="ca"><canvas id="effChart"></canvas></div>
    <div class="legend">
      <span class="li"><span class="ls" style="background:#4EFFA0"></span>fraction of window budget used at hit</span>
      <span style="font-size:10px;color:var(--dim)">approaches 1 as p&rarr;1, drops toward 0 as p&rarr;0</span>
    </div>
  </div>
</div>

<!-- SESSION TABLE -->
<div class="box">
  <div class="bt">Session results &mdash; observed vs theoretical efficiency</div>
  <div style="overflow-x:auto">
  <table>
    <thead><tr>
      <th></th><th>conf</th><th>window</th><th>windows fired</th>
      <th>hits/windows</th><th>hit rate (exp)</th>
      <th>E[h/hit-window] (theory)</th>
      <th>window savings</th>
      <th>total hashes</th><th>time</th>
    </tr></thead>
    <tbody>{session_rows()}</tbody>
  </table>
  </div>
</div>

<!-- FOUND NONCES -->
<div class="box">
  <div class="bt">Found valid nonces</div>
  <div style="overflow-x:auto;max-height:380px">
  <table>
    <thead><tr><th>session</th><th>nonce</th><th>double-SHA256</th></tr></thead>
    <tbody>{found_rows()}</tbody>
  </table>
  </div>
</div>

</div><!-- /w -->
<script>
const gc='rgba(255,255,255,0.04)', tx='#4A526A';
const font={{size:11,family:'JetBrains Mono'}};
const mkChart=(id,cfg)=>new Chart(document.getElementById(id),cfg);

// 1. ECDF chart
mkChart('ecdfChart',{{
  type:'line', data:{{datasets:{ecdf_json}}},
  options:{{
    responsive:true,maintainAspectRatio:false,animation:{{duration:500}},
    interaction:{{mode:'index',intersect:false}},
    plugins:{{legend:{{display:false}},tooltip:{{
      backgroundColor:'#111519',borderColor:'#191D27',borderWidth:1,
      titleColor:tx,bodyColor:'#BFC8DC',
      callbacks:{{title:i=>'gap/mean = '+i[0].parsed.x.toFixed(3),
                 label:c=>c.dataset.label+': '+c.parsed.y.toFixed(3)}}
    }}}},
    scales:{{
      x:{{type:'linear',min:0,max:5,grid:{{color:gc}},ticks:{{color:tx,font}},
         title:{{display:true,text:'gap / mean_gap',color:tx,font}}}},
      y:{{min:0,max:1.05,grid:{{color:gc}},ticks:{{color:tx,font,callback:v=>v.toFixed(1)}},
         title:{{display:true,text:'cumulative probability',color:tx,font}}}}
    }}
  }}
}});

// 2. Budget certainty + marginal chart (dual y-axis)
const hAx={h_json}, cumC={cc_json}, dpdh={dpdh_json};
mkChart('certaintyChart',{{
  type:'line',
  data:{{datasets:[
    {{label:'p(B)',data:hAx.map((h,i)=>({{x:h,y:cumC[i]}})),
      borderColor:'#4EFFA0',borderWidth:2.5,pointRadius:0,fill:false,tension:0.3,yAxisID:'y'}},
    {{label:'dp/dB',data:hAx.map((h,i)=>({{x:h,y:dpdh[i]}})),
      borderColor:'rgba(91,127,255,0.6)',borderWidth:1.5,pointRadius:0,
      fill:true,backgroundColor:'rgba(91,127,255,0.06)',tension:0.3,yAxisID:'y2'}},
  ]}},
  options:{{
    responsive:true,maintainAspectRatio:false,animation:{{duration:500}},
    interaction:{{mode:'index',intersect:false}},
    plugins:{{legend:{{display:false}},tooltip:{{
      backgroundColor:'#111519',borderColor:'#191D27',borderWidth:1,
      titleColor:tx,bodyColor:'#BFC8DC',
      callbacks:{{title:i=>'budget = '+i[0].parsed.x.toFixed(2)+' \u00d7 mean_gap'}}
    }}}},
    scales:{{
      x:{{type:'linear',min:0,max:5,grid:{{color:gc}},ticks:{{color:tx,font}},
         title:{{display:true,text:'budget / mean_gap',color:tx,font}}}},
      y:{{min:0,max:1.05,position:'left',grid:{{color:gc}},ticks:{{color:'#4EFFA0',font}},
         title:{{display:true,text:'confidence p(B)',color:'#4EFFA0',font}}}},
      y2:{{min:0,position:'right',grid:{{drawOnChartArea:false}},
           ticks:{{color:'rgba(91,127,255,0.8)',font}},
           title:{{display:true,text:'marginal dp/d(B/\u03bc)',color:'rgba(91,127,255,0.8)',font}}}}
    }}
  }}
}});

// 3. CDF inverse chart
const invPs={inv_ps_json}, invWs={inv_ws_json}, cmk={cmk_json};
mkChart('invChart',{{
  type:'line',
  data:{{datasets:[
    {{label:'window(p)=-ln(1-p)',data:invPs.map((p,i)=>({{x:p,y:invWs[i]}})),
      borderColor:'#4EFFA0',borderWidth:2,pointRadius:0,fill:false,tension:0}},
    {{label:'Selected',data:cmk.map(m=>({{x:m.x,y:m.y}})),
      borderColor:'transparent',backgroundColor:cmk.map(m=>m.color),
      pointRadius:8,showLine:false,
      tooltip:{{callbacks:{{label:c=>cmk[c.dataIndex].label+' \u2192 '+
        Math.round(c.parsed.y*{mg_js}).toLocaleString()+' hashes'}}}}}}
  ]}},
  options:{{
    responsive:true,maintainAspectRatio:false,animation:{{duration:500}},
    plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#111519',
      borderColor:'#191D27',borderWidth:1,titleColor:tx,bodyColor:'#BFC8DC'}}}},
    scales:{{
      x:{{type:'linear',min:0,max:1,grid:{{color:gc}},ticks:{{color:tx,font}},
         title:{{display:true,text:'confidence p',color:tx,font}}}},
      y:{{min:0,grid:{{color:gc}},ticks:{{color:tx,font}},
         title:{{display:true,text:'window / mean_gap',color:tx,font}}}}
    }}
  }}
}});

// 4. Hit-window efficiency chart
const effC={eff_c_json}, effG={eff_g_json};
mkChart('effChart',{{
  type:'line',
  data:{{datasets:[
    {{label:'E[gap|hit]/window',data:effC.map((p,i)=>({{x:p,y:effG[i]}})),
      borderColor:'#4EFFA0',borderWidth:2,pointRadius:0,fill:true,
      backgroundColor:'rgba(78,255,160,0.07)',tension:0}}
  ]}},
  options:{{
    responsive:true,maintainAspectRatio:false,animation:{{duration:500}},
    plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#111519',
      borderColor:'#191D27',borderWidth:1,titleColor:tx,bodyColor:'#BFC8DC',
      callbacks:{{title:i=>'conf = '+(i[0].parsed.x*100).toFixed(1)+'%',
                 label:c=>'window used: '+(c.parsed.y*100).toFixed(1)+'% on hit'}}}}}},
    scales:{{
      x:{{type:'linear',min:0,max:1,grid:{{color:gc}},ticks:{{color:tx,font}},
         title:{{display:true,text:'confidence p',color:tx,font}}}},
      y:{{min:0,max:1,grid:{{color:gc}},ticks:{{color:tx,font,callback:v=>(v*100).toFixed(0)+'%'}},
         title:{{display:true,text:'fraction of window budget used at hit',color:tx,font}}}}
    }}
  }}
}});
</script>
</body>
</html>"""

with open(args.out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"  Chart saved -> {args.out}")
print()
print("=" * 68)
print(f"  Done.  Found: {sum(len(s['found']) for s in sessions.values())}  |  "
      f"Total hashes: {sum(s['hashes'] for s in sessions.values()):,}")
print("=" * 68)
