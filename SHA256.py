"""
gap_ecdf_chart.py
=================
Mines SHA-256 PoW nonces at all specified difficulties, computes inter-nonce
gap ECDFs normalised by expected mean, then writes a self-contained HTML file
with an interactive Chart.js ECDF chart - no external data files needed.

Usage
-----
  python gap_ecdf_chart.py
  python gap_ecdf_chart.py --budget 500000 --diff 8 12 16 20
  python gap_ecdf_chart.py --header "MY_BLOCK" --out ecdf.html
"""

import argparse
import hashlib
import json
import math
import struct
import time

import numpy as np
from scipy.stats import kstest

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int,       default=15_000_000)
parser.add_argument("--diff",   type=int, nargs="+", default=[8,12,16,24,28])
parser.add_argument("--header", type=str,       default="PROTO_BLOCK_v1")
parser.add_argument("--out",    type=str,       default="gap_ecdf.html")
args = parser.parse_args()

HEADER = args.header.encode()
DIFFS  = sorted(args.diff)
BUDGET = args.budget

# ─── Miner ────────────────────────────────────────────────────────────────────
def mine(header: bytes, diff_bits: int, budget: int):
    target = 1 << (256 - diff_bits)
    nonces = []
    t0 = time.perf_counter()
    for nonce in range(budget):
        data = header + struct.pack("<I", nonce)
        h    = hashlib.sha256(hashlib.sha256(data).digest()).digest()
        if int.from_bytes(h, "big") < target:
            nonces.append(nonce)
    elapsed = time.perf_counter() - t0
    return np.array(nonces, dtype=np.int64), elapsed

# ─── Mine + analyse ───────────────────────────────────────────────────────────
print("=" * 60)
print("  SHA-256 PoW Gap ECDF")
print("=" * 60)
print(f"  Header : {args.header}")
print(f"  Budget : {BUDGET:,}  |  Diffs : {DIFFS}\n")

results   = {}   # diff -> analysis dict
skipped   = []

for diff in DIFFS:
    exp = BUDGET / (2 ** diff)
    print(f"  diff={diff:2d}  (p~1/{2**diff:,}, expected~{exp:.1f}) ...", end="", flush=True)
    nonces, elapsed = mine(HEADER, diff, BUDGET)
    print(f"  {len(nonces)} found  ({elapsed:.1f}s)")

    if len(nonces) < 2:
        print(f"           >> only {len(nonces)} nonce - skipping")
        skipped.append(diff)
        continue

    gaps     = np.diff(nonces).astype(float)
    mean_gap = gaps.mean()
    norm     = np.sort(gaps / mean_gap)
    n        = len(norm)
    cv       = float(gaps.std() / mean_gap)
    ks_stat, ks_p = kstest(norm, "expon")

    # Downsample ECDF to ≤400 points for the chart
    step   = max(1, n // 400)
    idx    = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    ecdf_x = [0.0] + [round(float(norm[i]), 4) for i in idx]
    ecdf_y = [0.0] + [round((i + 1) / n, 4)    for i in idx]

    results[diff] = {
        "n":       n,
        "cv":      round(cv, 4),
        "ks_stat": round(float(ks_stat), 4),
        "ks_p":    round(float(ks_p), 4),
        "mean":    round(float(mean_gap), 1),
        "std":     round(float(gaps.std()), 1),
        "ecdf_x":  ecdf_x,
        "ecdf_y":  ecdf_y,
    }
    verdict = "PASS" if ks_p > 0.05 else "FAIL"
    print(f"           CV={cv:.4f}  KS={ks_stat:.4f}  p={ks_p:.4f}  -> {verdict}")

if not results:
    print("\nNo difficulties had enough nonces. Raise --budget or lower --diff.")
    raise SystemExit(1)

# Theoretical Exp(1) CDF
theory_x = [round(x, 3) for x in np.linspace(0, 5, 200).tolist()]
theory_y = [round(1 - math.exp(-x), 5) for x in theory_x]

# ─── Colour palette (one per diff) ────────────────────────────────────────────
PALETTE = ["#1D9E75", "#7F77DD", "#EF9F27", "#E24B4A",
           "#378ADD", "#D85A30", "#5DCAA5", "#F4C0D1"]
diff_color = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(sorted(results))}

# ─── Build Chart.js datasets JSON ────────────────────────────────────────────
datasets = []

# Exp(1) theory line
datasets.append({
    "label":           "Exp(1) - Poisson",
    "data":            [{"x": x, "y": y} for x, y in zip(theory_x, theory_y)],
    "borderColor":     "#888780",
    "borderWidth":     1.6,
    "borderDash":      [6, 4],
    "pointRadius":     0,
    "fill":            False,
    "tension":         0,
    "order":           0,
})

for i, diff in enumerate(sorted(results)):
    r = results[diff]
    datasets.append({
        "label":       f"diff={diff} ECDF  (n={r['n']:,})",
        "data":        [{"x": x, "y": y} for x, y in zip(r["ecdf_x"], r["ecdf_y"])],
        "borderColor": diff_color[diff],
        "borderWidth": 2.0,
        "pointRadius": 0,
        "fill":        False,
        "tension":     0,
        "stepped":     "after",
        "order":       i + 1,
    })

datasets_json = json.dumps(datasets)

# ─── KS table rows ────────────────────────────────────────────────────────────
def ks_rows_html():
    rows = []
    for diff in sorted(results):
        r = results[diff]
        col     = diff_color[diff]
        verdict = "[PASS] Poisson" if r["ks_p"] > 0.05 else "[FAIL] reject"
        vclass  = "pass" if r["ks_p"] > 0.05 else "fail"
        rows.append(f"""
        <tr>
          <td><span class="dot" style="background:{col}"></span></td>
          <td>diff={diff}</td>
          <td>n={r['n']:,}</td>
          <td>CV={r['cv']:.4f}</td>
          <td>KS={r['ks_stat']:.4f}</td>
          <td>p={r['ks_p']:.4f}</td>
          <td class="{vclass}">{verdict}</td>
        </tr>""")
    return "\n".join(rows)

# ─── Legend items ─────────────────────────────────────────────────────────────
def legend_html():
    items = []
    items.append('<span class="leg-item"><span class="leg-line" style="background:#888780;opacity:.6"></span>Exp(1) - Poisson</span>')
    for diff in sorted(results):
        col = diff_color[diff]
        items.append(
            f'<span class="leg-item"><span class="leg-line" style="background:{col}"></span>'
            f'diff={diff} ECDF</span>')
    return "\n".join(items)

# ─── Full HTML ────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gap ECDF - SHA-256 PoW</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0E0F11;color:#E2E0D8;font-family:monospace;padding:28px 32px;min-height:100vh}}
  h1{{font-size:18px;font-weight:500;margin-bottom:4px}}
  .sub{{font-size:12px;color:#888780;margin-bottom:20px}}

  .cards{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:22px}}
  .card{{background:#17191D;border:0.5px solid #2A2C32;border-radius:10px;
         padding:12px 16px;flex:1;min-width:130px}}
  .card .lbl{{font-size:11px;color:#888780;margin-bottom:3px}}
  .card .val{{font-size:22px;font-weight:500}}
  .card .hint{{font-size:10px;color:#5F5E5A;margin-top:2px}}

  .chart-wrap{{position:relative;width:100%;height:360px;
               background:#17191D;border:0.5px solid #2A2C32;border-radius:10px;
               padding:16px}}
  canvas{{width:100%!important}}

  .legend{{display:flex;flex-wrap:wrap;gap:16px;margin:14px 0;font-size:12px;color:#888780}}
  .leg-item{{display:flex;align-items:center;gap:6px}}
  .leg-line{{width:22px;height:2.5px;border-radius:2px;flex-shrink:0}}

  .ks-box{{background:#17191D;border:0.5px solid #2A2C32;border-radius:10px;
           padding:14px 18px;margin-top:18px}}
  .ks-title{{font-size:11px;color:#888780;letter-spacing:.05em;margin-bottom:10px}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{text-align:left;color:#5F5E5A;font-weight:400;padding:0 8px 6px 0;border-bottom:0.5px solid #2A2C32}}
  td{{padding:6px 8px 6px 0;color:#888780;border-bottom:0.5px solid #17191D}}
  tr:last-child td{{border-bottom:none}}
  .dot{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px}}
  .pass{{color:#1D9E75;font-weight:500}}
  .fail{{color:#E24B4A;font-weight:500}}

  .note{{font-size:11px;color:#5F5E5A;margin-top:16px;line-height:1.6}}
</style>
</head>
<body>

<h1>Gap ECDF vs exponential (Poisson) fit</h1>
<p class="sub">
  header: {args.header} &nbsp;|&nbsp;
  budget: {BUDGET:,} nonces &nbsp;|&nbsp;
  diffs: {', '.join(str(d) for d in sorted(results))} &nbsp;|&nbsp;
  skipped: {skipped if skipped else 'none (insufficient hits)'}
</p>

<div class="cards">
  <div class="card">
    <div class="lbl">nonces scanned</div>
    <div class="val">{BUDGET//1000}k</div>
    <div class="hint">double SHA-256</div>
  </div>
  {''.join(f"""
  <div class="card">
    <div class="lbl">diff={d}</div>
    <div class="val" style="color:{diff_color[d]}">{results[d]['n']:,}</div>
    <div class="hint">valid nonces found</div>
  </div>""" for d in sorted(results))}
  <div class="card">
    <div class="lbl">KS verdict</div>
    <div class="val {'pass' if all(r['ks_p']>0.05 for r in results.values()) else 'fail'}">
      {'All pass' if all(r['ks_p']>0.05 for r in results.values()) else 'Some fail'}
    </div>
    <div class="hint">H0: gaps ~ Exp(1)</div>
  </div>
</div>

<div class="chart-wrap">
  <canvas id="ecdfChart"></canvas>
</div>

<div class="legend">
{legend_html()}
</div>

<div class="ks-box">
  <div class="ks-title">KS TEST &nbsp;--&nbsp; H0: inter-nonce gaps follow Exp(1)  (memoryless Poisson process)</div>
  <table>
    <thead>
      <tr>
        <th></th><th>diff</th><th>n</th><th>CV</th><th>KS stat</th><th>p-value</th><th>verdict</th>
      </tr>
    </thead>
    <tbody>
{ks_rows_html()}
    </tbody>
  </table>
</div>

<p class="note">
  CV = std/mean &nbsp;(Poisson CV ~ 1.0 = memoryless spacing). &nbsp;
  Gaps normalised by expected mean before plotting. &nbsp;
  Dashed line = theoretical Exp(1) CDF. &nbsp;
  KS p &gt; 0.05 -&gt; cannot reject Poisson hypothesis.
</p>

<script>
const isDark = true;
const gridCol = 'rgba(255,255,255,0.06)';
const tx2 = '#888780';

const datasets = {datasets_json};

new Chart(document.getElementById('ecdfChart'), {{
  type: 'line',
  data: {{ datasets }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    animation: {{ duration: 700 }},
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor: '#17191D',
        borderColor: '#2A2C32',
        borderWidth: 0.5,
        titleColor: tx2,
        bodyColor: tx2,
        callbacks: {{
          title: items => 'gap/mean = ' + items[0].parsed.x.toFixed(3),
          label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(3)
        }}
      }}
    }},
    scales: {{
      x: {{
        type: 'linear',
        min: 0, max: 4.0,
        grid: {{ color: gridCol }},
        ticks: {{ color: tx2, font: {{ size: 11, family: 'monospace' }} }},
        title: {{ display: true, text: 'gap / expected mean', color: tx2, font: {{ size: 12, family: 'monospace' }} }}
      }},
      y: {{
        min: 0, max: 1.05,
        grid: {{ color: gridCol }},
        ticks: {{
          color: tx2,
          font: {{ size: 11, family: 'monospace' }},
          callback: v => v.toFixed(1)
        }},
        title: {{ display: true, text: 'cumulative probability', color: tx2, font: {{ size: 12, family: 'monospace' }} }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

# ─── Write ────────────────────────────────────────────────────────────────────
with open(args.out, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nChart saved → {args.out}")