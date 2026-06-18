import argparse
import hashlib
import math
import struct
import time
import numpy as np


# ----------------------------
# CORE HASH
# ----------------------------
def dsha(header: bytes, nonce: int) -> bytes:
    d = header + struct.pack("<Q", nonce)
    return hashlib.sha256(hashlib.sha256(d).digest()).digest()


# ----------------------------
# TRUE Exp(1) STEP
# ----------------------------
def exp1():
    return -math.log(1.0 - np.random.random())


# ----------------------------
# PAIRWISE MEAN BALANCER
# Takes a block of N raw Exp(1) draws,
# balances them by replacing each with the
# mean of adjacent pairs, normalising the
# block so its total span is preserved.
# ----------------------------
def pairwise_balance_block(block: list[float]) -> list[float]:
    n = len(block)
    if n < 2:
        return block

    # Step 1: pairwise means
    paired = [(block[i] + block[i + 1]) / 2.0 for i in range(n - 1)]
    # keep last raw value to maintain length
    paired.append(block[-1])

    # Step 2: normalise so block sum is preserved
    raw_sum = sum(block)
    pair_sum = sum(paired)
    if pair_sum == 0:
        return block
    scale = raw_sum / pair_sum
    balanced = [v * scale for v in paired]

    return balanced


# ----------------------------
# EXP(1) LINE WALK — windowed pairwise balancing
# ----------------------------
def walk_exp_line(header, target, steps, window=8):
    x = 0.0
    start = time.time()
    i = 0

    while i < steps:
        # Draw a full window of raw Exp(1) steps
        raw_block = [exp1() for _ in range(window)]

        # Balance pairwise means across block
        block = pairwise_balance_block(raw_block)

        for step_val in block:
            if i >= steps:
                break

            x += step_val
            nonce = int(x)

            h = dsha(header, nonce)
            val = int.from_bytes(h, "big")

            if val < target:
                return {
                    "nonce": nonce,
                    "hash": h.hex(),
                    "step": i,
                    "x": x,
                    "raw_block": raw_block,
                    "balanced_block": block,
                    "time": time.time() - start,
                }

            i += 1

    return None


# ----------------------------
# VISUAL
# ----------------------------
def make_html(out_path, window=8):
    x_vals = np.linspace(0, 5, 400)
    pdf = [math.exp(-v) for v in x_vals]
    cdf = [1 - math.exp(-v) for v in x_vals]

    # simulate 200 blocks for visualisation
    raw_series, balanced_series = [], []
    for _ in range(200 // window + 1):
        raw = [exp1() for _ in range(window)]
        bal = pairwise_balance_block(raw)
        raw_series.extend(raw)
        balanced_series.extend(bal)

    raw_series = [round(v, 6) for v in raw_series[:200]]
    balanced_series = [round(v, 6) for v in balanced_series[:200]]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Exp(1) Pairwise Mean Balancer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ background: #0d0d0d; color: #eee; font-family: monospace; padding: 20px; }}
  canvas {{ max-width: 900px; margin: 24px auto; display: block; }}
  h2 {{ text-align: center; color: #fff; }}
  p  {{ text-align: center; color: #aaa; font-size: 0.85em; }}
</style>
</head>
<body>
<h2>🌊 Exp(1) Line Walker — Pairwise Mean Balancing per Window</h2>
<p>Window size: {window} &nbsp;|&nbsp; Each block balanced so pairwise means are preserved and sum is normalised</p>

<canvas id="distChart"></canvas>
<canvas id="stepsChart"></canvas>

<script>
const pdf = {pdf};
const cdf = {cdf};
const raw = {raw_series};
const balanced = {balanced_series};
const labels400 = Array.from({{length: pdf.length}}, (_, i) => (i * 5 / pdf.length).toFixed(2));
const labels200 = Array.from({{length: raw.length}}, (_, i) => i);

new Chart(document.getElementById("distChart"), {{
    type: "line",
    data: {{
        labels: labels400,
        datasets: [
            {{ label: "Exp(1) PDF", data: pdf, borderColor: "cyan",    pointRadius: 0, tension: 0.3 }},
            {{ label: "Exp(1) CDF", data: cdf, borderColor: "magenta", pointRadius: 0, tension: 0.3 }}
        ]
    }},
    options: {{
        plugins: {{ title: {{ display: true, text: "Exp(1) Distribution", color: "#eee" }} }},
        scales: {{ x: {{ ticks: {{ color:"#aaa" }} }}, y: {{ ticks: {{ color:"#aaa" }} }} }}
    }}
}});

new Chart(document.getElementById("stepsChart"), {{
    type: "line",
    data: {{
        labels: labels200,
        datasets: [
            {{ label: "Raw Exp(1) steps",     data: raw,      borderColor: "#888",   pointRadius: 0, tension: 0.2 }},
            {{ label: "Pairwise balanced",     data: balanced, borderColor: "orange", pointRadius: 0, tension: 0.4 }}
        ]
    }},
    options: {{
        plugins: {{ title: {{ display: true, text: "Raw vs Pairwise Mean Balanced Step Sizes", color: "#eee" }} }},
        scales: {{ x: {{ ticks: {{ color:"#aaa" }} }}, y: {{ ticks: {{ color:"#aaa" }} }} }}
    }}
}});
</script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ----------------------------
# MAIN
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff",   type=int, required=True)
    ap.add_argument("--steps",  type=int, default=500_000_000)
    ap.add_argument("--window", type=int, default=8,
                    help="Block size for pairwise mean balancing")
    ap.add_argument("--out",    type=str, default="exp_line.html")
    args = ap.parse_args()

    header = b"demo-header"
    target = 1 << (256 - args.diff)

    print(f"[+] Exp(1) line walker — pairwise mean balancing per window")
    print(f"[+] diff={args.diff}  steps={args.steps}  window={args.window}")

    result = walk_exp_line(header, target, args.steps, window=args.window)
    make_html(args.out, window=args.window)
    print(f"[+] HTML written → {args.out}")

    if result:
        print("\n[+] HIT FOUND")
        print(f"  nonce          : {result['nonce']}")
        print(f"  x              : {result['x']:.4f}")
        print(f"  step           : {result['step']}")
        print(f"  time           : {result['time']:.4f}s")
        print(f"  hash           : {result['hash']}")
        print(f"  raw block      : {[round(v,4) for v in result['raw_block']]}")
        print(f"  balanced block : {[round(v,4) for v in result['balanced_block']]}")
    else:
        print("[-] no hit in allotted steps")


if __name__ == "__main__":
    main()
