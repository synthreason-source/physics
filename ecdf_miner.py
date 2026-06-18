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
# CURVE A — Exp(1)
# PDF: e^-x   mean=1   covers low nonces, heavy at 0
# ----------------------------
def step_exp1() -> float:
    return -math.log(1.0 - np.random.random())


# ----------------------------
# CURVE B — Gamma(2, 1)
# PDF: x·e^-x   mean=2   peak at x=1, covers mid nonces
# Sum of two Exp(1) draws — shifts mass rightward
# The two curves are orthogonal in probability mass:
#   Exp(1)     concentrates near 0
#   Gamma(2,1) concentrates near 1-3
# Together they tile the nonce line with less overlap
# ----------------------------
def step_gamma2() -> float:
    return step_exp1() + step_exp1()   # Erlang(2,1) = Gamma(2,1)


# ----------------------------
# PAIRWISE MEAN BALANCER
# ----------------------------
def pairwise_balance_block(block: list[float]) -> list[float]:
    n = len(block)
    if n < 2:
        return block
    paired = [(block[i] + block[i + 1]) / 2.0 for i in range(n - 1)]
    paired.append(block[-1])
    raw_sum  = sum(block)
    pair_sum = sum(paired)
    if pair_sum == 0:
        return block
    scale = raw_sum / pair_sum
    return [v * scale for v in paired]


# ----------------------------
# CUMULATIVE MEAN EQUALISER
# drift correction so each curve stays on its own mean line
# Curve A target mean: 1 per step
# Curve B target mean: 2 per step
# ----------------------------
def equalised_step(step_fn, x: float, n: int,
                   mean: float, alpha: float = 1.0) -> float:
    raw      = step_fn()
    expected = mean * float(n)
    drift    = x - expected
    balance  = alpha * drift / (n + 1)
    return max(raw - balance, 1e-9)


# ----------------------------
# DUAL CURVE WALK
# Two walkers run in lockstep:
#   Walker A — Exp(1),     mean step = 1
#   Walker B — Gamma(2,1), mean step = 2
# Each step checks both nonces.
# They explore different regions of nonce space.
# ----------------------------
def walk_dual(header, target, steps, window=8, alpha=1.0):
    xa, xb = 0.0, 0.0
    start = time.time()
    i = 0

    wins_a, wins_b = [], []   # track winning nonces per curve

    while i < steps:
        # --- build balanced blocks for both curves ---
        raw_a = [equalised_step(step_exp1,  xa, max(i,1), mean=1.0, alpha=alpha)
                 for _ in range(window)]
        raw_b = [equalised_step(step_gamma2, xb, max(i,1), mean=2.0, alpha=alpha)
                 for _ in range(window)]

        block_a = pairwise_balance_block(raw_a)
        block_b = pairwise_balance_block(raw_b)

        # undo raw additions (equalised_step doesn't advance x)
        for sa, sb in zip(block_a, block_b):
            if i >= steps:
                break

            xa += sa
            xb += sb

            nonce_a = int(xa)
            nonce_b = int(xb)

            # check curve A
            h = dsha(header, nonce_a)
            if int.from_bytes(h, "big") < target:
                wins_a.append(nonce_a)
                return {
                    "curve"   : "A — Exp(1)",
                    "nonce"   : nonce_a,
                    "hash"    : h.hex(),
                    "step"    : i,
                    "xa"      : xa,
                    "xb"      : xb,
                    "wins_a"  : wins_a,
                    "wins_b"  : wins_b,
                    "time"    : time.time() - start,
                }

            # check curve B (skip if same nonce — no wasted hash)
            if nonce_b != nonce_a:
                h = dsha(header, nonce_b)
                if int.from_bytes(h, "big") < target:
                    wins_b.append(nonce_b)
                    return {
                        "curve"   : "B — Gamma(2,1)",
                        "nonce"   : nonce_b,
                        "hash"    : h.hex(),
                        "step"    : i,
                        "xa"      : xa,
                        "xb"      : xb,
                        "wins_a"  : wins_a,
                        "wins_b"  : wins_b,
                        "time"    : time.time() - start,
                    }

            i += 1

    return {
        "curve" : None,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "time"  : time.time() - start,
    }


# ----------------------------
# MULTI-TRIAL — collect win distribution
# ----------------------------
def multi_trial(header, target, trials=200, steps_per=2_000_000,
                window=8, alpha=1.0):
    wins_a, wins_b = [], []
    for t in range(trials):
        r = walk_dual(header, target, steps_per, window=window, alpha=alpha)
        if r["curve"] and "A" in r["curve"]:
            wins_a.append(r["nonce"])
        elif r["curve"] and "B" in r["curve"]:
            wins_b.append(r["nonce"])
        if (t + 1) % 10 == 0:
            print(f"  trial {t+1}/{trials}  wins_a={len(wins_a)}  wins_b={len(wins_b)}")
    return wins_a, wins_b


# ----------------------------
# VISUAL
# ----------------------------
def make_html(out_path, wins_a, wins_b, window=8):
    x_vals = np.linspace(0, 8, 500)

    # theoretical PDFs
    pdf_exp   = [round(math.exp(-v),                  8) for v in x_vals]
    pdf_gamma = [round(v * math.exp(-v),               8) for v in x_vals]
    cdf_exp   = [round(1 - math.exp(-v),               8) for v in x_vals]
    cdf_gamma = [round(1 - math.exp(-v) * (1 + v),    8) for v in x_vals]

    # normalise winning nonces into [0,8] for overlay
    def norm(ns):
        if not ns:
            return []
        mx = max(ns) or 1
        return [round(n / mx * 8, 4) for n in sorted(ns)]

    wa = norm(wins_a)
    wb = norm(wins_b)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Dual Curve Competing Nonce</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body  {{ background:#0d0d0d; color:#eee; font-family:monospace; padding:24px; }}
  canvas{{ max-width:900px; margin:24px auto; display:block; }}
  h2    {{ text-align:center; }}
  p     {{ text-align:center; color:#aaa; font-size:.85em; }}
  .legend {{ display:flex; gap:32px; justify-content:center; margin:8px; font-size:.9em; }}
  .dot  {{ display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:6px; }}
</style>
</head>
<body>
<h2>⚡ Dual Curve Competing Nonce Walker</h2>
<p>
  Curve A: <b style="color:cyan">Exp(1)</b> — mean step 1, covers low nonces &nbsp;|&nbsp;
  Curve B: <b style="color:orange">Gamma(2,1)</b> — mean step 2, covers mid nonces
</p>
<div class="legend">
  <span><span class="dot" style="background:cyan"></span>Curve A wins: {len(wins_a)}</span>
  <span><span class="dot" style="background:orange"></span>Curve B wins: {len(wins_b)}</span>
</div>

<canvas id="pdfChart"></canvas>
<canvas id="cdfChart"></canvas>
<canvas id="winChart"></canvas>

<script>
const x8     = Array.from({{length:500}}, (_,i)=>(i*8/500).toFixed(3));
const pdfExp   = {pdf_exp};
const pdfGamma = {pdf_gamma};
const cdfExp   = {cdf_exp};
const cdfGamma = {cdf_gamma};
const winsA  = {wa};
const winsB  = {wb};

const opts = (title) => ({{
  animation: false,
  plugins: {{ title: {{ display:true, text:title, color:"#eee" }} }},
  scales: {{ x: {{ ticks:{{color:"#aaa"}} }}, y: {{ ticks:{{color:"#aaa"}} }} }}
}});

new Chart(document.getElementById("pdfChart"), {{
  type:"line",
  data:{{
    labels: x8,
    datasets:[
      {{label:"Exp(1) PDF",     data:pdfExp,   borderColor:"cyan",   pointRadius:0, tension:.3}},
      {{label:"Gamma(2,1) PDF", data:pdfGamma, borderColor:"orange", pointRadius:0, tension:.3}}
    ]
  }},
  options: opts("PDF — Exp(1) vs Gamma(2,1): competing nonce regions")
}});

new Chart(document.getElementById("cdfChart"), {{
  type:"line",
  data:{{
    labels: x8,
    datasets:[
      {{label:"Exp(1) CDF",     data:cdfExp,   borderColor:"cyan",   pointRadius:0, tension:.3}},
      {{label:"Gamma(2,1) CDF", data:cdfGamma, borderColor:"orange", pointRadius:0, tension:.3}}
    ]
  }},
  options: opts("CDF — probability mass coverage per curve")
}});

new Chart(document.getElementById("winChart"), {{
  type:"scatter",
  data:{{
    datasets:[
      {{
        label:"Curve A wins (Exp(1))",
        data: winsA.map((v,i)=>( {{x:v, y:0.1}} )),
        backgroundColor:"cyan", pointRadius:3
      }},
      {{
        label:"Curve B wins (Gamma(2,1))",
        data: winsB.map((v,i)=>( {{x:v, y:0.2}} )),
        backgroundColor:"orange", pointRadius:3
      }}
    ]
  }},
  options: opts("Winning nonce positions — normalised (A low, B mid)")
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
    ap.add_argument("--diff",    type=int,   required=True)
    ap.add_argument("--steps",   type=int,   default=500_000_000)
    ap.add_argument("--window",  type=int,   default=8)
    ap.add_argument("--alpha",   type=float, default=1.0)
    ap.add_argument("--trials",  type=int,   default=0,
                    help="Run N trials to collect win distribution (0=single run)")
    ap.add_argument("--out",     type=str,   default="dual_curve.html")
    args = ap.parse_args()

    header = b"demo-header"
    target = 1 << (256 - args.diff)

    print(f"[+] Dual curve competing nonce walker")
    print(f"[+] Curve A: Exp(1) mean=1  |  Curve B: Gamma(2,1) mean=2")
    print(f"[+] diff={args.diff}  window={args.window}  alpha={args.alpha}")

    if args.trials > 0:
        print(f"[+] Running {args.trials} trials...")
        wins_a, wins_b = multi_trial(
            header, target,
            trials=args.trials,
            steps_per=args.steps,
            window=args.window,
            alpha=args.alpha
        )
        print(f"\n[+] Curve A wins: {len(wins_a)}")
        print(f"[+] Curve B wins: {len(wins_b)}")
        make_html(args.out, wins_a, wins_b, window=args.window)

    else:
        result = walk_dual(header, target, args.steps,
                           window=args.window, alpha=args.alpha)
        make_html(args.out, result.get("wins_a",[]),
                  result.get("wins_b",[]), window=args.window)

        if result["curve"]:
            print(f"\n[+] HIT — {result['curve']}")
            print(f"  nonce : {result['nonce']}")
            print(f"  xa    : {result['xa']:.4f}")
            print(f"  xb    : {result['xb']:.4f}")
            print(f"  step  : {result['step']}")
            print(f"  time  : {result['time']:.4f}s")
            print(f"  hash  : {result['hash']}")
        else:
            print(f"[-] no hit  ({result['time']:.2f}s)")

    print(f"[+] HTML written → {args.out}")


if __name__ == "__main__":
    main()
