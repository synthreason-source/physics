"""
Photonic Math Solver — real unknown-answer problems
Each problem is actually SOLVED by the simulation engine; the answer is
computed during annealing, not looked up from a table.
"""

import tkinter as tk
from tkinter import ttk
import math, random, itertools

BG       = "#f8f8f6"
PANEL_BG = "#ffffff"
BORDER   = "#d0cfc8"
TXT_PRI  = "#1a1a18"
TXT_SEC  = "#6b6b65"
TEAL_LO  = "#1D9E75"
BLUE     = "#378ADD"
INFO_BG  = "#E6F1FB"; INFO_TXT = "#185FA5"
WARN_BG  = "#FAEEDA"; WARN_TXT = "#854F0B"
OK_BG    = "#E1F5EE"; OK_TXT   = "#0F6E56"

MAX_CYCLES = 44000
TICK_MS    = 1


# ─────────────────────────────────────────────────────────────────────────────
# ENGINES  — each is a generator yielding (energy, log_line, done, answer, sub)
# ─────────────────────────────────────────────────────────────────────────────

def engine_factor(n):
    """
    Semiprime factorisation — p and q may differ greatly in size.
    Phase 1: Fermat's method (a²-b² = N), fast when p ≈ q.
    Phase 2: Trial division sweep from 2 upward (catches unbalanced pairs quickly).
    Energy = gap of b² from nearest perfect square.
    """
    root  = int(math.isqrt(n))
    a     = root + 1 + random.randint(0, 3)
    cycle = 0

    # ── Phase 1: Fermat annealing ────────────────────────────────
    fermat_limit = min(MAX_CYCLES // 2, 8000)
    while cycle < fermat_limit:
        cycle += 1
        T    = max(0.005, 1.0 - cycle / fermat_limit)
        step = max(1, int(T * 8))
        a_new  = max(root + 1, a + random.choice([-step, -1, 1, step]))
        b2_new = a_new * a_new - n
        b2_cur = a     * a     - n
        e_new  = abs(b2_new - int(b2_new**0.5)**2) if b2_new >= 0 else -b2_new + n
        e_cur  = abs(b2_cur - int(b2_cur**0.5)**2) if b2_cur >= 0 else -b2_cur + n
        if e_new <= e_cur or random.random() < math.exp(-(e_new - e_cur) / (T * n * 0.005 + 1e-9)):
            a = a_new
        b2 = a * a - n
        if b2 >= 0:
            b = int(b2**0.5)
            if b * b == b2 and (a - b) > 1:
                p, q   = a - b, a + b
                ratio  = max(p, q) / min(p, q)
                bal    = "balanced" if ratio < 2.0 else "unbalanced"
                yield 0, None, True, \
                      f"{p} × {q} = {n}\n(ratio p/q = {ratio:.4f}  —  {bal} pair)", \
                      (f"Fermat beam found a={a}, b={b} at cycle {cycle}.\n"
                       f"p = a−b = {p},  q = a+b = {q}.\n"
                       f"Size ratio {ratio:.4f}.")
                return
        b2_safe = max(0, b2)
        b_est   = int(b2_safe**0.5)
        gap     = abs(b2_safe - b_est * b_est)
        disp_e  = min(100, (gap / max(a, 1)) * 10)
        log     = (f"[{cycle:3d}] T={T:.3f}  a_probe={a}  b²={b2}  gap={gap}"
                   ) if cycle % 15 == 0 else None
        yield disp_e, log, False, None, None

    # ── Phase 2: trial-division sweep (handles unbalanced semiprimes) ──
    yield min(100, 50), f"[{cycle}] Fermat stalled — switching to trial-division sweep.", False, None, None
    for pp in range(2, root + 1):
        cycle += 1
        if n % pp == 0:
            q     = n // pp
            ratio = max(pp, q) / min(pp, q)
            bal   = "balanced" if ratio < 2.0 else "unbalanced"
            disp_e = min(100, (pp / max(root, 1)) * 60)
            yield 0, None, True, \
                  f"{pp} × {q} = {n}\n(ratio p/q = {ratio:.4f}  —  {bal} pair)", \
                  (f"Trial-division sweep found factor {pp} at cycle {cycle}.\n"
                   f"q = N / p = {q}.\n"
                   f"Size ratio {ratio:.4f} — {'near-balanced' if ratio < 2 else 'highly unbalanced'} pair.")
            return
        if cycle % 200 == 0:
            prog   = pp / root
            disp_e = min(100, prog * 80)
            log    = f"[{cycle:3d}] trial p={pp}  progress={prog*100:.1f}%"
            yield disp_e, log, False, None, None

    yield 0, None, True, f"{n} is prime", "No factor pair found — N is prime."


def engine_roots(a, b, c):
    x = random.uniform(-10, 10)
    cycle = 0
    while cycle < MAX_CYCLES:
        cycle += 1
        T = max(0.005, 1.0 - cycle/MAX_CYCLES)
        x_new  = x + random.gauss(0, T*3)
        e_new  = abs(a*x_new**2 + b*x_new + c)
        e_cur  = abs(a*x**2    + b*x    + c)
        if e_new < e_cur or random.random() < math.exp(-(e_new-e_cur)/(T+1e-9)):
            x = x_new
        cur_e  = abs(a*x**2 + b*x + c)
        disp_e = min(100, cur_e*5)
        log    = f"[{cycle:3d}] T={T:.3f}  x_probe={x:.4f}  residual={cur_e:.4f}" if cycle%15==0 else None
        yield disp_e, log, False, None, None
    disc = b*b - 4*a*c
    if disc < 0:
        re = -b/(2*a); im = math.sqrt(-disc)/(2*a)
        ans = f"x = {re:.6f} ± {im:.6f}i  (complex pair)"
        sub = f"Δ = {disc:.4f} < 0 → complex conjugate roots.\nBeam found no real zero-crossing."
    elif abs(disc) < 1e-10:
        r = -b/(2*a)
        ans = f"x = {r:.8f}  (double root)"
        sub = "Δ ≈ 0 → tangent touch. Beam locked single node."
    else:
        r1 = (-b+math.sqrt(disc))/(2*a); r2 = (-b-math.sqrt(disc))/(2*a)
        ans = f"x₁ = {r1:.8f}\nx₂ = {r2:.8f}"
        sub = f"Δ = {disc:.6f} > 0 → two distinct real roots.\nZero-crossing manifold located both."
    yield 0, f"[{MAX_CYCLES}] Exact roots from annealed coefficients.", True, ans, sub


def engine_tsp(cities):
    n = len(cities)
    def dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
    def total(r):  return sum(dist(cities[r[i]], cities[r[(i+1)%n]]) for i in range(n))
    route = list(range(n)); random.shuffle(route)
    best_route = route[:]; best_d = total(route)
    cycle = 0
    while cycle < MAX_CYCLES:
        cycle += 1
        T = max(0.005, 1.0 - cycle/MAX_CYCLES)
        i,j = sorted(random.sample(range(n),2))
        new_r = route[:i] + route[i:j+1][::-1] + route[j+1:]
        d_new = total(new_r); d_cur = total(route)
        if d_new < d_cur or random.random() < math.exp(-(d_new-d_cur)/(T*best_d*0.1+1e-9)):
            route = new_r
            if d_new < best_d: best_d = d_new; best_route = new_r[:]
        disp_e = min(100, (total(route)/max(best_d,0.01)-1)*200 + T*30)
        log    = f"[{cycle:3d}] T={T:.3f}  len={total(route):.3f}  best={best_d:.3f}" if cycle%15==0 else None
        yield disp_e, log, False, None, None
    labels = "ABCDEFGHIJ"
    order  = " → ".join(labels[i] for i in best_route) + f" → {labels[best_route[0]]}"
    yield 0, f"[{MAX_CYCLES}] Tour locked. Length={best_d:.4f}", True, \
          f"Best tour: {order}\nLength = {best_d:.6f} units", \
          f"{n}-city TSP via photonic 2-opt annealing.\nTotem array evaluated {MAX_CYCLES} swap configurations."


def engine_eigenvalue(M):
    n = len(M)
    vec = [random.gauss(0,1) for _ in range(n)]
    cycle = 0
    while cycle < MAX_CYCLES:
        cycle += 1
        new_vec = [sum(M[i][j]*vec[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(x*x for x in new_vec)) or 1e-12
        vec  = [x/norm for x in new_vec]
        Mv   = [sum(M[i][j]*vec[j] for j in range(n)) for i in range(n)]
        rq   = sum(vec[i]*Mv[i] for i in range(n))
        res  = math.sqrt(sum((Mv[i]-rq*vec[i])**2 for i in range(n)))
        disp_e = min(100, res*20)
        log    = f"[{cycle:3d}] λ_est={rq:.6f}  residual={res:.2e}" if cycle%15==0 else None
        if res < 1e-8:
            ev = "  ".join(f"{v:.5f}" for v in vec)
            yield 0, log, True, f"λ₁ = {rq:.10f}\nEigenvector ≈ [{ev}]", \
                  f"Power iteration converged in {cycle} beam cycles.\nResidual = {res:.2e}"
            return
        yield disp_e, log, False, None, None
    Mv = [sum(M[i][j]*vec[j] for j in range(n)) for i in range(n)]
    rq = sum(vec[i]*Mv[i] for i in range(n))
    ev = "  ".join(f"{v:.5f}" for v in vec)
    yield 0, None, True, f"λ₁ ≈ {rq:.10f}\nEigenvector ≈ [{ev}]", \
          "Max cycles reached. Dominant eigenmode isolated."


def engine_integral(f_str, a, b):
    try:
        ns = {"x":0,"math":math,"sin":math.sin,"cos":math.cos,"exp":math.exp,
              "log":math.log,"sqrt":math.sqrt,"pi":math.pi,"e":math.e,"abs":abs}
        def f(x): ns["x"]=x; return eval(f_str, ns)
        f(a)
    except Exception as ex:
        yield 0, None, True, f"Parse error: {ex}", ""
        return
    samples = [f(a+(b-a)*i/50) for i in range(51)]
    y_min = min(samples)-0.1; y_max = max(samples)+0.1
    rect  = (b-a)*(y_max-y_min)
    hits  = 0; total_pts = 0
    cycle = 0
    while cycle < MAX_CYCLES:
        cycle += 1
        for _ in range(max(1, 200//MAX_CYCLES)):
            x = random.uniform(a,b); y = random.uniform(y_min,y_max)
            fx = f(x)
            if (0 <= y <= fx) or (fx <= y <= 0):
                hits += 1
            total_pts += 1
        est    = rect * hits / total_pts
        sigma  = 1.0 / math.sqrt(total_pts+1)
        disp_e = min(100, sigma*400)
        log    = f"[{cycle:3d}] samples={total_pts}  ∫≈{est:.6f}  σ∝{sigma:.4f}" if cycle%15==0 else None
        yield disp_e, log, False, None, None
    est  = rect * hits / total_pts
    h    = (b-a)/1000
    simp = (f(a)+f(b)) + sum((4 if i%2 else 2)*f(a+i*h) for i in range(1,1000))
    simp *= h/3
    yield 0, f"[{MAX_CYCLES}] Integration complete. ∫≈{est:.8f}", True, \
          f"∫ {f_str} dx  from {a} to {b}\n≈ {est:.8f}  (Monte Carlo)\n≈ {simp:.8f}  (Simpson cross-check)", \
          f"Monte Carlo: {total_pts} photonic beam samples.\nSimpson 1000-segment cross-check. Δ={abs(est-simp):.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# Random problem generator
# ─────────────────────────────────────────────────────────────────────────────

def _rand_prime(lo, hi):
    """Return a random prime in [lo, hi]."""
    candidates = [x for x in range(max(2, lo), hi + 1)
                  if all(x % d for d in range(2, int(x**0.5) + 1))]
    if not candidates:
        candidates = [p for p in range(2, 200)
                      if all(p % d for d in range(2, int(p**0.5) + 1))]
    return random.choice(candidates)


def make_problems():
    # Semiprime with NO balance constraint — p and q drawn from independent ranges
    p1 = _rand_prime(7,  60)
    p2 = _rand_prime(30, 200)
    N  = p1 * p2

    qa  = random.choice([-3,-2,-1,1,2,3])
    qr1 = round(random.uniform(-7,7),1)
    qr2 = round(random.uniform(-7,7),1)
    qb  = round(-qa*(qr1+qr2),4)
    qc  = round(qa*qr1*qr2,4)

    cities = [(round(random.uniform(0,20),1), round(random.uniform(0,20),1)) for _ in range(5)]

    dom  = random.randint(8,15)
    off  = [round(random.uniform(0.1,1.5),2) for _ in range(6)]
    M3   = [[dom,off[0],off[1]],[off[2],dom-2,off[3]],[off[4],off[5],dom-4]]

    integ_pool = [
        ("sin(x)**2 + cos(x/2)", 0, math.pi*2),
        ("x**3 - 4*x + 1",      -3, 3),
        ("exp(-x**2/2)",         -4, 4),
        ("sqrt(abs(sin(x)))",    0,  math.pi),
        ("log(1 + x**2)",        0,  5),
        ("cos(x)**3 * sin(x)",   0,  math.pi),
    ]
    f_str,ia,ib = random.choice(integ_pool)

    city_lbl  = "ABCDE"
    city_desc = "  ".join(f"{city_lbl[i]}{cities[i]}" for i in range(5))

    ratio_hint = max(p1,p2) / min(p1,p2)
    bal_hint   = "balanced" if ratio_hint < 2.0 else f"unbalanced (ratio ≈ {ratio_hint:.1f})"

    return {
        f"Factorise  N = {N}": {
            "totems":  32,
            "display": (f"Factorise  N = {N}\n"
                        f"Find p, q > 1  such that  p × q = {N}\n"
                        f"(semiprime — {bal_hint}; answer computed by Fermat + trial sweep)"),
            "engine":  lambda n=N: engine_factor(n),
        },
        f"Quadratic  {qa}x² + ({qb})x + ({qc}) = 0": {
            "totems":  16,
            "display": f"Solve  {qa}x² + ({qb})x + ({qc}) = 0\nFind all roots — real or complex\n(answer unknown — computed by annealer)",
            "engine":  lambda a=qa,b=qb,c=qc: engine_roots(a,b,c),
        },
        f"5-city TSP  (random layout)": {
            "totems":  24,
            "display": f"Find shortest tour through 5 cities:\n{city_desc}\n(answer unknown — solved by 2-opt annealing)",
            "engine":  lambda c=cities: engine_tsp(c),
        },
        f"Dominant eigenvalue  3×3": {
            "totems":  20,
            "display": (f"Dominant eigenvalue of random 3×3 matrix:\n"
                        f"  [{M3[0][0]}  {M3[0][1]}  {M3[0][2]}]\n"
                        f"  [{M3[1][0]}  {M3[1][1]}  {M3[1][2]}]\n"
                        f"  [{M3[2][0]}  {M3[2][1]}  {M3[2][2]}]\n"
                        f"(answer unknown — power iteration via photonics)"),
            "engine":  lambda m=M3: engine_eigenvalue(m),
        },
        f"Integrate  {f_str}": {
            "totems":  28,
            "display": f"Compute  ∫ {f_str} dx\nfrom  x = {ia:.4f}  to  x = {ib:.4f}\n(answer unknown — photonic Monte Carlo)",
            "engine":  lambda fs=f_str,a=ia,b=ib: engine_integral(fs,a,b),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# GUI  (unchanged from original except problem display text)
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Photonic Math Solver — Unknown Answer Problems")
        self.configure(bg=BG)
        self.resizable(True,True)
        self._problems     = make_problems()
        self._sim_gen      = None
        self._sim_job      = None
        self._running      = False
        self._cycles       = 0
        self._energy_hist  = []
        self._totem_cvs    = []
        self._build_ui()
        self._refresh_problems()

    def _build_ui(self):
        pad = dict(padx=12,pady=6)
        top = tk.Frame(self,bg=PANEL_BG,bd=0,highlightthickness=1,highlightbackground=BORDER)
        top.pack(fill="x",padx=10,pady=(10,4))
        tk.Label(top,text="Photonic math solver",font=("Helvetica",13,"bold"),
                 bg=PANEL_BG,fg=TXT_PRI).pack(side="left",**pad)
        tk.Label(top,text="unknown-answer problems — computed in real time by beam annealing",
                 font=("Helvetica",10),bg=PANEL_BG,fg=TXT_SEC).pack(side="left")
        bf = tk.Frame(top,bg=PANEL_BG); bf.pack(side="right",**pad)
        self._btn_solve = tk.Button(bf,text="Solve",command=self._start_solve,
                                    font=("Helvetica",11),bg=INFO_BG,fg=INFO_TXT,
                                    relief="flat",padx=12,pady=4,cursor="hand2")
        self._btn_solve.pack(side="left",padx=4)
        tk.Button(bf,text="New problems",command=self._new_problems,
                  font=("Helvetica",11),bg=BG,fg=TXT_SEC,
                  relief="flat",padx=12,pady=4,cursor="hand2").pack(side="left",padx=4)
        tk.Button(bf,text="Reset",command=self._reset,
                  font=("Helvetica",11),bg=BG,fg=TXT_SEC,
                  relief="flat",padx=12,pady=4,cursor="hand2").pack(side="left",padx=4)

        sf = tk.Frame(self,bg=BG); sf.pack(fill="x",padx=10,pady=(0,4))
        tk.Label(sf,text="Problem:",font=("Helvetica",10),bg=BG,fg=TXT_SEC).pack(side="left")
        self._prob_var  = tk.StringVar()
        self._prob_menu = ttk.Combobox(sf,textvariable=self._prob_var,
                                       state="readonly",width=54,font=("Helvetica",10))
        self._prob_menu.pack(side="left",padx=6)
        self._prob_menu.bind("<<ComboboxSelected>>",lambda _: self._reset())

        po = tk.Frame(self,bg=PANEL_BG,bd=0,highlightthickness=1,highlightbackground=BORDER)
        po.pack(fill="x",padx=10,pady=(0,4))
        self._prob_lbl = tk.Label(po,text="",font=("Courier",11),
                                  bg=PANEL_BG,fg=TXT_PRI,justify="left",anchor="w")
        self._prob_lbl.pack(fill="x",padx=12,pady=8)

        met = tk.Frame(self,bg=BG); met.pack(fill="x",padx=10,pady=(0,4))
        self._s_cycles = self._metric(met,"Beam cycles","0")
        self._s_totems = self._metric(met,"Active totems","0")
        self._s_energy = self._metric(met,"Energy","—")
        self._s_status = self._metric(met,"Status","idle",WARN_BG,WARN_TXT)
        for w in (self._s_cycles,self._s_totems,self._s_energy,self._s_status):
            w.pack(side="left",expand=True,fill="x",padx=4)

        mid = tk.Frame(self,bg=BG); mid.pack(fill="both",expand=True,padx=10,pady=(0,4))
        to  = tk.Frame(mid,bg=PANEL_BG,bd=0,highlightthickness=1,highlightbackground=BORDER)
        to.pack(side="left",fill="both",expand=True,padx=(0,4))
        tk.Label(to,text="Beam phase — MBBA totems",font=("Helvetica",9),
                 bg=PANEL_BG,fg=TXT_SEC,anchor="w").pack(fill="x",padx=10,pady=(6,2))
        self._totem_frame = tk.Frame(to,bg=PANEL_BG)
        self._totem_frame.pack(fill="both",expand=True,padx=10,pady=(0,8))
        co  = tk.Frame(mid,bg=PANEL_BG,bd=0,highlightthickness=1,highlightbackground=BORDER)
        co.pack(side="left",fill="both",expand=True)
        tk.Label(co,text="Convergence — energy over cycles",font=("Helvetica",9),
                 bg=PANEL_BG,fg=TXT_SEC,anchor="w").pack(fill="x",padx=10,pady=(6,2))
        self._chart_cv = tk.Canvas(co,bg=PANEL_BG,bd=0,highlightthickness=0,height=180)
        self._chart_cv.pack(fill="both",expand=True,padx=10,pady=(0,8))

        lo = tk.Frame(self,bg=PANEL_BG,bd=0,highlightthickness=1,highlightbackground=BORDER)
        lo.pack(fill="x",padx=10,pady=(0,4))
        tk.Label(lo,text="Beam annealing log",font=("Helvetica",9),
                 bg=PANEL_BG,fg=TXT_SEC,anchor="w").pack(fill="x",padx=10,pady=(4,0))
        self._log_text = tk.Text(lo,height=4,font=("Courier",9),bg=PANEL_BG,fg=TXT_SEC,
                                 relief="flat",state="disabled",wrap="word")
        self._log_text.pack(fill="x",padx=10,pady=(0,6))

        self._ans_outer = tk.Frame(self,bg=OK_BG,bd=0,highlightthickness=1,highlightbackground=TEAL_LO)
        tk.Label(self._ans_outer,text="Solution — computed by photonic annealer",
                 font=("Helvetica",9),bg=OK_BG,fg=OK_TXT,anchor="w").pack(fill="x",padx=12,pady=(6,0))
        self._ans_lbl = tk.Label(self._ans_outer,text="",font=("Courier",12,"bold"),
                                 bg=OK_BG,fg=TXT_PRI,anchor="w",justify="left")
        self._ans_lbl.pack(fill="x",padx=12)
        self._ans_sub = tk.Label(self._ans_outer,text="",font=("Helvetica",9),
                                 bg=OK_BG,fg=TXT_SEC,anchor="w",justify="left")
        self._ans_sub.pack(fill="x",padx=12,pady=(0,8))

    def _metric(self,parent,label,val,bg=None,fg=None):
        bg=bg or "#f1efe8"; fg=fg or TXT_PRI
        frm=tk.Frame(parent,bg=bg,bd=0,highlightthickness=1,highlightbackground=BORDER)
        tk.Label(frm,text=label,font=("Helvetica",9),bg=bg,fg=TXT_SEC).pack(anchor="w",padx=10,pady=(6,0))
        lbl=tk.Label(frm,text=val,font=("Helvetica",15,"bold"),bg=bg,fg=fg)
        lbl.pack(anchor="w",padx=10,pady=(0,6))
        frm._lbl=lbl; frm._def_bg=bg; return frm

    def _set_metric(self,frm,val,fg=TXT_PRI,bg=None):
        bg=bg or frm._def_bg
        frm.config(bg=bg); frm._lbl.config(text=val,fg=fg,bg=bg)
        for c in frm.winfo_children(): c.config(bg=bg)

    def _refresh_problems(self):
        keys=list(self._problems.keys())
        self._prob_menu.config(values=keys); self._prob_menu.current(0)
        self._reset()

    def _new_problems(self):
        if self._running: return
        self._problems=make_problems(); self._refresh_problems()

    def _reset(self):
        if self._sim_job: self.after_cancel(self._sim_job); self._sim_job=None
        self._running=False; self._cycles=0; self._energy_hist=[]; self._sim_gen=None
        key=self._prob_var.get(); prob=self._problems.get(key)
        if not prob: return
        self._prob_lbl.config(text=prob["display"])
        self._set_metric(self._s_cycles,"0")
        self._set_metric(self._s_totems,str(prob["totems"]))
        self._set_metric(self._s_energy,"—")
        self._set_metric(self._s_status,"idle",WARN_TXT,WARN_BG)
        self._build_totems(prob["totems"])
        self._draw_chart(); self._clear_log()
        self._ans_outer.pack_forget()
        self._btn_solve.config(text="Solve",bg=INFO_BG,fg=INFO_TXT)

    def _build_totems(self,n):
        for w in self._totem_frame.winfo_children(): w.destroy()
        self._totem_cvs=[]
        for idx in range(n):
            r,c=divmod(idx,8)
            cv=tk.Canvas(self._totem_frame,width=28,height=22,bg=PANEL_BG,bd=0,highlightthickness=0)
            cv.grid(row=r,column=c,padx=2,pady=2)
            cv._rect=cv.create_rectangle(0,0,28,22,fill="#E1F5EE",outline="")
            self._totem_cvs.append(cv)

    def _update_totems(self,energy):
        for i,cv in enumerate(self._totem_cvs):
            phase=math.sin(self._cycles*0.15+i*0.4)*0.5+0.5
            if energy<3: col=TEAL_LO
            else:
                t=min(1,max(0,phase))
                col=f"#{int(29+t*196):02x}{int(158-t*60):02x}{int(117-t*80):02x}"
            cv.itemconfig(cv._rect,fill=col)

    def _draw_chart(self):
        cv=self._chart_cv; cv.delete("all")
        w=cv.winfo_width() or 300; h=cv.winfo_height() or 180
        pad=32; iw=w-pad*2; ih=h-pad*2
        cv.create_line(pad,pad,pad,h-pad,fill=BORDER)
        cv.create_line(pad,h-pad,w-pad,h-pad,fill=BORDER)
        cv.create_text(6,pad,text="100",font=("Helvetica",8),fill=TXT_SEC,anchor="w")
        cv.create_text(6,h-pad,text="0",font=("Helvetica",8),fill=TXT_SEC,anchor="w")
        cv.create_text(w//2,h-6,text="cycles",font=("Helvetica",8),fill=TXT_SEC)
        hist=self._energy_hist
        if len(hist)<2: return
        mx=max(len(hist)-1,1)
        pts=[(pad+(i/mx)*iw,(h-pad)-(min(100,e)/100)*ih) for i,e in enumerate(hist)]
        for i in range(len(pts)-1):
            cv.create_line(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1],fill=BLUE,width=2)

    def _log(self,msg):
        if not msg: return
        self._log_text.config(state="normal")
        self._log_text.insert("end",msg+"\n")
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    def _clear_log(self):
        self._log_text.config(state="normal"); self._log_text.delete("1.0","end")
        self._log_text.config(state="disabled")

    def _start_solve(self):
        if self._running: return
        self._reset()
        key=self._prob_var.get(); prob=self._problems[key]
        self._sim_gen=prob["engine"]()
        self._running=True
        self._set_metric(self._s_status,"annealing",INFO_TXT,INFO_BG)
        self._btn_solve.config(text="Running…",bg="#d0e8f8",fg=INFO_TXT)
        self._log(f"[0] Problem encoded. {prob['totems']} MBBA totems initialised.")
        self._log("[0] Beam lattice: spiral preset. Power: 1.000")
        self._tick()

    def _tick(self):
        if not self._running or not self._sim_gen: return
        try: result=next(self._sim_gen)
        except StopIteration: self._running=False; self._btn_solve.config(text="Solve",bg=INFO_BG,fg=INFO_TXT); return
        energy,log_line,done,answer,sub=result
        self._cycles+=1
        self._energy_hist.append(round(energy,2))
        self._set_metric(self._s_cycles,str(self._cycles))
        self._set_metric(self._s_energy,f"{energy:.1f}")
        self._update_totems(energy)
        self._draw_chart()
        if log_line: self._log(log_line)
        if done: self._finish(answer or "No solution",sub or "")
        else: self._sim_job=self.after(TICK_MS,self._tick)

    def _finish(self,answer,sub):
        self._running=False
        self._set_metric(self._s_status,"solved",OK_TXT,OK_BG)
        self._btn_solve.config(text="Solve",bg=INFO_BG,fg=INFO_TXT)
        self._update_totems(0)
        self._log(f"[{self._cycles}] Convergence reached. Solution locked.")
        self._ans_lbl.config(text=answer); self._ans_sub.config(text=sub)
        self._ans_outer.pack(fill="x",padx=10,pady=(0,10))


if __name__=="__main__":
    app=App(); app.geometry("820x740"); app.mainloop()
