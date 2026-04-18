import tkinter as tk
from tkinter import ttk
import math
import random
import time

# ── colour palette ──────────────────────────────────────────────
BG       = "#f8f8f6"
PANEL_BG = "#ffffff"
BORDER   = "#d0cfc8"
TXT_PRI  = "#1a1a18"
TXT_SEC  = "#6b6b65"
TEAL_LO  = "#1D9E75"   # low energy / solved
AMBER    = "#BA7517"   # mid energy
RED_HI   = "#D85A30"   # high energy
BLUE     = "#378ADD"   # chart line
INFO_BG  = "#E6F1FB"
INFO_TXT = "#185FA5"
WARN_BG  = "#FAEEDA"
WARN_TXT = "#854F0B"
OK_BG    = "#E1F5EE"
OK_TXT   = "#0F6E56"

PROBLEMS = {
    "Integer factorisation  N = 391": {
        "totems": 32,
        "display": "Factorise  N = 391\nFind p, q  such that  p × q = 391",
        "answer":  "17 × 23 = 391",
        "sub":     "Phase encoding: p → beam angle θ₁, q → θ₂\nProduct constraint collapsed manifold to solution.",
        "energy_fn": lambda c, mx: max(0, (1 - (c/mx)**2 * 0.9 - random.random()*0.05) * 100),
    },
    "Quadratic roots  2x²−7x+3=0": {
        "totems": 16,
        "display": "Solve  2x² − 7x + 3 = 0\nFind real roots",
        "answer":  "x = 3.0   and   x = 0.5",
        "sub":     "Discriminant Δ = 25 > 0.\nPhase amplitude encoded coefficients; zero-crossing manifold located roots.",
        "energy_fn": lambda c, mx: max(0, (1 - c/mx*1.1)*80 + math.sin(c*0.4)*3 + random.random()*2),
    },
    "Shortest path  A→B→C→A": {
        "totems": 24,
        "display": "Shortest path through:\n  A(0,0)  B(3,4)  C(6,1)  → back to A",
        "answer":  "A → B → C → A  ≈ 16.1 units",
        "sub":     "AB=5, BC≈5.83, CA≈6.08.\nPhotonic constraint annealing evaluated all route orderings.",
        "energy_fn": lambda c, mx: max(0, 90*(1-c/mx) + abs(math.sin(c*0.7))*8 + random.random()*4),
    },
    "Matrix eigenvalue  M=[[4,1],[2,3]]": {
        "totems": 20,
        "display": "Dominant eigenvalue of\n  M = [[4, 1], [2, 3]]",
        "answer":  "λ₁ = 5,   eigenvector ≈ [1, 1]",
        "sub":     "Char. poly: λ²−7λ+10=0 → λ=5, 2.\nBeam power iteration locked onto dominant mode.",
        "energy_fn": lambda c, mx: max(0, 95*math.exp(-3.5*c/mx) + random.random()*3),
    },
}

MAX_CYCLES = 120
TICK_MS    = 55   # ms per simulation step


class PhotonicSolverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Photonic Math Solver — MBBA Totem Array")
        self.configure(bg=BG)
        self.resizable(True, True)

        self._sim_job   = None
        self._running   = False
        self._cycles    = 0
        self._energy_hist = []
        self._totem_cvs   = []

        self._build_ui()
        self._load_problem()

    # ── UI construction ──────────────────────────────────────────
    def _build_ui(self):
        pad = dict(padx=12, pady=6)

        # ── top bar ──────────────────────────────────────────────
        top = tk.Frame(self, bg=PANEL_BG, bd=0, highlightthickness=1,
                       highlightbackground=BORDER)
        top.pack(fill="x", padx=10, pady=(10, 4))

        tk.Label(top, text="Photonic math solver", font=("Helvetica", 13, "bold"),
                 bg=PANEL_BG, fg=TXT_PRI).pack(side="left", **pad)
        tk.Label(top, text="MBBA totem array anneals to the solution",
                 font=("Helvetica", 10), bg=PANEL_BG, fg=TXT_SEC).pack(side="left")

        btn_frame = tk.Frame(top, bg=PANEL_BG)
        btn_frame.pack(side="right", **pad)
        self._btn_solve = tk.Button(btn_frame, text="Solve", command=self._start_solve,
                                    font=("Helvetica", 11), bg=INFO_BG, fg=INFO_TXT,
                                    relief="flat", padx=12, pady=4, cursor="hand2")
        self._btn_solve.pack(side="left", padx=4)
        tk.Button(btn_frame, text="Reset", command=self._reset,
                  font=("Helvetica", 11), bg=BG, fg=TXT_SEC,
                  relief="flat", padx=12, pady=4, cursor="hand2").pack(side="left", padx=4)

        # ── problem selector ─────────────────────────────────────
        sel_frame = tk.Frame(self, bg=BG)
        sel_frame.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(sel_frame, text="Problem:", font=("Helvetica", 10),
                 bg=BG, fg=TXT_SEC).pack(side="left")
        self._prob_var = tk.StringVar()
        prob_menu = ttk.Combobox(sel_frame, textvariable=self._prob_var,
                                 values=list(PROBLEMS.keys()), state="readonly",
                                 width=42, font=("Helvetica", 10))
        prob_menu.pack(side="left", padx=6)
        prob_menu.current(0)
        prob_menu.bind("<<ComboboxSelected>>", lambda _: self._load_problem())

        # ── problem text ─────────────────────────────────────────
        prob_outer = tk.Frame(self, bg=PANEL_BG, bd=0, highlightthickness=1,
                              highlightbackground=BORDER)
        prob_outer.pack(fill="x", padx=10, pady=(0, 4))
        self._prob_lbl = tk.Label(prob_outer, text="", font=("Courier", 11),
                                  bg=PANEL_BG, fg=TXT_PRI, justify="left",
                                  anchor="w")
        self._prob_lbl.pack(fill="x", padx=12, pady=8)

        # ── metric row ───────────────────────────────────────────
        met = tk.Frame(self, bg=BG)
        met.pack(fill="x", padx=10, pady=(0, 4))
        self._stat_cycles = self._metric(met, "Beam cycles",    "0")
        self._stat_totems = self._metric(met, "Active totems",  "0")
        self._stat_energy = self._metric(met, "Energy",         "—")
        self._stat_status = self._metric(met, "Status",         "idle", WARN_BG, WARN_TXT)
        for w in (self._stat_cycles, self._stat_totems, self._stat_energy, self._stat_status):
            w.pack(side="left", expand=True, fill="x", padx=4)

        # ── middle row: totems + chart ───────────────────────────
        mid = tk.Frame(self, bg=BG)
        mid.pack(fill="both", expand=True, padx=10, pady=(0, 4))

        # totem panel
        totem_outer = tk.Frame(mid, bg=PANEL_BG, bd=0, highlightthickness=1,
                               highlightbackground=BORDER)
        totem_outer.pack(side="left", fill="both", expand=True, padx=(0, 4))
        tk.Label(totem_outer, text="Beam phase — MBBA totems",
                 font=("Helvetica", 9), bg=PANEL_BG, fg=TXT_SEC,
                 anchor="w").pack(fill="x", padx=10, pady=(6, 2))
        self._totem_frame = tk.Frame(totem_outer, bg=PANEL_BG)
        self._totem_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        # chart panel
        chart_outer = tk.Frame(mid, bg=PANEL_BG, bd=0, highlightthickness=1,
                               highlightbackground=BORDER)
        chart_outer.pack(side="left", fill="both", expand=True)
        tk.Label(chart_outer, text="Convergence — energy over cycles",
                 font=("Helvetica", 9), bg=PANEL_BG, fg=TXT_SEC,
                 anchor="w").pack(fill="x", padx=10, pady=(6, 2))
        self._chart_cv = tk.Canvas(chart_outer, bg=PANEL_BG, bd=0,
                                   highlightthickness=0, height=180)
        self._chart_cv.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        # ── log ──────────────────────────────────────────────────
        log_outer = tk.Frame(self, bg=PANEL_BG, bd=0, highlightthickness=1,
                             highlightbackground=BORDER)
        log_outer.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(log_outer, text="Beam annealing log",
                 font=("Helvetica", 9), bg=PANEL_BG, fg=TXT_SEC,
                 anchor="w").pack(fill="x", padx=10, pady=(4, 0))
        self._log_text = tk.Text(log_outer, height=4, font=("Courier", 9),
                                 bg=PANEL_BG, fg=TXT_SEC, relief="flat",
                                 state="disabled", wrap="word")
        self._log_text.pack(fill="x", padx=10, pady=(0, 6))

        # ── answer box ───────────────────────────────────────────
        self._ans_outer = tk.Frame(self, bg=OK_BG, bd=0, highlightthickness=1,
                                   highlightbackground=TEAL_LO)
        tk.Label(self._ans_outer, text="Solution found",
                 font=("Helvetica", 9), bg=OK_BG, fg=OK_TXT,
                 anchor="w").pack(fill="x", padx=12, pady=(6, 0))
        self._ans_lbl = tk.Label(self._ans_outer, text="",
                                 font=("Helvetica", 14, "bold"),
                                 bg=OK_BG, fg=TXT_PRI, anchor="w", justify="left")
        self._ans_lbl.pack(fill="x", padx=12)
        self._ans_sub = tk.Label(self._ans_outer, text="",
                                 font=("Helvetica", 9),
                                 bg=OK_BG, fg=TXT_SEC, anchor="w", justify="left")
        self._ans_sub.pack(fill="x", padx=12, pady=(0, 8))

    def _metric(self, parent, label, val, bg=None, fg=None):
        bg  = bg  or "#f1efe8"
        fg  = fg  or TXT_PRI
        frm = tk.Frame(parent, bg=bg, bd=0, highlightthickness=1,
                       highlightbackground=BORDER)
        tk.Label(frm, text=label, font=("Helvetica", 9), bg=bg,
                 fg=TXT_SEC).pack(anchor="w", padx=10, pady=(6, 0))
        lbl = tk.Label(frm, text=val, font=("Helvetica", 16, "bold"),
                       bg=bg, fg=fg)
        lbl.pack(anchor="w", padx=10, pady=(0, 6))
        frm._val_lbl = lbl
        frm._bg      = bg
        return frm

    def _set_metric(self, frm, val, fg=None):
        frm._val_lbl.config(text=val, fg=fg or TXT_PRI)

    # ── problem loading ──────────────────────────────────────────
    def _load_problem(self):
        self._reset()

    def _reset(self):
        if self._sim_job:
            self.after_cancel(self._sim_job)
            self._sim_job = None
        self._running   = False
        self._cycles    = 0
        self._energy_hist = []

        key  = self._prob_var.get()
        prob = PROBLEMS[key]
        self._prob_lbl.config(text=prob["display"])
        self._set_metric(self._stat_cycles, "0")
        self._set_metric(self._stat_totems, str(prob["totems"]))
        self._set_metric(self._stat_energy, "—")
        self._set_metric(self._stat_status, "idle", WARN_TXT)
        self._stat_status.config(bg=WARN_BG)
        self._stat_status._val_lbl.config(bg=WARN_BG)

        self._build_totems(prob["totems"])
        self._draw_chart()
        self._clear_log()
        self._ans_outer.pack_forget()
        self._btn_solve.config(text="Solve", bg=INFO_BG, fg=INFO_TXT)

    def _build_totems(self, n):
        for w in self._totem_frame.winfo_children():
            w.destroy()
        self._totem_cvs = []
        cols = 8
        rows = math.ceil(n / cols)
        for r in range(rows):
            for c in range(cols):
                idx = r*cols + c
                if idx >= n:
                    break
                cv = tk.Canvas(self._totem_frame, width=28, height=22,
                               bg=PANEL_BG, bd=0, highlightthickness=0)
                cv.grid(row=r, column=c, padx=2, pady=2)
                rect = cv.create_rectangle(0, 0, 28, 22, fill="#E1F5EE",
                                           outline="", tags="cell")
                cv._rect = rect
                self._totem_cvs.append(cv)

    def _colour_for(self, phase, energy):
        if energy < 3:
            return TEAL_LO
        t = min(1, max(0, phase))
        r = int(29  + t*196)
        g = int(158 - t*60)
        b = int(117 - t*80)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _update_totems(self, energy):
        for i, cv in enumerate(self._totem_cvs):
            phase = math.sin(self._cycles*0.15 + i*0.4)*0.5 + 0.5
            col   = self._colour_for(phase, energy)
            cv.itemconfig(cv._rect, fill=col)

    # ── chart ────────────────────────────────────────────────────
    def _draw_chart(self):
        cv  = self._chart_cv
        cv.delete("all")
        w   = cv.winfo_width()  or 300
        h   = cv.winfo_height() or 180
        pad = 32
        iw  = w - pad*2
        ih  = h - pad*2

        # axes
        cv.create_line(pad, pad, pad, h-pad, fill=BORDER)
        cv.create_line(pad, h-pad, w-pad, h-pad, fill=BORDER)
        cv.create_text(8, pad,   text="100", font=("Helvetica",8), fill=TXT_SEC, anchor="w")
        cv.create_text(8, h-pad, text="0",   font=("Helvetica",8), fill=TXT_SEC, anchor="w")
        cv.create_text(w//2, h-8, text="cycles", font=("Helvetica",8), fill=TXT_SEC)

        hist = self._energy_hist
        if len(hist) < 2:
            return
        mx_x = max(len(hist)-1, 1)
        pts   = []
        for i, e in enumerate(hist):
            x = pad + (i/mx_x)*iw
            y = (h-pad) - (e/100)*ih
            pts.append((x, y))
        for i in range(len(pts)-1):
            cv.create_line(pts[i][0], pts[i][1],
                           pts[i+1][0], pts[i+1][1],
                           fill=BLUE, width=2)

    # ── log ──────────────────────────────────────────────────────
    def _log(self, msg):
        self._log_text.config(state="normal")
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    # ── simulation ───────────────────────────────────────────────
    def _start_solve(self):
        if self._running:
            return
        self._reset()
        prob = PROBLEMS[self._prob_var.get()]
        self._running = True
        self._set_metric(self._stat_status, "annealing", INFO_TXT)
        self._stat_status.config(bg=INFO_BG)
        self._stat_status._val_lbl.config(bg=INFO_BG)
        self._btn_solve.config(text="Running…", bg="#d0e8f8", fg=INFO_TXT)
        self._log(f"[0] Problem encoded. {prob['totems']} MBBA totems initialised.")
        self._log("[0] Beam lattice: spiral preset. Power: 1.000")
        self._tick(prob)

    def _tick(self, prob):
        if not self._running:
            return
        self._cycles += 1
        energy = prob["energy_fn"](self._cycles, MAX_CYCLES)
        self._energy_hist.append(round(energy, 2))

        self._set_metric(self._stat_cycles, str(self._cycles))
        self._set_metric(self._stat_energy, f"{energy:.1f}")
        self._update_totems(energy)
        self._draw_chart()

        if self._cycles % 15 == 0:
            temp = 1 - self._cycles/MAX_CYCLES
            self._log(f"[{self._cycles:3d}] T={temp:.2f}  E={energy:.2f}  — beam phase converging")

        done = energy < 2 or self._cycles >= MAX_CYCLES
        if done:
            self._finish(prob)
        else:
            self._sim_job = self.after(TICK_MS, lambda: self._tick(prob))

    def _finish(self, prob):
        self._running = False
        self._set_metric(self._stat_status, "solved", OK_TXT)
        self._stat_status.config(bg=OK_BG)
        self._stat_status._val_lbl.config(bg=OK_BG)
        self._btn_solve.config(text="Solve", bg=INFO_BG, fg=INFO_TXT)
        self._update_totems(0)
        self._log(f"[{self._cycles}] Convergence reached. Solution locked.")

        self._ans_lbl.config(text=prob["answer"])
        self._ans_sub.config(text=prob["sub"])
        self._ans_outer.pack(fill="x", padx=10, pady=(0, 10))


if __name__ == "__main__":
    app = PhotonicSolverApp()
    app.geometry("780x680")
    app.mainloop()