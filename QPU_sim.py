
import tkinter as tk
from tkinter import ttk
import numpy as np

class LatticeModel:
    def __init__(self, n=60):
        self.n = n
        self.reset()

    def reset(self):
        self.temp = np.full((self.n, self.n), 20.0, dtype=np.float32)
        self.phase = np.zeros((self.n, self.n), dtype=np.float32)
        self.director = np.zeros((self.n, self.n), dtype=np.float32)
        self.beam_phase = 0.0
        self.beam_hist = []
        self.beam_x = 0.0
        self.totems = []

    def add_totem(self, x, y, power=1.0, radius=0.0, kind='spot'):
        self.totems.append({'x': int(x), 'y': int(y), 'power': float(power), 'radius': float(radius), 'kind': kind, 'age': 0.0})

    def clear_totems(self):
        self.totems = []

    def p_state(self, T):
        if T < 21.0:
            return 0
        if T <= 45.0:
            return 1
        return 2

    def delta_n(self, T):
        if T < 21.0:
            return 0.02
        if T <= 45.0:
            x = (T - 21.0) / 24.0
            return 0.15 + 0.25 * (1.0 - abs(2.0 * x - 1.0))
        return 0.01

    def grad_scale(self, kind, x, y, n):
        cx = cy = (n - 1) / 2.0
        dx = x - cx
        dy = y - cy
        r = (dx * dx + dy * dy) ** 0.5
        if kind == 'stripe':
            return 1.0 + 0.2 * (x / max(1, n - 1))
        if kind == 'checker':
            return 0.7 + 0.3 * ((x + y) % 2)
        if kind == 'spiral':
            th = np.arctan2(dy, dx)
            return 0.6 + 0.4 * ((th + np.pi) / (2 * np.pi)) + 0.15 * min(1.0, r / max(1.0, n / 2.0))
        return 1.0

    def step(self, dt, tc=None):
        n = self.n
        self.temp[:] = 20.0
        for t in self.totems:
            if 0 <= t['x'] < n and 0 <= t['y'] < n:
                self.temp[t['y'], t['x']] = 20.0 + 35.0 * t['power']
        for y in range(n):
            for x in range(n):
                self.phase[y, x] = self.p_state(float(self.temp[y, x]))
                if self.phase[y, x] == 1:
                    self.director[y, x] = 1.0
                elif self.phase[y, x] == 0:
                    self.director[y, x] = 0.2
                else:
                    self.director[y, x] = 0.0
        self.beam_x = (self.beam_x + dt * 18.0) % n
        bx = int(self.beam_x)
        self.beam_phase = 0.0
        for y in range(n):
            T = float(self.temp[y, bx])
            dn = self.delta_n(T)
            self.beam_phase += dn
        self.beam_hist.append(self.beam_phase)
        if len(self.beam_hist) > 240:
            self.beam_hist.pop(0)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title('MBBA Beam Phase Lattice')
        self.model = LatticeModel(n=60)
        self.sliders = {}
        self.running = False
        self.cell = 10
        self.canvas_size = self.model.n * self.cell
        self.max_totems = self.model.n * self.model.n

        left = ttk.Frame(root)
        left.pack(side='left', fill='both', expand=True)
        right = ttk.Frame(root)
        right.pack(side='right', fill='y')

        self.canvas = tk.Canvas(left, width=self.canvas_size, height=self.canvas_size, bg='black', highlightthickness=0)
        self.canvas.pack(padx=6, pady=6)
        self.canvas.bind('<Button-1>', self.on_click)

        ttk.Label(right, text='MBBA Controls', font=('Arial', 12, 'bold')).pack(pady=(8, 4))
        self.make_slider(right, 'Power', 0.0, 1.0, 1.0)
        self.make_slider(right, 'Radius', 0.0, 1.0, 0.0)
        self.make_slider(right, 'Beam Speed', 0.0, 40.0, 18.0)

        btns = ttk.Frame(right)
        btns.pack(pady=8, fill='x')
        for txt, cmd in [('Run', self.start), ('Pause', self.stop), ('Reset', self.reset), ('Clear', self.clear_totems)]:
            ttk.Button(btns, text=txt, command=cmd).pack(fill='x', pady=2)

        ttk.Label(right, text='Presets', font=('Arial', 11, 'bold')).pack(pady=(10, 4))
        preset_frame = ttk.Frame(right)
        preset_frame.pack(fill='x')
        for txt, cmd in [('One per cell', self.preset_cells), ('Hot stripe', self.preset_stripe), ('Checker', self.preset_checker), ('Spiral', self.preset_spiral)]:
            ttk.Button(preset_frame, text=txt, command=cmd).pack(fill='x', pady=2)

        self.info = tk.StringVar(value='Click a cell to heat it. MBBA: crystal <21 C, nematic 21-45 C, isotropic >45 C.')
        ttk.Label(right, textvariable=self.info, wraplength=240, justify='left').pack(pady=10, padx=6)

        self.preset_cells()
        self.running = True
        self.tick()

    def make_slider(self, parent, label, lo, hi, val):
        frm = ttk.Frame(parent)
        frm.pack(fill='x', padx=6, pady=4)
        ttk.Label(frm, text=label).pack(anchor='w')
        s = tk.Scale(frm, from_=lo, to=hi, resolution=(hi-lo)/200 if hi != lo else 0.01, orient='horizontal', length=220)
        s.set(val)
        s.pack(fill='x')
        self.sliders[label] = s

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def reset(self):
        self.model.reset()
        self.preset_cells()

    def clear_totems(self):
        self.model.clear_totems()

    def preset_cells(self):
        self.model.clear_totems()
        n = self.model.n
        for y in range(n):
            for x in range(n):
                self.model.add_totem(x, y, power=0.0, radius=0.0, kind='cell')
        for y in range(n):
            for x in range(n):
                if x == n // 2:
                    self.model.totems[y * n + x]['power'] = 1.0
        self.info.set('One totem per cell. Heat is local only; no spread between cells.')

    def preset_stripe(self):
        self.model.clear_totems()
        n = self.model.n
        for y in range(n):
            for x in range(n):
                p = 1.0 if x in (n//3, n//3 + 1, n//3 + 2) else 0.0
                self.model.add_totem(x, y, power=p, radius=0.0, kind='stripe')
        self.info.set('Hot stripe preset loaded.')

    def preset_checker(self):
        self.model.clear_totems()
        n = self.model.n
        for y in range(n):
            for x in range(n):
                p = 1.0 if ((x // 4 + y // 4) % 2 == 0) else 0.0
                self.model.add_totem(x, y, power=p, radius=0.0, kind='checker')
        self.info.set('Checker preset loaded.')

    def preset_spiral(self):
        self.model.clear_totems()
        n = self.model.n
        cx = cy = n // 2
        turns = 4.5
        pts = set()
        for i in range(320):
            th = i * 0.18
            r = 0.35 * i
            x = int(round(cx + r * np.cos(th)))
            y = int(round(cy + r * np.sin(th)))
            if 0 <= x < n and 0 <= y < n:
                pts.add((x, y))
        for y in range(n):
            for x in range(n):
                self.model.add_totem(x, y, power=1.0 if (x, y) in pts else 0.0, radius=0.0, kind='spiral')
        self.spiral_pts = pts
        self.info.set('Spiral preset loaded.')

    def on_click(self, event):
        x = event.x // self.cell
        y = event.y // self.cell
        if 0 <= x < self.model.n and 0 <= y < self.model.n:
            idx = y * self.model.n + x
            if idx < len(self.model.totems):
                self.model.totems[idx]['power'] = 1.0 if self.model.totems[idx]['power'] <= 0.0 else 0.0

    def draw(self):
        self.canvas.delete('all')
        n = self.model.n
        for y in range(n):
            for x in range(n):
                T = float(self.model.temp[y, x])
                st = int(self.model.phase[y, x])
                pwr = float(self.model.totems[y * n + x]['power']) if y * n + x < len(self.model.totems) else 0.0
                if st == 0:
                    c = '#203050'
                elif st == 1:
                    c = '#2d8f8a'
                else:
                    c = '#d9d9d9'
                if T > 20.0 and st == 1:
                    c = '#39c6b8'
                if pwr > 0.0:
                    c = '#ff8844' if st == 2 else c
                x0, y0 = x * self.cell, y * self.cell
                self.canvas.create_rectangle(x0, y0, x0 + self.cell, y0 + self.cell, outline='', fill=c)
        bx = int(self.model.beam_x)
        for y in range(n):
            x0, y0 = bx * self.cell, y * self.cell
            self.canvas.create_rectangle(x0, y0, x0 + self.cell, y0 + self.cell, outline='yellow', width=1)
        self.info.set(f"Totems: {len(self.model.totems)} | Beam phase: {self.model.beam_phase:.3f} | Beam x: {bx}")

    def tick(self):
        if self.running:
            self.model.step(0.1)
            self.model.beam_x = (self.model.beam_x + float(self.sliders['Beam Speed'].get()) * 0.01) % self.model.n
        self.draw()
        self.root.after(33, self.tick)

if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
