
import tkinter as tk
from tkinter import ttk
import numpy as np

class LatticeModel:
    def __init__(self, n=60):
        self.n = n
        self.reset()

    def reset(self):
        self.temp = np.full((self.n, self.n), 0.25, dtype=np.float32)
        self.mask = np.ones((self.n, self.n), dtype=np.float32)
        self.phase = np.zeros((self.n, self.n), dtype=np.float32)
        self.order = np.ones((self.n, self.n), dtype=np.float32)
        self.totems = []
        self.t = 0.0
        self.beam_x = 0.0
        self.beam_phase = 0.0
        self.beam_hist = []

    def add_totem(self, x, y, power=1.0, radius=4.0, kind='pulse'):
        self.totems.append({'x': x, 'y': y, 'power': power, 'radius': radius, 'kind': kind, 'age': 0.0})

    def clear_totems(self):
        self.totems = []

    def step(self, dt, diffusion=0.0, cooling=0.0, tc=0.7, hysteresis=0.06):
        n = self.n
        T = self.temp
        lap = (
            np.roll(T, 1, 0) + np.roll(T, -1, 0) + np.roll(T, 1, 1) + np.roll(T, -1, 1) - 4 * T
        )
        T = T + 0.0 * lap * dt - 0.0 * (T - 0.25) * dt
        yy, xx = np.mgrid[0:n, 0:n]
        self.totems = [] if len(self.totems) > n * n else self.totems
        for totem in self.totems:
            dx = xx - totem['x']
            dy = yy - totem['y']
            r2 = dx * dx + dy * dy
            sigma2 = max(totem['radius'], 1.0) ** 2
            pulse = 1.0
            if totem['kind'] == 'pulse':
                pulse = 1.0 if (int(totem['age'] * 8) % 2 == 0) else 0.2
            elif totem['kind'] == 'ramp':
                pulse = min(1.5, 0.15 + totem['age'] * 0.5)
            elif totem['kind'] == 'sweep':
                pulse = 1.0
            elif totem['kind'] == 'ring':
                pulse = 1.0
            elif totem['kind'] == 'checker':
                pulse = 1.0
            elif totem['kind'] == 'anneal':
                pulse = max(0.3, 1.5 - totem['age'] * 0.2)
            T += totem['power'] * pulse * np.exp(-r2 / (2.0 * sigma2)) * dt * 0.8 * self.mask
            totem['age'] += dt
        self.temp = np.clip(T, 0.0, 2.0)
        self.temp *= self.mask
        above = self.temp > (tc + hysteresis * 0.5)
        below = self.temp < (tc - hysteresis * 0.5)
        self.phase[above] = 1.0
        self.phase[below] = -1.0
        self.order = np.clip(1.0 - np.abs(self.temp - tc) * 1.5, 0.0, 1.0)
        self.beam_x = (self.beam_x + dt * 18.0) % n
        bx = int(self.beam_x)
        column = self.phase[:, bx]
        tempcol = self.temp[:, bx]
        self.beam_phase += float(np.sum((tempcol - tc) * (0.5 + 0.5 * column)) * dt * 0.12)
        self.beam_hist.append(self.beam_phase)
        if len(self.beam_hist) > 240:
            self.beam_hist.pop(0)
        return self.temp, self.phase

class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Thermal Lattice Beam Simulator')
        self.model = LatticeModel(n=60)
        self.max_totems = self.model.n * self.model.n
        self.running = False
        self.sliders = {}
        self.cell = 10
        self.canvas_size = self.model.n * self.cell

        left = ttk.Frame(root)
        left.pack(side='left', fill='both', expand=True)
        right = ttk.Frame(root)
        right.pack(side='right', fill='y')

        self.canvas = tk.Canvas(left, width=self.canvas_size, height=self.canvas_size, bg='black', highlightthickness=0)
        self.canvas.pack(padx=6, pady=6)
        self.canvas.bind('<Button-1>', self.on_click)

        self.info = tk.StringVar()
        ttk.Label(right, text='Controls', font=('Arial', 12, 'bold')).pack(pady=(8, 4))
        self.make_slider(right, 'Temp C', 0.2, 1.4, 0.7)
        self.make_slider(right, 'Diffusion', 0.0, 0.2, 0.08)
        self.make_slider(right, 'Cooling', 0.0, 0.03, 0.008)
        self.make_slider(right, 'Beam Speed', 2.0, 40.0, 18.0)
        self.make_slider(right, 'Heat Power', 0.1, 3.0, 1.0)
        self.make_slider(right, 'Radius', 1.0, 10.0, 4.0)

        btns = ttk.Frame(right)
        btns.pack(pady=8, fill='x')
        for txt, cmd in [('Run', self.start), ('Pause', self.stop), ('Reset', self.reset), ('Clear Totems', self.clear_totems)]:
            ttk.Button(btns, text=txt, command=cmd).pack(fill='x', pady=2)

        ttk.Label(right, text='Presets', font=('Arial', 11, 'bold')).pack(pady=(10, 4))
        preset_frame = ttk.Frame(right)
        preset_frame.pack(fill='x')
        presets = ['pulse', 'ramp', 'sweep', 'ring', 'checker', 'anneal']
        for p in presets:
            ttk.Button(preset_frame, text=p.title(), command=lambda k=p: self.apply_preset(k)).pack(fill='x', pady=2)

        ttk.Label(right, textvariable=self.info, wraplength=240, justify='left').pack(pady=10, padx=6)
        self.running = True
        self.tick()

    def make_slider(self, parent, label, lo, hi, val):
        frm = ttk.Frame(parent)
        frm.pack(fill='x', padx=6, pady=4)
        ttk.Label(frm, text=label).pack(anchor='w')
        s = tk.Scale(frm, from_=lo, to=hi, resolution=(hi-lo)/300, orient='horizontal', length=220)
        s.set(val)
        s.pack(fill='x')
        self.sliders[label] = s

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def reset(self):
        self.model.reset()

    def clear_totems(self):
        self.model.clear_totems()

    def apply_preset(self, kind):
        self.model.clear_totems()
        n = self.model.n
        if kind == 'pulse':
            for y in range(n):
                for x in range(n):
                    self.model.add_totem(x, y, kind='pulse', power=1.4 if (x == n//2 and y == n//2) else 0.0, radius=5)
        elif kind == 'ramp':
            for y in range(n):
                for x in range(n):
                    power = 0.9 if x % 10 == 0 and y == n//2 else 0.0
                    self.model.add_totem(x, y, kind='ramp', power=power, radius=4)
        elif kind == 'sweep':
            for y in range(n):
                for x in range(n):
                    power = 1.1 if x == n//3 and y % 10 == 0 else 0.0
                    self.model.add_totem(x, y, kind='sweep', power=power, radius=4)
        elif kind == 'ring':
            cx = cy = n//2
            pts = {(int(cx + 14*np.cos(ang)), int(cy + 14*np.sin(ang))) for ang in np.linspace(0, 2*np.pi, 8, endpoint=False)}
            for y in range(n):
                for x in range(n):
                    self.model.add_totem(x, y, kind='ring', power=0.9 if (x, y) in pts else 0.0, radius=3)
        elif kind == 'checker':
            for y in range(n):
                for x in range(n):
                    power = 0.7 if ((x // 14 + y // 14) % 2 == 0) else 0.0
                    self.model.add_totem(x, y, kind='checker', power=power, radius=3)
        elif kind == 'anneal':
            for y in range(n):
                for x in range(n):
                    self.model.add_totem(x, y, kind='anneal', power=2.0 if (x == n//2 and y == n//2) else 0.0, radius=8)

    def on_click(self, event):
        x = event.x // self.cell
        y = event.y // self.cell
        if 0 <= x < self.model.n and 0 <= y < self.model.n:
            power = float(self.sliders['Heat Power'].get()) if 'Heat Power' in self.sliders else 1.0
            radius = float(self.sliders['Radius'].get()) if 'Radius' in self.sliders else 4.0
            if len(self.model.totems) < self.max_totems:
                self.model.add_totem(x, y, power=power, radius=radius, kind='pulse')

    def draw(self):
        self.canvas.delete('all')
        n = self.model.n
        T = self.model.temp
        P = self.model.phase
        for y in range(n):
            for x in range(n):
                temp = T[y, x]
                phase = P[y, x]
                if phase > 0:
                    r = int(40 + 180 * min(1.0, temp))
                    g = int(80 + 90 * (1.0 - abs(temp - 0.7)))
                    b = int(180 - 90 * min(1.0, temp))
                else:
                    r = int(30 + 80 * temp)
                    g = int(40 + 120 * (1.0 - temp/2.0))
                    b = int(80 + 140 * (1.0 - temp/2.0))
                color = f'#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}'
                x0, y0 = x * self.cell, y * self.cell
                self.canvas.create_rectangle(x0, y0, x0 + self.cell, y0 + self.cell, outline='', fill=color)
        bx = int(self.model.beam_x)
        for y in range(n):
            x0, y0 = bx * self.cell, y * self.cell
            self.canvas.create_rectangle(x0, y0, x0 + self.cell, y0 + self.cell, outline='yellow', width=1)
        for totem in self.model.totems:
            x0, y0 = totem['x'] * self.cell, totem['y'] * self.cell
            r = int(max(2, totem['radius']) * self.cell)
            self.canvas.create_oval(x0-r//2, y0-r//2, x0+r//2, y0+r//2, outline='white', width=2)
        self.info.set(f"Totems: {len(self.model.totems)} Beam phase: {self.model.beam_phase:.3f} Hot cells: {int((self.model.temp > float(self.sliders['Temp C'].get())).sum())}")

    def tick(self):
        if self.running:
            self.model.step(
                0.1,
                diffusion=float(self.sliders['Diffusion'].get()),
                cooling=float(self.sliders['Cooling'].get()),
                tc=float(self.sliders['Temp C'].get())
            )
            self.model.beam_x = (self.model.beam_x + float(self.sliders['Beam Speed'].get()) * 0.01) % self.model.n
        self.draw()
        self.root.after(33, self.tick)

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
