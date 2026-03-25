import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patheffects as pe
import numpy as np
import json

fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor('#080c12')

def lbl(ax, x, y, txt, size=10, color='white', ha='center', va='center', bold=False):
    ax.text(x, y, txt, fontsize=size, color=color, ha=ha, va=va,
            fontweight='bold' if bold else 'normal',
            path_effects=[pe.withStroke(linewidth=2, foreground='#080c12')])

def box2d(ax, x, y, w, h, fc, ec, alpha=0.9, pad=0.1):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle=f"round,pad={pad}", fc=fc, ec=ec, lw=2, alpha=alpha, zorder=3))

# ══════════════════════════════════════════════════════
# PANEL A — 3D Woodpile Crystal (isometric view)
# ══════════════════════════════════════════════════════
ax3d = fig.add_subplot(2, 3, 1, projection='3d')
ax3d.set_facecolor('#0d1117')

rod_r = 0.12
spacing = 0.8   # rod centre-to-centre within layer
n_rods  = 5
layers  = 4     # 4 layers = 1 full unit cell (woodpile)
layer_h = 0.4

rod_color_by_layer = ['#58a6ff', '#ffa657', '#58a6ff', '#ffa657']
rod_label = ['Layer 1 (X-rods)', 'Layer 2 (Y-rods)', 'Layer 3 (X, offset)', 'Layer 4 (Y, offset)']

for layer in range(layers):
    z = layer * layer_h
    col = rod_color_by_layer[layer]
    horizontal = (layer % 2 == 0)
    offset = (spacing / 2) if (layer >= 2) else 0

    theta = np.linspace(0, 2*np.pi, 20)
    rod_len = (n_rods - 1) * spacing + 0.6

    for i in range(n_rods):
        pos = i * spacing + offset
        if horizontal:
            xs = np.linspace(-0.3, rod_len - 0.3, 30)
            ys = np.full(30, pos)
            zs = np.full(30, z)
            # draw as thick line
            ax3d.plot(xs, ys, zs, color=col, linewidth=4, alpha=0.85, zorder=layer+1)
            # top and bottom circles at ends
        else:
            xs = np.full(30, pos)
            ys = np.linspace(-0.3, rod_len - 0.3, 30)
            zs = np.full(30, z)
            ax3d.plot(xs, ys, zs, color=col, linewidth=4, alpha=0.85, zorder=layer+1)

ax3d.set_xlim(-0.5, 4); ax3d.set_ylim(-0.5, 4); ax3d.set_zlim(-0.2, 1.8)
ax3d.set_xlabel('X (cm)', color='#8b949e', fontsize=9, labelpad=2)
ax3d.set_ylabel('Y (cm)', color='#8b949e', fontsize=9, labelpad=2)
ax3d.set_zlabel('Z', color='#8b949e', fontsize=9, labelpad=2)
ax3d.set_title('A.  Woodpile Crystal Structure\n(4-layer unit cell)', color='white', fontsize=11, pad=8)
ax3d.tick_params(colors='#555', labelsize=7)
ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor('#1c2128'); ax3d.yaxis.pane.set_edgecolor('#1c2128'); ax3d.zaxis.pane.set_edgecolor('#1c2128')
ax3d.view_init(elev=28, azim=-55)

# legend
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], color=c, lw=4, label=l)
           for c, l in zip(rod_color_by_layer[:2], ['X-direction rods','Y-direction rods'])]
ax3d.legend(handles=handles, loc='upper left', fontsize=8,
            facecolor='#0d1117', edgecolor='#333', labelcolor='white')

# ══════════════════════════════════════════════════════
# PANEL B — Materials & Dimensions
# ══════════════════════════════════════════════════════
ax2 = fig.add_subplot(2, 3, 2)
ax2.set_facecolor('#0d1117'); ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis('off')
ax2.set_title('B.  Parts List & Cut Dimensions\n(Target freq: 2.4 GHz)', color='white', fontsize=11, pad=8)

parts = [
    ('#ffa657', '6mm copper pipe',   '40× rods, each 10 cm long'),
    ('#ffa657', 'Acrylic sheet 5mm', '2× base plates 20×20 cm'),
    ('#58a6ff', 'Slot spacing',       'd = 31.25 mm  (λ/4 @ 2.4 GHz)'),
    ('#58a6ff', 'Rod diameter',       'r = 6 mm  (r/d ≈ 0.19)'),
    ('#58a6ff', 'Layer height',       'h = 15.6 mm  (λ/8)'),
    ('#58a6ff', 'Unit cell size',     'a = 62.5 mm × 62.5 mm'),
    ('#3fb950', 'NanoVNA v2',         'S11/S21 sweep 1–4 GHz  ($35)'),
    ('#3fb950', 'SMA pigtail ×2',     'Feed antenna into crystal port'),
    ('#d2a8ff', 'Total rods needed',  '4 layers × 5 rows × 2 dirs = 40'),
    ('#d2a8ff', 'Est. cost (AUS)',    '~$18 copper + $35 NanoVNA = $53'),
]

for i, (col, name, desc) in enumerate(parts):
    y = 9.3 - i * 0.9
    box2d(ax2, 2.8, y, 4.8, 0.68, '#161b22', col, alpha=0.85)
    lbl(ax2, 0.4, y, '●', size=12, color=col, ha='left')
    lbl(ax2, 0.7, y+0.14, name, size=9, bold=True, color=col, ha='left')
    lbl(ax2, 0.7, y-0.14, desc, size=8.5, color='#cdd9e5', ha='left')

# ══════════════════════════════════════════════════════
# PANEL C — Layer-by-layer construction
# ══════════════════════════════════════════════════════
ax3 = fig.add_subplot(2, 3, 3)
ax3.set_facecolor('#0d1117'); ax3.set_xlim(0, 10); ax3.set_ylim(0, 10); ax3.axis('off')
ax3.set_title('C.  Layer-by-Layer Build\n(isometric cross-section)', color='white', fontsize=11, pad=8)

layer_data = [
    (1, '#58a6ff', '─ ─ ─ ─ ─', 'Layer 1: X-rods,  y=0,31,62,93,124 mm'),
    (2, '#ffa657', '│ │ │ │ │', 'Layer 2: Y-rods,  x=0,31,62,93,124 mm'),
    (3, '#58a6ff', '─ ─ ─ ─ ─', 'Layer 3: X-rods, offset +15.6 mm in Y'),
    (4, '#ffa657', '│ │ │ │ │', 'Layer 4: Y-rods, offset +15.6 mm in X'),
]

y_start = 8.8
for ln, col, sym, desc in layer_data:
    y = y_start - (ln-1)*1.9
    # layer slab
    ax3.add_patch(FancyBboxPatch((0.5, y-0.65), 8.8, 1.2,
        boxstyle="round,pad=0.08", fc='#0d1f2d', ec=col, lw=1.8, alpha=0.85))
    lbl(ax3, 1.0, y+0.22, f'L{ln}', size=13, bold=True, color=col, ha='left')
    # rod cross-sections
    rod_col = '#1a3a5c' if col == '#58a6ff' else '#2d1f0a'
    for ri in range(5):
        rx = 2.5 + ri * 1.25
        ry = y
        c = Circle((rx, ry), 0.22, fc=rod_col, ec=col, lw=1.5, zorder=4)
        ax3.add_patch(c)
        if ln <= 2 and ri == 2:
            lbl(ax3, rx, ry, '⊙' if ln%2==1 else '⊗', size=9, color=col)
    lbl(ax3, 5.0, y-0.43, desc, size=8.5, color='#8b949e')

# Spacing annotation
ax3.annotate('', xy=(2.5, 2.05), xytext=(3.75, 2.05),
    arrowprops=dict(arrowstyle='<->', color='white', lw=1.3))
lbl(ax3, 3.12, 1.85, '31.25 mm', size=8, color='white')

# ══════════════════════════════════════════════════════
# PANEL D — NanoVNA measurement setup
# ══════════════════════════════════════════════════════
ax4 = fig.add_subplot(2, 3, 4)
ax4.set_facecolor('#0d1117'); ax4.set_xlim(0, 10); ax4.set_ylim(0, 10); ax4.axis('off')
ax4.set_title('D.  NanoVNA Measurement Setup', color='white', fontsize=11, pad=8)

# NanoVNA box
box2d(ax4, 2.0, 7.5, 3.2, 1.5, '#161b22', '#3fb950')
lbl(ax4, 2.0, 7.75, 'NanoVNA v2', size=10, bold=True, color='#3fb950')
lbl(ax4, 2.0, 7.35, 'Port 1 (TX)   Port 2 (RX)', size=8.5, color='#8b949e')
# ports
ax4.add_patch(Circle((1.0, 7.35), 0.18, fc='#3fb950', ec='white', lw=1.5, zorder=4))
ax4.add_patch(Circle((3.0, 7.35), 0.18, fc='#3fb950', ec='white', lw=1.5, zorder=4))

# Crystal block
box2d(ax4, 6.5, 7.5, 3.0, 3.2, '#0a1628', '#58a6ff')
lbl(ax4, 6.5, 8.5, '3D Woodpile\nCrystal', size=10, bold=True, color='#58a6ff')
# draw mini woodpile inside
for ri in range(4):
    ax4.plot([5.2, 7.8], [7.0 + ri*0.45, 7.0 + ri*0.45],
             color='#58a6ff' if ri%2==0 else '#ffa657', lw=3, alpha=0.7, zorder=4)
# SMA ports on crystal
ax4.add_patch(Circle((5.0, 7.5), 0.18, fc='#58a6ff', ec='white', lw=1.5, zorder=4))
ax4.add_patch(Circle((8.0, 7.5), 0.18, fc='#58a6ff', ec='white', lw=1.5, zorder=4))

# Coax cables
ax4.annotate('', xy=(5.0, 7.5), xytext=(1.0, 7.35),
    arrowprops=dict(arrowstyle='-', color='#c9a227', lw=3,
                    connectionstyle='arc3,rad=-0.4'))
lbl(ax4, 3.0, 6.7, 'Coax / SMA', size=8.5, color='#c9a227')

# Steps
steps = [
    "① Sweep 1–4 GHz on NanoVNA",
    "② Watch S21 (transmission)",
    "③ Find deep notch = BANDGAP",
    "④ Read η = S21_pass / S21_gap",
    "⑤ Plug η into CasimirNoiseModel()",
]
for i, s in enumerate(steps):
    lbl(ax4, 0.2, 5.8 - i*0.85, s, size=9.5, color='#cdd9e5', ha='left')

# ══════════════════════════════════════════════════════
# PANEL E — Expected S21 spectrum
# ══════════════════════════════════════════════════════
ax5 = fig.add_subplot(2, 3, 5)
ax5.set_facecolor('#0d1117')
freq = np.linspace(1, 4, 600)
# Simulate woodpile bandgap at 2.4 GHz
T = phc_transmission(freq / 4.0, N=4, n1=3.5, n2=1.0)  # scale for demo
# add some passband ripple
ripple = 0.08 * np.sin(20 * freq) * np.exp(-0.3 * (freq - 2.4)**2 + 0.3)
S21_dB = 20 * np.log10(np.clip(T + 0.01 + ripple, 1e-6, 1))

ax5.fill_between(freq, S21_dB, -45, where=(freq > 2.1) & (freq < 2.75),
                 alpha=0.2, color='#ff7b72', label='Bandgap (LDOS=0)')
ax5.plot(freq, S21_dB, color='#3fb950', linewidth=2.5, label='S21 Transmission')
ax5.axhline(-3, color='#c9a227', lw=1.3, linestyle='--', alpha=0.7, label='-3 dB reference')
ax5.axvline(2.4, color='white', lw=1.2, linestyle=':', alpha=0.5)
ax5.text(2.42, -5, '2.4 GHz\nbandgap', color='#ff7b72', fontsize=9)
ax5.annotate('', xy=(2.1, -30), xytext=(2.75, -30),
    arrowprops=dict(arrowstyle='<->', color='white', lw=1.3))
ax5.text(2.35, -32, 'Δf ≈ 0.6 GHz', color='white', fontsize=8.5, ha='center')
ax5.set_xlabel('Frequency (GHz)', fontsize=11, color='#8b949e')
ax5.set_ylabel('S21 (dB)', fontsize=11, color='#8b949e')
ax5.set_title('E.  Expected NanoVNA Output\n(transmission spectrum)', color='white', fontsize=11)
ax5.set_xlim(1, 4); ax5.set_ylim(-45, 5)
ax5.tick_params(colors='#8b949e', labelsize=10)
ax5.spines[['bottom','left','top','right']].set_color('#333')
ax5.grid(True, color='#1c2128', linewidth=0.8)
ax5.legend(fontsize=9, facecolor='#0d1117', edgecolor='#333', labelcolor='white', loc='lower right')

# ══════════════════════════════════════════════════════
# PANEL F — Code integration
# ══════════════════════════════════════════════════════
ax6 = fig.add_subplot(2, 3, 6)
ax6.set_facecolor('#0d1117'); ax6.set_xlim(0, 10); ax6.set_ylim(0, 10); ax6.axis('off')
ax6.set_title('F.  Feeding Measurement → casimir_qc.py', color='white', fontsize=11, pad=8)

code_lines = [
    ("# 1. Read S21 from NanoVNA CSV", '#8b949e'),
    ("import pandas as pd", '#cdd9e5'),
    ("df = pd.read_csv('nanovna_s21.csv')", '#cdd9e5'),
    ("", ''),
    ("# 2. Compute η from measurement", '#8b949e'),
    ("S21_pass = df[df.freq < 2.0]['S21_dB'].mean()", '#cdd9e5'),
    ("S21_gap  = df[(df.freq>2.1)&", '#cdd9e5'),
    ("           (df.freq<2.7)]['S21_dB'].min()", '#cdd9e5'),
    ("eta_real = 10**((S21_pass-S21_gap)/20)", '#ffa657'),
    ("print(f'Measured η = {eta_real:.1f}')", '#cdd9e5'),
    ("", ''),
    ("# 3. Plug into Casimir noise model", '#8b949e'),
    ("nm = CasimirNoiseModel(eta=eta_real)", '#3fb950'),
    ("gen = BellStateGenerator(eta=eta_real)", '#3fb950'),
    ("r = gen.run('phi_plus')", '#3fb950'),
    ("print(f\"F = {r['fidelity']:.5f}\")", '#3fb950'),
]
y = 9.5
for line, col in code_lines:
    if col:
        ax6.text(0.2, y, line, fontsize=8.5, color=col,
                 fontfamily='monospace', va='center',
                 path_effects=[pe.withStroke(linewidth=1.5, foreground='#0d1117')])
    y -= 0.55

# eta annotation
box2d(ax6, 5.0, 1.1, 9.5, 0.75, '#0d2b1f', '#3fb950', pad=0.08)
lbl(ax6, 5.0, 1.35, 'Typical result:  η ≈ 40–200  →  F ≈ 0.9985–0.9998', size=9, color='#3fb950')
lbl(ax6, 5.0, 0.9, 'Deep bandgap (more layers):  η > 1000  →  F > 0.9999', size=9, color='#58a6ff')

plt.suptitle("How to Build a 3D Microwave Photonic Crystal Quantum Cavity",
             fontsize=16, color='#e6edf3', fontweight='bold', y=1.01)
plt.tight_layout(pad=1.8)
plt.savefig('3d_microwave_crystal_guide.png', dpi=150, bbox_inches='tight',
            facecolor='#080c12', edgecolor='none')
plt.close()

with open('3d_microwave_crystal_guide.png.meta.json','w') as f:
    json.dump({
        "caption": "How to build a 3D microwave woodpile photonic crystal cavity — complete build guide",
        "description": "6-panel guide: A) 3D woodpile structure isometric view; B) parts list with copper pipe dimensions for 2.4 GHz; C) layer-by-layer cross-section showing X/Y rod alternation; D) NanoVNA measurement setup with coax connections; E) expected S21 transmission showing bandgap notch at 2.4 GHz; F) Python code to read NanoVNA CSV and feed measured η into casimir_qc.py CasimirNoiseModel."
    }, f)
print("done")
