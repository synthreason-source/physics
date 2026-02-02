import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# 2D WAVE + MULTIPLE CYLINDERS (5 spinning independently!)
# =============================================================================

nx, ny = 101, 101
Lx, Ly = 1.0, 1.0
dx, dy = Lx/(nx-1), Ly/(ny-1)
c = 1.0
dt = 0.25 * dx / c
steps_per_frame = 5

u_prev = np.zeros((nx, ny))
u = np.zeros((nx, ny))
u_next = np.zeros((nx, ny))

# DOUBLE PULSE (creates complex interactions)
cx1, cy1 = nx//4, ny//3      # top-left
cx2, cy2 = 3*nx//4, 2*ny//3  # bottom-right
R_pulse = 14
for i in range(nx):
    for j in range(ny):
        r1 = np.sqrt((i-cx1)**2 + (j-cy1)**2)
        r2 = np.sqrt((i-cx2)**2 + (j-cy2)**2)
        if r1 < R_pulse:
            u[i,j] += 1.0 * np.exp(-(r1**2)/50)
        if r2 < R_pulse:
            u[i,j] += 0.8 * np.exp(-(r2**2)/50)
u_prev[:] = u

# === 5 CYLINDERS ===
class Cylinder:
    def __init__(self, center, color, label):
        self.center = center
        self.radius = 0.12 * Lx
        self.theta = 0.0
        self.omega = 0.0
        self.color = color
        self.label = label
        self.history = []

cylinders = [
    Cylinder((0.25*Lx, 0.75*Ly), 'lime', 'A'),
    Cylinder((0.75*Lx, 0.75*Ly), 'cyan', 'B'),
    Cylinder((0.25*Lx, 0.25*Ly), 'magenta', 'C'),
    Cylinder((0.75*Lx, 0.25*Ly), 'yellow', 'D'),
    Cylinder((0.50*Lx, 0.50*Ly), 'orange', 'E')  # center
]

params = dict(Icyl=0.2, damp_rot=0.001, coupling_strength=100.0)

def step_wave(u_prev, u, u_next):
    uxx = (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / dx**2
    uyy = (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) / dy**2
    u_next[1:-1,1:-1] = 2*u[1:-1,1:-1] - u_prev[1:-1,1:-1] + c**2*dt**2*(uxx+uyy)
    u_next[[0,-1],:] = 0; u_next[:,[0,-1]] = 0

def get_slope(cx_idx, cy_idx, u):
    i0, j0 = int(cx_idx), int(cy_idx)
    if 2<=i0<=nx-3:
        return (u[i0+1,j0] - u[i0-1,j0]) / (2*dx)
    return (u[i0+1,j0] - u[i0,j0]) / dx

# === SPECTACULAR MULTI-CYLINDER ANIMATION ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Wave field
im = ax1.imshow(u.T, origin='lower', extent=[0,Lx,0,Ly],
                cmap='RdYlBu_r', vmin=-1.0, vmax=1.5, animated=True)

# 10 lines total (2 per cylinder)
lines1 = []; lines2 = []; cyl_circles = []; labels = []
colors = ['lime', 'cyan', 'magenta', 'yellow', 'orange']
for cyl, col in zip(cylinders, colors):
    line1, = ax1.plot([], [], col, lw=10)
    line2, = ax1.plot([], [], col, lw=10)
    circle = plt.Circle(cyl.center, cyl.radius, fc='none', ec=col, lw=4)
    ax1.add_patch(circle)
    text = ax1.text(cyl.center[0], cyl.center[1], cyl.label, fontsize=14,
                   ha='center', va='center', color=col, weight='bold',
                   bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
    lines1.append(line1); lines2.append(line2)
    cyl_circles.append(circle); labels.append(text)

ax1.set_xlim(0,Lx); ax1.set_ylim(0,Ly)
ax1.set_aspect('equal')
ax1.set_title('🌊 5 Spinning Cylinders Driven by Wave Pulses')

info = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat'))

# Live stats
ax2.set_title('Live Rotations (°)')
ax2.set_xlabel('Time'); ax2.set_ylabel('Angle')
for i, cyl in enumerate(cylinders):
    line, = ax2.plot([], [], c=colors[i], label=cyl.label, lw=3)
    cyl.plot_line = line
for line in [c.plot_line for c in cylinders]:
    ax2.legend(loc='upper left')

def animate(frame):
    global u_prev, u, u_next
    
    t = frame * steps_per_frame * dt
    
    # Wave step
    for _ in range(steps_per_frame):
        step_wave(u_prev, u, u_next)
        u_prev, u, u_next = u, u_next, u_prev
    
    # Update ALL cylinders
    for cyl in cylinders:
        cx_idx = cyl.center[0]/Lx * (nx-1)
        cy_idx = cyl.center[1]/Ly * (ny-1)
        slope = get_slope(cx_idx, cy_idx, u)
        torque = params['coupling_strength'] * slope
        
        domega_dt = (torque - params['damp_rot'] * cyl.omega) / params['Icyl']
        cyl.omega += domega_dt * dt
        cyl.theta += cyl.omega * dt
        cyl.history.append(np.degrees(cyl.theta))
    
    # Wave field
    im.set_array(u.T)
    
    # Update cylinder visuals
    for k, cyl in enumerate(cylinders):
        x0, y0 = cyl.center
        r = cyl.radius
        
        # Line 1
        ang1 = cyl.theta % (2*np.pi)
        lines1[k].set_data([x0, x0+r*np.cos(ang1)], [y0, y0+r*np.sin(ang1)])
        
        # Line 2 (90° offset)
        ang2 = (cyl.theta + np.pi/2) % (2*np.pi)
        lines2[k].set_data([x0, x0+r*np.cos(ang2)], [y0, y0+r*np.sin(ang2)])
        
        # Update label position/angle
        labels[k].set_position((x0 + r*1.3*np.cos(ang1), y0 + r*1.3*np.sin(ang1)))
    
    # Live stats
    for cyl in cylinders:
        cyl.plot_line.set_data([t], [cyl.history[-1]])
    
    max_theta = max(np.abs([c.history[-1] for c in cylinders]))
    info.set_text(f't={t:.1f}s | Max rotation: {max_theta:.0f}° | '
                  f'Pulses: 2 | Cylinders: 5')
    
    return [im] + lines1 + lines2 + cyl_circles + labels + [info]

print('🌈 MULTI-CYLINDER SPECTACLE! Watch 5 colored spinners go wild!')
ani = animation.FuncAnimation(fig, animate, frames=140, interval=60, blit=True)
plt.tight_layout()
plt.show()

# Final stats
print('\n🏆 FINAL MULTI-CYLINDER STATS:')
for i, cyl in enumerate(cylinders):
    print(f'Cyl {cyl.label}: {cyl.history[-1]:+.0f}° ({len(cyl.history)} steps)')
