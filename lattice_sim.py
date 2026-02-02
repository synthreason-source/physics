import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# 2D WAVE + MULTIPLE CYLINDERS - FULLY FIXED (NO EMOJIS + GENERATOR ERROR)
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

# SINGLE CENTRAL PULSE
cx1, cy1 = nx//2, ny//2
R_pulse = 14
for i in range(nx):
    for j in range(ny):
        r1 = np.sqrt((i-cx1)**2 + (j-cy1)**2)
        if r1 < R_pulse:
            u[i,j] += 1.0 * np.exp(-(r1**2)/50)
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
    Cylinder((0.50*Lx, 0.50*Ly), 'orange', 'E')
]

params = dict(Icyl=0.2, damp_rot=0.001, coupling_strength=100.0)

def step_wave(u_prev, u, u_next):
    uxx = (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / dx**2
    uyy = (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) / dy**2
    u_next[1:-1,1:-1] = 2*u[1:-1,1:-1] - u_prev[1:-1,1:-1] + c**2*dt**2*(uxx+uyy)
    u_next[[0,-1],:] = 0; u_next[:,[0,-1]] = 0

def get_slope(cx_idx, cy_idx, u):
    i0, j0 = int(cx_idx), int(cy_idx)
    if 2<=i0<=nx-3 and 2<=j0<=ny-3:
        return (u[i0+1,j0] - u[i0-1,j0]) / (2*dx)
    return (u[i0+1,j0] - u[i0,j0]) / dx

# === STABLE CHART SETUP ===
fig = plt.figure(figsize=(16, 8))

ax1 = fig.add_subplot(121)
im = ax1.imshow(u.T, origin='lower', extent=[0,Lx,0,Ly],
                cmap='RdYlBu_r', vmin=-0.5, vmax=1.5, animated=True)

lines1 = []; lines2 = []; cyl_circles = []; labels = []
colors = ['lime', 'cyan', 'magenta', 'yellow', 'orange']
for cyl, col in zip(cylinders, colors):
    line1, = ax1.plot([], [], col, lw=8)
    line2, = ax1.plot([], [], col, lw=8)
    circle = plt.Circle(cyl.center, cyl.radius, fc='none', ec=col, lw=3)
    ax1.add_patch(circle)
    text = ax1.text(cyl.center[0], cyl.center[1], cyl.label, fontsize=16,
                   ha='center', va='center', color=col, weight='bold',
                   bbox=dict(boxstyle='circle', facecolor='white', alpha=0.9))
    lines1.append(line1); lines2.append(line2)
    cyl_circles.append(circle); labels.append(text)

ax1.set_xlim(0,Lx); ax1.set_ylim(0,Ly)
ax1.set_aspect('equal')
ax1.set_title('5 Spinning Cylinders - Single Wave Pulse', fontsize=14, pad=20)

info = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                verticalalignment='top')

# Rotation chart
ax2 = fig.add_subplot(122)
ax2.set_title('Live Cylinder Rotations (degrees)', fontsize=14)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Rotation Angle (degrees)')
ax2.grid(True, alpha=0.3)

plot_lines = []
time_data = []
for i, cyl in enumerate(cylinders):
    line, = ax2.plot([], [], c=colors[i], label=cyl.label, lw=3)
    cyl.plot_line = line
    plot_lines.append(line)

ax2.legend(loc='upper left')
ax2.set_xlim(0, 1.5)
ax2.set_ylim(-180, 180)

def animate(frame):
    global u_prev, u, u_next
    
    t = frame * steps_per_frame * dt
    time_data.append(t)
    
    # Wave propagation
    for _ in range(steps_per_frame):
        step_wave(u_prev, u, u_next)
        u_prev, u, u_next = u, u_next, u_prev
    
    # Cylinder physics
    for cyl in cylinders:
        cx_idx = cyl.center[0]/Lx * (nx-1)
        cy_idx = cyl.center[1]/Ly * (ny-1)
        slope = get_slope(cx_idx, cy_idx, u)
        torque = params['coupling_strength'] * slope
        
        domega_dt = (torque - params['damp_rot'] * cyl.omega) / params['Icyl']
        cyl.omega += domega_dt * dt * steps_per_frame
        cyl.theta += cyl.omega * dt * steps_per_frame
        cyl.history.append(np.degrees(cyl.theta))
    
    # Update wavefield
    im.set_array(u.T)
    
    # Update cylinder visuals
    for k, cyl in enumerate(cylinders):
        x0, y0 = cyl.center
        r = cyl.radius
        ang1 = cyl.theta % (2*np.pi)
        lines1[k].set_data([x0, x0+r*np.cos(ang1)], [y0, y0+r*np.sin(ang1)])
        ang2 = (cyl.theta + np.pi/2) % (2*np.pi)
        lines2[k].set_data([x0, x0+r*np.cos(ang2)], [y0, y0+r*np.sin(ang2)])
        labels[k].set_position((x0 + r*1.4*np.cos(ang1), y0 + r*1.4*np.sin(ang1)))
    
    # FIXED: Update rotation plots safely
    for cyl in cylinders:
        if cyl.history:
            cyl.plot_line.set_data(time_data[:len(cyl.history)], cyl.history)
    
    # FIXED: Safe max calculation
    max_thetas = [abs(c.history[-1]) for c in cylinders if c.history]
    max_theta = max(max_thetas) if max_thetas else 0
    max_omega = max(abs(c.omega) for c in cylinders)
    
    info.set_text(f't={t:.2f}s | Max rot: {max_theta:.0f}° | Max ω: {max_omega:.1f}\n'
                  f'Pulse: 1 | Cylinders: 5')
    
    # Dynamic axis limits
    if t > 0:
        ax2.set_xlim(0, max(1.5, t*1.1))
        if max_thetas:
            max_all = max(max_thetas)
            ax2.set_ylim(-max(200, max_all*1.2), max(200, max_all*1.2))
    
    return [im] + lines1 + lines2 + [info] + plot_lines

print('Multi-Cylinder Wave Simulation - Fully Stable')
ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=False, repeat=True)
plt.tight_layout()
plt.show()

# SAFE FINAL STATISTICS
print('\nFINAL ROTATION RESULTS:')
print('Cylinder | Final Angle | Max Speed')
print('-' * 35)

for i, cyl in enumerate(cylinders):
    final_angle = cyl.history[-1] if cyl.history else 0
    if len(cyl.history) > 1:
        speeds = np.abs(np.diff(cyl.history))
        max_speed = np.max(speeds)
    else:
        max_speed = 0.0
    print(f'{cyl.label:8} | {final_angle:+6.0f}°  | {max_speed:5.1f}°/frame')
