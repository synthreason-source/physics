import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# 2D WAVE + 5 CYLINDERS - SINGLE CONTINUOUS CHART
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
cx1, cy1 = nx//3, ny//3
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

# === ONE SINGLE CONTINUOUS CHART ===
fig = plt.figure(figsize=(18, 9))

ax1 = fig.add_subplot(121)
im = ax1.imshow(u.T, origin='lower', extent=[0,Lx,0,Ly],
                cmap='RdYlBu_r', vmin=-0.5, vmax=1.5, animated=True)

# SINGLE CONTINUOUS DATA STORAGE
time_data = np.array([])
all_thetas = np.array([]).reshape(0, 5)  # [time, cylinders]

colors = ['lime', 'cyan', 'magenta', 'yellow', 'orange']
lines1, lines2, cyl_circles, labels = [], [], [], []

for i, (cyl, col) in enumerate(zip(cylinders, colors)):
    line1, = ax1.plot([], [], col, lw=8, solid_capstyle='round')
    line2, = ax1.plot([], [], col, lw=8, solid_capstyle='round')
    lines1.append(line1)
    lines2.append(line2)
    
    circle = plt.Circle(cyl.center, cyl.radius, fc='none', ec=col, lw=3)
    ax1.add_patch(circle)
    cyl_circles.append(circle)
    
    text = ax1.text(cyl.center[0], cyl.center[1]+cyl.radius*1.5, cyl.label, 
                   fontsize=18, ha='center', va='center', color=col, 
                   weight='bold', bbox=dict(boxstyle='circle,pad=0.3', 
                   facecolor='white', alpha=0.95, ec=col, lw=1))
    labels.append(text)

ax1.set_xlim(0,Lx); ax1.set_ylim(0,Ly)
ax1.set_aspect('equal')
ax1.set_title('Wave-Driven Cylinder Rotation', fontsize=14, pad=20)

info = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                verticalalignment='top')

# ONE SINGLE CONTINUOUS PLOT
ax2 = fig.add_subplot(122)
ax2.set_title('All Cylinders - Single Continuous Tracking', fontsize=14)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angle (degrees)')
ax2.grid(True, alpha=0.3, ls='--')

plot_lines = []
for i, label in enumerate(['A', 'B', 'C', 'D', 'E']):
    line, = ax2.plot([], [], c=colors[i], label=label, lw=3)
    plot_lines.append(line)

ax2.legend(loc='upper left', fontsize=10)

def animate(frame):
    global u_prev, u, u_next, time_data, all_thetas
    
    t = frame * steps_per_frame * dt
    
    # Wave propagation
    for _ in range(steps_per_frame):
        step_wave(u_prev, u, u_next)
        u_prev, u, u_next = u, u_next, u_prev
    
    # Physics - SINGLE FRAME DATA
    frame_thetas = []
    for cyl in cylinders:
        cx_idx = cyl.center[0]/Lx * (nx-1)
        cy_idx = cyl.center[1]/Ly * (ny-1)
        slope = get_slope(cx_idx, cy_idx, u)
        torque = params['coupling_strength'] * slope
        
        domega_dt = (torque - params['damp_rot'] * cyl.omega) / params['Icyl']
        cyl.omega += domega_dt * dt * steps_per_frame
        cyl.theta += cyl.omega * dt * steps_per_frame
        frame_thetas.append(np.degrees(cyl.theta))
    
    # SINGLE CONTINUOUS APPEND
    time_data = np.append(time_data, t) if len(time_data) > 0 else np.array([t])
    all_thetas = np.vstack([all_thetas, frame_thetas]) if len(all_thetas) > 0 else np.array([frame_thetas])
    
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
        label_x = x0 + r*1.6*np.cos(ang1)
        label_y = y0 + r*1.6*np.sin(ang1)
        labels[k].set_position((label_x, label_y))
    
    # SINGLE CONTINUOUS CHART UPDATE
    for i in range(5):
        if len(all_thetas) > 0:
            plot_lines[i].set_data(time_data, all_thetas[:,i])
    
    # Stats
    max_theta = np.max(np.abs(all_thetas[-1])) if len(all_thetas) > 0 else 0
    max_omega = max(abs(c.omega) for c in cylinders)
    
    info.set_text(f'Time: {t:.2f}s | Max: {max_theta:.0f}° | ω_max: {max_omega:.1f}\n'
                  f'Frame: {frame} | Data pts: {len(time_data)}')
    
    # Dynamic limits
    if len(time_data) > 0:
        ax2.set_xlim(0, max(2.5, time_data[-1]*1.1))
        ax2.set_ylim(-max(250, max_theta*1.4), max(250, max_theta*1.4))
    
    return [im] + lines1 + lines2 + [info] + plot_lines

print('SINGLE CONTINUOUS CHART - All 5 Cylinders Perfect Tracking')
ani = animation.FuncAnimation(fig, animate, frames=400, interval=30, blit=False, repeat=True)
plt.tight_layout()
plt.show()

# Final single continuous results
print('\nFINAL SINGLE CONTINUOUS RESULTS:')
print('Cyl | Final °  | Total Steps | Max Speed')
print('-' * 40)
for i, cyl in enumerate(cylinders):
    final_angle = all_thetas[-1,i] if len(all_thetas) > 0 else 0
    n_steps = len(all_thetas)
    speeds = np.abs(np.diff(all_thetas[:,i])) if n_steps > 1 else np.array([0])
    max_speed = np.max(speeds) if len(speeds) > 0 else 0
    print(f'{cyl.label} | {final_angle:+6.1f} | {n_steps:9d} | {max_speed:7.1f}')
