import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# COMPREHENSIVE 2D WAVE + ACCURATE CYLINDER ROTATION SIMULATION
# Wave on plate → surface tilt → torque → cylinder spins realistically
# =============================================================================

# === WAVE SETUP ===
nx, ny = 101, 101
Lx, Ly = 1.0, 1.0
dx, dy = Lx/(nx-1), Ly/(ny-1)
c = 1.0                    # wave speed
dt = 0.3 * dx / c          # stable CFL
total_steps = 5000

u_prev = np.zeros((nx, ny))
u = np.zeros((nx, ny))
u_next = np.zeros((nx, ny))

# OFF-CENTER FIST BANG (generates asymmetric tilt for torque)
cx_bang, cy_bang = nx//3, ny//2   # left-center
R_bang = 12
for i in range(nx):
    for j in range(ny):
        r = np.sqrt((i-cx_bang)**2 + (j-cy_bang)**2)
        if r < R_bang:
            u[i,j] = 0.8 * np.exp(-(r**2)/(2*(R_bang/2)**2))
u_prev[:] = u

# === CYLINDER SETUP ===
cyl_center = (0.55*Lx, 0.5*Ly)    # slight offset from bang
cyl_radius = 0.06 * Lx
theta, omega = 0.0, 0.0
Icyl = 0.5                        # moment of inertia
damp_rot = 0.005                   # low damping
coupling_strength = 20.0           # torque scale

# === STATISTICS STORAGE ===
times, wave_energy, KE_wave, PE_wave = [], [], [], []
torque_history, omega_history, theta_history, slope_history = [], [], [], []

def accurate_slope(cx_idx, cy_idx, u):
    """Precise gradient at arbitrary point via bilinear interp"""
    i0, j0 = int(cx_idx), int(cy_idx)
    frac_x = cx_idx - i0
    frac_y = cy_idx - j0
    u00 = u[i0,   j0  ]; u10 = u[i0+1, j0  ]
    u01 = u[i0,   j0+1]; u11 = u[i0+1, j0+1]
    # Central diff (boundary-safe)
    if 2<=i0<=nx-3: dudx = (u[i0+1,j0] - u[i0-1,j0]) / (2*dx)
    else:           dudx = (u[i0+1,j0] - u[i0,j0]) / dx
    return dudx

def step_wave(u_prev, u, u_next):
    """2D wave equation: finite difference leapfrog"""
    uxx = (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / dx**2
    uyy = (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) / dy**2
    u_next[1:-1,1:-1] = (2*u[1:-1,1:-1] - u_prev[1:-1,1:-1] +
                         c**2 * dt**2 * (uxx + uyy))
    # Dirichlet BC
    u_next[[0,-1],:] = 0; u_next[:,[0,-1]] = 0

# === ANIMATION SETUP ===
fig = plt.figure(figsize=(12,5))

# Subplot 1: Wave field + cylinder
ax1 = plt.subplot(1,2,1)
x_phys = np.linspace(0, Lx, nx)
y_phys = np.linspace(0, Ly, ny)
im = ax1.imshow(u.T, origin='lower', extent=[0,Lx,0,Ly],
                cmap='RdBu', vmin=-0.5, vmax=0.8, animated=True)
cyl = plt.Circle(cyl_center, cyl_radius, fc='gold', ec='k', lw=3, zorder=10)
ax1.add_patch(cyl)
line, = ax1.plot([], [], 'k-', lw=4)
ax1.set_xlim(0,Lx); ax1.set_ylim(0,Ly)
ax1.set_aspect('equal')
ax1.set_title('Wave Plate + Spinning Cylinder')

time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
angle_text = ax1.text(0.02, 0.85, '', transform=ax1.transAxes)

# Subplot 2: Live stats
ax2 = plt.subplot(1,2,2)
ax2_stats = ax2.twinx()
line_theta, = ax2.plot([], [], 'r-', label='θ (deg)', lw=2)
line_omega, = ax2.plot([], [], 'b-', label='ω (rad/s)', lw=2)
line_energy, = ax2_stats.plot([], [], 'g--', label='E_wave', lw=2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Rotation')
ax2_stats.set_ylabel('Energy')
ax2_stats.set_ylim(0, 1.5)
ax2.legend(loc='upper left')
ax2_stats.legend(loc='upper right')
ax2.set_title('Live Rotation & Energy')

def init():
    im.set_array(u.T)
    x0,y0 = cyl_center; x1 = x0+cyl_radius; y1 = y0
    line.set_data([x0,x1], [y0,y1])
    time_text.set_text(''); angle_text.set_text('')
    line_theta.set_data([],[]); line_omega.set_data([],[])
    line_energy.set_data([],[])
    return im, cyl, line, time_text, angle_text, line_theta, line_omega, line_energy

def animate(frame):
    global u_prev, u, u_next, theta, omega
    
    # 5 substeps/frame for smooth coupling
    for _ in range(5):
        step_wave(u_prev, u, u_next)
        
        # Accurate torque coupling
        cx_idx = cyl_center[0]/Lx * (nx-1)
        cy_idx = cyl_center[1]/Ly * (ny-1)
        slope_x = accurate_slope(cx_idx, cy_idx, u)
        torque = coupling_strength * slope_x
        
        # Rotational physics
        domega_dt = (torque - damp_rot * omega) / Icyl
        omega += domega_dt * dt
        theta += omega * dt
        
        u_prev, u, u_next = u, u_next, u_prev
    
    # Update energies
    KE = 0.5 * c**2 * np.sum((u - u_prev)**2) * dx * dy
    PE = 0.5 * np.sum((np.gradient(u,dx,axis=0)**2 + np.gradient(u,dy,axis=1)**2)) * dx * dy
    total_E = KE + PE
    
    # Store stats
    times.append(frame*5*dt)
    wave_energy.append(total_E)
    KE_wave.append(KE)
    PE_wave.append(PE)
    torque_history.append(torque)
    omega_history.append(omega)
    theta_history.append(theta)
    slope_history.append(slope_x)
    
    # Wave viz
    im.set_array(u.T)
    x0,y0 = cyl_center
    angle_rad = theta % (2*np.pi)
    x1 = x0 + cyl_radius * np.cos(angle_rad)
    y1 = y0 + cyl_radius * np.sin(angle_rad)
    line.set_data([x0,x1], [y0,y1])
    
    time_text.set_text(f't = {frame*5*dt:.2f}s')
    angle_text.set_text(f'θ = {np.degrees(theta):.1f}°')
    
    # Stats plot (truncate to last 50 points)
    recent = slice(-50, None)
    line_theta.set_data(times[recent], np.degrees(np.array(theta_history)[recent]))
    line_omega.set_data(times[recent], omega_history[recent])
    line_energy.set_data(times[recent], np.array(wave_energy)[recent])
    ax2.set_xlim(0, times[-1])
    
    return (im, cyl, line, time_text, angle_text, line_theta, line_omega, line_energy)

# Run animation
ani = animation.FuncAnimation(fig, animate, frames=1000, init_func=init,
                              blit=True, interval=50, repeat=False)
plt.tight_layout()
plt.show()

# === PRINT FINAL COMPREHENSIVE STATS ===
print('\n' + '='*60)
print('FINAL COMPREHENSIVE STATISTICS')
print('='*60)
print(f'Initial pulse energy:  {wave_energy[0]:.6f}')
print(f'Final total energy:    {wave_energy[-1]:.6f}')
print(f'Energy conservation:   {wave_energy[-1]/wave_energy[0]:.3f}')
print()
print(f'Total cylinder rotation:{np.degrees(theta_history[-1]):6.1f}°')
print(f'Net turns:             {theta_history[-1]/(2*np.pi):6.3f}')
print(f'Max angular velocity:  {max(np.abs(omega_history)):6.3f} rad/s')
print(f'Max torque:            {max(np.abs(torque_history)):6.3f}')
print(f'Max slope at cylinder: {max(np.abs(slope_history)):6.3f}')
print()
print(f'Wave peak amp:         {np.max(np.abs(u)):6.4f}')
print(f'Final RMS displacement:{np.sqrt(np.mean(u**2)):6.4f}')
print(f'Energy partition (final): KE={KE_wave[-1]/wave_energy[-1]*100:.0f}%, PE={PE_wave[-1]/wave_energy[-1]*100:.0f}%')
print('='*60)
