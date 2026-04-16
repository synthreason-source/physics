import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, mu_0, epsilon_0, e, m_e, pi

# 1. PHYSICAL CALIBRATION
f0 = 2.45e9                  # 2.45 GHz Microwave source
B0 = 0.0875                  # 875 Gauss (0.0875 Tesla)
omega_ce = (e * B0) / m_e    # Cyclotron frequency
omega_src = 2 * pi * f0
wavelength = c / f0          # ~12.24 cm

# 2. DOMAIN & GRID SETUP
Nx, Ny, Nz = 60, 60, 60
Lx = Ly = Lz = 0.5           # 50 cm cube domain
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
dt = 0.95 / (c * np.sqrt((1/dx**2) + (1/dy**2) + (1/dz**2)))
Nt = 500

# 3. GEOMETRY: Woodpile Photonic Crystal & ECR Defect
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
z = np.linspace(0, Lz, Nz, endpoint=False)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

copper_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
rod_w, spacing = 0.02, 0.085 # 2cm rod width, 8.5cm spacing

for k in range(Nz):
    if (k // 2) % 2 == 0:
        mask = (np.abs(Y[:, :, k] - Ly/2) < rod_w/2) & (np.mod(X[:, :, k], spacing) < rod_w)
    else:
        mask = (np.abs(X[:, :, k] - Lx/2) < rod_w/2) & (np.mod(Y[:, :, k], spacing) < rod_w)
    copper_mask[:, :, k] = mask

# Central Defect for Plasma (Resonance Zone)
defect_radius = 0.07 
plasma_zone = (np.sqrt((X-Lx/2)**2 + (Y-Ly/2)**2 + (Z-Lz/2)**2)) < defect_radius
copper_mask[plasma_zone] = False

# 4. PLASMA INITIALIZATION
n_e = 7.5e16                 # Electron density
omega_p2 = (n_e * e**2) / (m_e * epsilon_0)
nu = 1e7                     # Collision frequency

# Field and Current Arrays
Ex, Ey, Ez = [np.zeros((Nx, Ny, Nz)) for _ in range(3)]
Hx, Hy, Hz = [np.zeros((Nx, Ny, Nz)) for _ in range(3)]
Jx, Jy, Jz = [np.zeros((Nx, Ny, Nz)) for _ in range(3)]

# 5. SIMULATION LOOP
src_idx = (Nx//2, Ny//2, 4)
plasma_energy = []
field_energy = []

for n in range(Nt):
    # Update Magnetic Fields
    Hx[:, :-1, :-1] -= (dt/mu_0) * ((Ez[:, 1:, :-1] - Ez[:, :-1, :-1])/dy - (Ey[:, :-1, 1:] - Ey[:, :-1, :-1])/dz)
    Hy[:-1, :, :-1] -= (dt/mu_0) * ((Ex[:-1, :, 1:] - Ex[:-1, :, :-1])/dz - (Ez[1:, :, :-1] - Ez[:-1, :, :-1])/dx)
    Hz[:-1, :-1, :] -= (dt/mu_0) * ((Ey[1:, :-1, :] - Ey[:-1, :-1, :])/dx - (Ex[:-1, 1:, :] - Ex[:-1, :-1, :])/dy)

    # Update Plasma Current (Magnetized Lorentz update for ECR)
    jx_old, jy_old = Jx[plasma_zone].copy(), Jy[plasma_zone].copy()
    Jx[plasma_zone] = (1/(1+nu*dt))*(jx_old + dt*omega_p2*epsilon_0*Ex[plasma_zone] - omega_ce*dt*jy_old)
    Jy[plasma_zone] = (1/(1+nu*dt))*(jy_old + dt*omega_p2*epsilon_0*Ey[plasma_zone] + omega_ce*dt*jx_old)
    Jz[plasma_zone] = (1/(1+nu*dt))*(Jz[plasma_zone] + dt*omega_p2*epsilon_0*Ez[plasma_zone])

    # Update Electric Fields
    Ex[:, 1:-1, 1:-1] += (dt/epsilon_0) * ((Hz[:, 1:-1, 1:-1] - Hz[:, :-2, 1:-1])/dy - (Hy[:, 1:-1, 1:-1] - Hy[:, 1:-1, :-2])/dz - Jx[:, 1:-1, 1:-1])
    Ey[1:-1, :, 1:-1] += (dt/epsilon_0) * ((Hx[1:-1, :, 1:-1] - Hx[1:-1, :, :-2])/dz - (Hz[1:-1, :, 1:-1] - Hz[:-2, :, 1:-1])/dx - Jy[1:-1, :, 1:-1])
    Ez[1:-1, 1:-1, :] += (dt/epsilon_0) * ((Hy[1:-1, 1:-1, :] - Hy[:-2, 1:-1, :])/dx - (Hx[1:-1, 1:-1, :] - Hx[1:-1, :-2, :])/dy - Jz[1:-1, 1:-1, :])

    # PEC Boundaries and Source
    Ex[copper_mask] = Ey[copper_mask] = Ez[copper_mask] = 0
    pulse = np.sin(omega_src * n * dt) * np.exp(-0.5 * ((n-40)/15)**2)
    Ez[src_idx] += pulse

    # Energy Tracking
    w_field = 0.5 * np.sum(epsilon_0*(Ex**2+Ey**2+Ez**2) + mu_0*(Hx**2+Hy**2+Hz**2)) * (dx*dy*dz)
    w_plasma = 0.5 * (1 / (epsilon_0 * omega_p2)) * np.sum(Jx**2+Jy**2+Jz**2) * (dx*dy*dz)
    field_energy.append(w_field)
    plasma_energy.append(w_plasma)

# 6. STATISTICS REPORTING
print("--- ECR SIMULATION STATISTICS ---")
print(f"Source Frequency:     {f0/1e9:.2f} GHz")
print(f"Resonant B-Field:     {B0*1e4:.1f} Gauss")
print(f"Cyclotron Frequency:  {omega_ce/(2*pi*1e9):.4f} GHz")
print(f"Resonance Match:      {((1 - abs(omega_ce/(2*pi)-f0)/f0)*100):.2f}%")
print(f"Domain Resolution:    {wavelength/dx:.2f} cells per wavelength")
print(f"Total Time Duration:  {Nt*dt*1e9:.2f} ns")
print(f"Final Plasma Energy:  {plasma_energy[-1]:.3e} J")
print("---------------------------------")

# 7. VISUALIZATION
plt.figure(figsize=(10, 5))
plt.plot(field_energy, label='EM Field Energy')
plt.plot(plasma_energy, label='Plasma Kinetic Energy', linewidth=2)
plt.title("ECR Energy Exchange at 875 Gauss / 2.45 GHz")
plt.xlabel("Time Step")
plt.ylabel("Energy (J)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
