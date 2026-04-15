import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, mu_0, epsilon_0, e, m_e, pi

Nx=Ny=Nz=40
L=1e-3
Dx=Dy=Dz=L/Nx
Dt=0.95/(c*np.sqrt((1/Dx**2)+(1/Dy**2)+(1/Dz**2)))
Nt=120

x=np.linspace(0,L,Nx,endpoint=False)
y=np.linspace(0,L,Ny,endpoint=False)
z=np.linspace(0,L,Nz,endpoint=False)
X,Y,Z=np.meshgrid(x,y,z,indexing='ij')

eps_r=np.ones((Nx,Ny,Nz))
rod_eps=11.7
rod_w=80e-9*3
spacing=120e-9*8
for k in range(Nz):
    if (k//2)%2==0:
        mask=(np.abs(Y[:,:,k]-L/2)<rod_w/2) & (np.mod(X[:,:,k],spacing)<rod_w)
    else:
        mask=(np.abs(X[:,:,k]-L/2)<rod_w/2) & (np.mod(Y[:,:,k],spacing)<rod_w)
    eps_r[:,:,k][mask]=rod_eps

B0=0.0875
n_e0=5e18
omega_pe=np.sqrt(n_e0*e**2/(epsilon_0*m_e))
plasma_eps=1-(omega_pe/(2*pi*2.45e9))**2
plasma_mask=(eps_r==1.0)
eps_r[plasma_mask]=plasma_eps

Ex=np.zeros((Nx,Ny,Nz)); Ey=np.zeros_like(Ex); Ez=np.zeros_like(Ex)
Hx=np.zeros_like(Ex); Hy=np.zeros_like(Ex); Hz=np.zeros_like(Ex)

src=(Nx//2,Ny//2,Nz//2)
probe_pt=(Nx//2,Ny//2,Nz//2+5)
t0=25
spread=8
f0=2.45e9
energy=[]
probe=[]

for n in range(Nt):
    Hx[:-1,:-1,:-1] -= (Dt/mu_0)*((Ez[:-1,1:,:-1]-Ez[:-1,:-1,:-1])/Dy - (Ey[:-1,:-1,1:]-Ey[:-1,:-1,:-1])/Dz)
    Hy[:-1,:-1,:-1] -= (Dt/mu_0)*((Ex[:-1,:-1,1:]-Ex[:-1,:-1,:-1])/Dz - (Ez[1:,:-1,:-1]-Ez[:-1,:-1,:-1])/Dx)
    Hz[:-1,:-1,:-1] -= (Dt/mu_0)*((Ey[1:,:-1,:-1]-Ey[:-1,:-1,:-1])/Dx - (Ex[:-1,1:,:-1]-Ex[:-1,:-1,:-1])/Dy)

    Ex[1:-1,1:-1,1:-1] += (Dt/(epsilon_0*eps_r[1:-1,1:-1,1:-1]))*((Hz[1:-1,1:-1,1:-1]-Hz[1:-1,:-2,1:-1])/Dy - (Hy[1:-1,1:-1,1:-1]-Hy[1:-1,1:-1,:-2])/Dz)
    Ey[1:-1,1:-1,1:-1] += (Dt/(epsilon_0*eps_r[1:-1,1:-1,1:-1]))*((Hx[1:-1,1:-1,1:-1]-Hx[1:-1,1:-1,:-2])/Dz - (Hz[1:-1,1:-1,1:-1]-Hz[:-2,1:-1,1:-1])/Dx)
    Ez[1:-1,1:-1,1:-1] += (Dt/(epsilon_0*eps_r[1:-1,1:-1,1:-1]))*((Hy[1:-1,1:-1,1:-1]-Hy[:-2,1:-1,1:-1])/Dx - (Hx[1:-1,1:-1,1:-1]-Hx[1:-1,:-2,1:-1])/Dy)

    pulse=np.exp(-0.5*((n-t0)/spread)**2)*np.sin(2*pi*f0*Dt*n)
    Ez[src]+=pulse
    Ex[0,:,:]=Ex[-1,:,:]=0; Ey[:,0,:]=Ey[:,-1,:]=0; Ez[:,:,0]=Ez[:,:,-1]=0

    energy.append(np.sum(Ex**2+Ey**2+Ez**2)+np.sum(Hx**2+Hy**2+Hz**2))
    probe.append(Ez[probe_pt]**2)

energy=np.array(energy)
probe=np.array(probe)
np.savetxt('fdtd3d_energy.csv', energy, delimiter=',')
np.savetxt('fdtd3d_probe.csv', probe, delimiter=',')

fig,ax=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
ax[0].plot(energy,color='black')
ax[0].set_title('3D FDTD Energy')
ax[0].set_ylabel('Energy')
ax[0].grid(alpha=0.3)
ax[1].plot(10*np.log10((probe+1e-30)/(probe.max()+1e-30)),color='red')
ax[1].set_title('Probe Power (dB)')
ax[1].set_ylabel('dB')
ax[1].set_xlabel('Time step')
ax[1].grid(alpha=0.3)
plt.savefig('fdtd3d_summary.png', dpi=150)

print('OK')
print(f'B field: {B0*1e4:.1f} gauss')
print(f'ECR freq: {(e*B0/m_e)/(2*pi)/1e9:.3f} GHz')
print(f'Plasma eps: {plasma_eps:.2f}')
print(f'Final energy: {energy[-1]:.3e}')
print('Files: fdtd3d_energy.csv, fdtd3d_probe.csv, fdtd3d_summary.png') 
