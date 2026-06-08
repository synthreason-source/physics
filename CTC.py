import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. SIMULATION CONFIGURATION ---
N_CONTROL_POINTS = 2
EPSILON = 0.4          # Tubular neighborhood threshold radius
K_SIGMOID = 15.0       # Sigmoid sharpness parameter
LAMBDA_SMOOTH = 0.1    # Smoothness penalty weight
LEARNING_RATE = 0.05
TOTAL_FRAMES = 240      # Total length of the generated GIF

# Physical Energy Constants
VOLTAGE = 1.2          
CAPACITANCE = 2e-15    
BOLTZMANN_K = 1.380649e-23  
TEMPERATURE = 298.15   
LANDAUER_CONSTANT = BOLTZMANN_K * TEMPERATURE * np.log(2)

# --- 2. SETUP GEOMETRY & OBSERVATIONS ---
# Ground Truth Latent 5D Curve
t = np.linspace(0, 2 * np.pi, 100)
gt_curve = np.zeros((100, 5))
gt_curve[:, 0] = np.sin(t) * 2
gt_curve[:, 1] = np.cos(t) * 2
gt_curve[:, 2] = t / 2
gt_curve[:, 3] = np.sin(2 * t) * 0.5  
gt_curve[:, 4] = np.cos(2 * t) * 0.5  

# Scan Trajectory S(s)
s_vals = np.linspace(0, 2 * np.pi, 300)
scan_points = np.zeros((300, 5))
scan_points[:, 0] = np.sin(s_vals) * 2.2
scan_points[:, 1] = np.cos(s_vals) * 2.2
scan_points[:, 2] = s_vals / 2
scan_points[:, 3] = np.sin(3 * s_vals) * 0.3
scan_points[:, 4] = np.cos(3 * s_vals) * 0.3

# Binary Observations Chi
chi = np.zeros(len(scan_points))
for i, s in enumerate(scan_points):
    dists = np.linalg.norm(gt_curve - s, axis=1)
    min_dist = np.min(dists)
    chi[i] = 1.0 if min_dist <= EPSILON else 0.0

# Separate sensor hits/misses for clear plotting
hit_points = scan_points[chi == 1]

# Initialize Estimated Control Points from positive detections
indices = np.linspace(0, len(hit_points) - 1, N_CONTROL_POINTS).astype(int)
control_points = hit_points[indices].copy() + np.random.normal(0, 0.1, (N_CONTROL_POINTS, 5))

# State tracking for binary toggles
prev_hat_chi_binary = np.zeros(len(scan_points))

# --- 3. ANIMATION & OPTIMIZATION SETUP ---
fig = plt.figure(figsize=(10, 7), facecolor='#f4f6f9')
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    global control_points, prev_hat_chi_binary
    ax.clear()
    
    # --- MATH STEP (Gradient Descent Optimization) ---
    g = control_points
    N = len(g)
    hat_chi = np.zeros(len(scan_points))
    min_indices = np.zeros(len(scan_points), dtype=int)
    distances = np.zeros(len(scan_points))
    
    for i, s in enumerate(scan_points):
        dists = np.linalg.norm(g - s, axis=1)
        min_idx = np.argmin(dists)
        min_indices[i] = min_idx
        distances[i] = dists[min_idx]
        hat_chi[i] = 1.0 / (1.0 + np.exp(-K_SIGMOID * (EPSILON - distances[i])))

    # Compute Joules expenditure per binary toggle
    current_hat_chi_binary = (hat_chi >= 0.5).astype(int)
    num_toggles = np.sum(current_hat_chi_binary != prev_hat_chi_binary)
    dynamic_joules = num_toggles * (CAPACITANCE * (VOLTAGE ** 2))
    landauer_joules = num_toggles * LANDAUER_CONSTANT
    prev_hat_chi_binary = current_hat_chi_binary.copy()

    # Compute Gradients
    grad = np.zeros_like(g)
    for i, s in enumerate(scan_points):
        idx = min_indices[i]
        dist = distances[i]
        if dist < 1e-5: continue
        d_loss = -2 * (chi[i] - hat_chi[i])
        d_sig = hat_chi[i] * (1.0 - hat_chi[i]) * (-K_SIGMOID)
        d_dist = (g[idx] - s) / dist
        grad[idx] += d_loss * d_sig * d_dist
        
    for i in range(1, N - 1):
        laplacian = g[i+1] - 2*g[i] + g[i-1]
        grad[i] += LAMBDA_SMOOTH * (-4 * laplacian)
        grad[i+1] += LAMBDA_SMOOTH * (2 * laplacian)
        grad[i-1] += LAMBDA_SMOOTH * (2 * laplacian)

    control_points -= LEARNING_RATE * grad

    # --- 3D RENDERING (Projected to X, Y, Z for visualization) ---
    # Ground Truth Curve
    ax.plot(gt_curve[:, 0], gt_curve[:, 1], gt_curve[:, 2], color='#2c3e50', linewidth=2.5, label='Latent GT Curve')
    # Sensor Hits
    ax.scatter(hit_points[:, 0], hit_points[:, 1], hit_points[:, 2], color='#2ecc71', s=6, alpha=0.5, label='Sensor Hits ($\chi=1$)')
    # Estimated Active Curve
    ax.plot(g[:, 0], g[:, 1], g[:, 2], color='#e67e22', marker='o', markersize=4, linewidth=2, label='Estimated Curve ($\gamma$)')
    
    # Labels & Title Overlay
    ax.set_title(f"Tomography Reconstruction Loop — Frame {frame+1}/{TOTAL_FRAMES}", fontsize=12, fontweight='bold', pad=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper left')
    
    # Thermodynamic HUD Overlay Text
    hud_text = (
        f"Iteration Toggles: {num_toggles}\n"
        f"Dynamic Joule Loss: {dynamic_joules:.3e} J\n"
        f"Landauer Limit: {landauer_joules:.3e} J"
    )
    ax.text2D(0.02, 0.02, hud_text, transform=ax.transAxes, fontsize=10, 
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.85))
    
    # Slowly rotate camera viewport for dynamic 3D depth perception
    camera_elevation = 20 + (np.sin(frame / 10) * 5)
    ax.view_init(elev=camera_elevation, azim=frame * 1.5)

# --- 4. GENERATE AND SAVE ---
print("Compiling math iterations and rendering frames into GIF...")
ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=150)

# Save the animation using Pillow
ani.save('tomography_reconstruction.gif', writer='pillow', fps=7)
plt.close()
print("Success! Animation saved as 'tomography_reconstruction.gif'")
