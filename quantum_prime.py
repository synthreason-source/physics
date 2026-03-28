import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def auto_quantum_factor(N):
    # Auto-select qubits based on the binary length of N
    # Each register must be large enough to hold the maximum possible factor.
    m = N.bit_length()
    total_qubits = 2 * m
    N_states = 2**total_qubits
    
    print(f"--- FACTORING N = {N} ---")
    print(f"Auto-selected {m} qubits per register (Total: {total_qubits} qubits)")
    print(f"Superposition Grid: {N_states} possible combinations")
    
    # 1. Linear Forking (Superposition of all coordinates)
    state = np.full(N_states, 1.0 / np.sqrt(N_states), dtype=np.float64)
    
    # 2. Map the Contingent Counting Targets (The Oracle)
    # We find the specific (X, Y) states where X * Y == N and X, Y > 1
    targets = []
    for x in range(2, 2**m):
        if N % x == 0:
            y = N // x
            if y < 2**m:
                # Combine Y and X into a single state index
                idx = (y << m) | x
                targets.append(idx)
                
    # Calculate optimal wave interference cycles (Grover Iterations)
    # The larger the grid relative to the targets, the more bounces needed to amplify the resonance.
    if len(targets) > 0:
        iterations = int((np.pi / 4.0) * np.sqrt(N_states / len(targets)))
    else:
        iterations = 1
        print("No factors found (Prime number).")
        
    print(f"Optimal interference cycles: {iterations}")
    
    # 3. Wave Propagation (Interference Grid)
    for i in range(iterations):
        # Phase Kickback (Contingent counting toggle)
        for t in targets:
            state[t] *= -1.0
        
        # Grover Diffusion (Resonant amplification)
        mean_val = np.mean(state)
        state = 2.0 * mean_val - state
        
    probs = np.abs(state)**2
    return m, probs, targets

# Let's factor a larger semi-prime, like N = 35 (Factors: 5, 7)
N_val = 3500
m, probs, targets = auto_quantum_factor(N_val)

# Extract the top 15 most probable measurements for plotting
top_indices = np.argsort(probs)[-15:][::-1]

labels = []
plot_probs = []
colors = []

print("\nTop Measurement Results:")
for idx in top_indices:
    y = idx >> m
    x = idx & ((1 << m) - 1)
    labels.append(f"X={x}\nY={y}")
    plot_probs.append(probs[idx])
    
    is_target = idx in targets
    if is_target:
        colors.append('#00ffcc')  # Cyan for correct factors
        print(f" -> X={x:2}, Y={y:2} : {probs[idx]:.4f} (CORRECT FACTOR)")
    else:
        colors.append('#444444')  # Gray for quantum noise
        print(f"    X={x:2}, Y={y:2} : {probs[idx]:.4f}")

# Plotting the data
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#111111')
ax.set_facecolor('#111111')

bars = ax.bar(labels, plot_probs, color=colors)

ax.set_title(f"Auto-Scaled Quantum Factorization (N={N_val})", color='white', pad=20, fontsize=14)
ax.set_ylabel("Measurement Probability", color='lightgray')
ax.tick_params(colors='lightgray', rotation=0)

for spine in ax.spines.values():
    spine.set_edgecolor('#333333')

plt.tight_layout()
fig.savefig("output/auto_quantum_factor.png", dpi=150)
print(f"\nGenerated output/auto_quantum_factor.png")