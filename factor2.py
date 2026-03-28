import os
import subprocess

# Ensure required libraries are installed
subprocess.run("pip install qiskit qiskit-aer pylatexenc -q", shell=True)

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import math

os.makedirs('output', exist_ok=True)

def run_auto_quantum_sieve(N_target, shots=4000):
    # -------------------------------------------------------------------------
    # 1. AUTO-ALLOCATE QUBITS
    # -------------------------------------------------------------------------
    # Automatically determine required grid size based on binary length
    m = N_target.bit_length()
    total_q = 2 * m
    
    print(f"--- FACTORING N = {N_target} ---")
    print(f"Auto-selected {m} qubits per register.")
    print(f"Total Grid Qubits: {total_q} (Grid Size: {2**m} x {2**m})")

    # Initialize Circuit
    qc = QuantumCircuit(total_q + 1, total_q)

    # Set oracle qubit to |-> for phase kickback
    qc.x(total_q)
    qc.h(total_q)

    # Superposition
    qc.h(range(total_q))
    qc.barrier()

    # Pre-calculate targets for the Oracle
    targets = []
    for x in range(2, 2**m):
        if N_target % x == 0:
            y = N_target // x
            if y < 2**m:
                targets.append((x, y))

    # Calculate optimal Grover iterations based on dynamic grid size
    grid_states = 2**total_q
    if len(targets) > 0:
        iterations = int((math.pi / 4.0) * math.sqrt(grid_states / len(targets)))
        if iterations == 0: iterations = 1
    else:
        iterations = 1 # Run at least once to create flat noise if Prime
        
    print(f"Optimal wave iterations (bounces): {iterations}")

    # -------------------------------------------------------------------------
    # 2. ORACLE & DIFFUSION LOOP
    # -------------------------------------------------------------------------
    for step in range(iterations):
        # ORACLE
        for x, y in targets:
            # Flip zeros to ones
            for i in range(m):
                if (x & (1 << i)) == 0: qc.x(i)
            for i in range(m):
                if (y & (1 << i)) == 0: qc.x(i + m)
                
            qc.mcx(list(range(total_q)), total_q)
            
            # Uncompute flips
            for i in range(m):
                if (x & (1 << i)) == 0: qc.x(i)
            for i in range(m):
                if (y & (1 << i)) == 0: qc.x(i + m)
        qc.barrier()

        # DIFFUSION (Grover)
        qc.h(range(total_q))
        qc.x(range(total_q))

        qc.h(total_q - 1)
        qc.mcx(list(range(total_q - 1)), total_q - 1)
        qc.h(total_q - 1)

        qc.x(range(total_q))
        qc.h(range(total_q))
        qc.barrier()

    # -------------------------------------------------------------------------
    # 3. MEASUREMENT & DYNAMIC PARSING
    # -------------------------------------------------------------------------
    qc.measure(range(total_q), range(total_q))

    sim = AerSimulator()
    compiled_qc = transpile(qc, sim)
    result = sim.run(compiled_qc, shots=shots).result()
    raw_counts = result.get_counts()

    # Dynamic noise threshold
    noise_floor = shots / grid_states
    threshold = noise_floor * 2

    formatted_counts = {}
    for bitstring, count in raw_counts.items():
        # DYNAMIC SLICING: Qiskit is little-endian (q_N ... q_0)
        # Top m bits are Y, Bottom m bits are X
        y_val = int(bitstring[0 : m], 2)
        x_val = int(bitstring[m : 2*m], 2)
        
        if count > threshold:
            label = f"X={x_val},Y={y_val}"
            formatted_counts[label] = count

    # Safety Check for Primes
    if not formatted_counts:
        print(f"\n=> No factors found! The signal is flat noise. N={N_target} is likely PRIME.")
        sorted_noise = sorted(raw_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for bitstring, count in sorted_noise:
            y_val = int(bitstring[0 : m], 2)
            x_val = int(bitstring[m : 2*m], 2)
            formatted_counts[f"Noise({x_val},{y_val})"] = count
    else:
        print(f"\n=> Significant intersections found: {formatted_counts}")

    # -------------------------------------------------------------------------
    # 4. PLOTTING
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_histogram(formatted_counts, ax=ax, color='#00ffcc')

    ax.set_title(f"Auto-Scaled Qiskit Sieve (N={N_target}, {total_q} Qubits)", fontsize=14)
    ax.set_xlabel("Grid Coordinates (X, Y)")
    ax.set_ylabel("Measurement Amplitude (Shots)")
    ax.set_facecolor('#111111')
    fig.patch.set_facecolor('#111111')
    ax.xaxis.label.set_color('lightgray')
    ax.yaxis.label.set_color('lightgray')
    ax.title.set_color('white')
    ax.tick_params(colors='lightgray')

    plt.tight_layout()
    file_path = f"output/qiskit_auto_{N_target}.png"
    fig.savefig(file_path, dpi=150)
    print(f"Saved quantum output to {file_path}")

# Run the simulation for a larger composite number
run_auto_quantum_sieve(21)
