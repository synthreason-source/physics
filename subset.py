import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import math
import random

def run_auto_quantum_subset(set_size=8, target_sum=None):
    # -------------------------------------------------------------------------
    # 0. AUTO-GENERATE RANDOM SET AND TARGET
    # -------------------------------------------------------------------------
    # Generate a random set of unique integers
    subset = sorted(random.sample(range(1, set_size * 100), set_size))
    
    # If no target is provided, randomly pick a valid subset to guarantee at least one answer
    if target_sum is None:
        # Pick a random number of elements to sum (between 2 and set_size-1)
        num_to_sum = random.randint(2, set_size - 1)
        hidden_subset = random.sample(subset, num_to_sum)
        target_sum = sum(hidden_subset)
        print(f"\\n[AUTO-GEN] Planted hidden subset: {hidden_subset}")

    # -------------------------------------------------------------------------
    # 1. AUTO-ALLOCATE QUBITS
    # -------------------------------------------------------------------------
    total_q = len(subset)
    grid_states = 2**total_q
    
    print(f"--- SOLVING SUBSET SUM ---")
    print(f"Random Set S = {subset}")
    print(f"Target Sum = {target_sum}")
    print(f"Allocated {total_q} qubits (Search Space: {grid_states} combinations)")

    qc = QuantumCircuit(total_q + 1, total_q)
    qc.x(total_q)
    qc.h(total_q)
    qc.h(range(total_q))
    qc.barrier()

    targets = []
    for state in range(grid_states):
        current_sum = sum(subset[i] for i in range(total_q) if (state & (1 << i)))
        if current_sum == target_sum:
            targets.append(state)

    if len(targets) > 0:
        iterations = int((math.pi / 4.0) * math.sqrt(grid_states / len(targets)))
        if iterations == 0: iterations = 1
    else:
        iterations = 1 
        
    print(f"Optimal wave iterations (bounces): {iterations}")

    # -------------------------------------------------------------------------
    # 2. ORACLE & DIFFUSION LOOP
    # -------------------------------------------------------------------------
    for step in range(iterations):
        for state in targets:
            for i in range(total_q):
                if (state & (1 << i)) == 0: qc.x(i)
            qc.mcx(list(range(total_q)), total_q)
            for i in range(total_q):
                if (state & (1 << i)) == 0: qc.x(i)
        qc.barrier()

        qc.h(range(total_q))
        qc.x(range(total_q))
        qc.h(total_q - 1)
        if total_q > 1:
            qc.mcx(list(range(total_q - 1)), total_q - 1)
        qc.h(total_q - 1)
        qc.x(range(total_q))
        qc.h(range(total_q))
        qc.barrier()

    # -------------------------------------------------------------------------
    # 3. MEASUREMENT & DYNAMIC PARSING
    # -------------------------------------------------------------------------
    qc.measure(range(total_q), range(total_q))

    shots = 4000
    sim = AerSimulator()
    compiled_qc = transpile(qc, sim)
    result = sim.run(compiled_qc, shots=shots).result()
    raw_counts = result.get_counts()

    noise_floor = shots / grid_states
    threshold = noise_floor * 2

    formatted_counts = {}
    for bitstring, count in raw_counts.items():
        if count > threshold:
            selected_elements = []
            for i in range(total_q):
                if bitstring[total_q - 1 - i] == '1':
                    selected_elements.append(str(subset[i]))
            
            label = "Empty" if not selected_elements else f"[{'+'.join(selected_elements)}]"
            formatted_counts[label] = count

    if not formatted_counts:
        print(f"\\n=> No valid subsets found! The signal is flat noise.")
        sorted_noise = sorted(raw_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for bitstring, count in sorted_noise:
            formatted_counts[f"Noise({bitstring})"] = count
    else:
        print(f"\\n=> Significant subsets found hitting Target={target_sum}: {formatted_counts}")

    # -------------------------------------------------------------------------
    # 4. PLOTTING
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_histogram(formatted_counts, ax=ax, color='#ff00cc')

    ax.set_title(f"Quantum Subset Sum (Target: {target_sum}, Set Size: {total_q})", fontsize=14)
    ax.set_xlabel("Valid Subsets Found")
    ax.set_ylabel("Measurement Amplitude (Shots)")
    ax.set_facecolor('#111111')
    fig.patch.set_facecolor('#111111')
    ax.xaxis.label.set_color('lightgray')
    ax.yaxis.label.set_color('lightgray')
    ax.title.set_color('white')
    ax.tick_params(colors='lightgray', labelsize=10)

    plt.tight_layout()
    fig.savefig('output/qiskit_auto_subset.png', dpi=150)
    print(f"Saved quantum output to output/qiskit_auto_subset.png")

# Let's test it with an auto-generated set of 8 random numbers
run_auto_quantum_subset(set_size=20)