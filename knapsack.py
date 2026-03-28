import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import math
import random

def run_auto_quantum_knapsack(items, target_capacity, shots=4000):
    # -------------------------------------------------------------------------
    # 0. PARSE THE ITEMS DICTIONARY (LABEL -> WEIGHT)
    # -------------------------------------------------------------------------
    # Ensure strict separation: Labels (Strings) vs Weights (Integers)
    labels = list(items.keys())
    weights = list(items.values())
    total_q = len(items)
    grid_states = 2**total_q
    
    print(f"--- SOLVING QUANTUM KNAPSACK / SUBSET SUM ---")
    print(f"Inventory (Label: Weight): {items}")
    print(f"Target Capacity = {target_capacity}")
    print(f"Allocated {total_q} qubits (Search Space: {grid_states} combinations)")

    # -------------------------------------------------------------------------
    # 1. INITIALIZE CIRCUIT
    # -------------------------------------------------------------------------
    qc = QuantumCircuit(total_q + 1, total_q)
    qc.x(total_q)
    qc.h(total_q)
    qc.h(range(total_q))
    qc.barrier()

    # Pre-calculate targets for the Oracle using purely the integer WEIGHTS
    targets = []
    for state in range(grid_states):
        # Calculate sum by matching the bitmask against the weight integer list
        current_sum = sum(weights[i] for i in range(total_q) if (state & (1 << i)))
        if current_sum == target_capacity:
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

    sim = AerSimulator()
    compiled_qc = transpile(qc, sim)
    result = sim.run(compiled_qc, shots=shots).result()
    raw_counts = result.get_counts()

    noise_floor = shots / grid_states
    threshold = noise_floor * 2

    formatted_counts = {}
    for bitstring, count in raw_counts.items():
        if count > threshold:
            # Map the collapsed bits back to the String LABELS, appending the Weight in brackets for clarity
            selected_labels = []
            for i in range(total_q):
                if bitstring[total_q - 1 - i] == '1':
                    selected_labels.append(f"{labels[i]} ({weights[i]})")
            
            label_text = "Empty" if not selected_labels else "\\n+\\n".join(selected_labels)
            formatted_counts[label_text] = count

    if not formatted_counts:
        print(f"\\n=> No valid subsets found! The signal is flat noise.")
    else:
        print(f"\\n=> Significant item combinations found hitting Target={target_capacity}:")
        for k, v in formatted_counts.items():
            clean_k = k.replace('\\n', ' ')
            print(f"   {clean_k} -> {v} shots")

    # -------------------------------------------------------------------------
    # 4. PLOTTING
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_histogram(formatted_counts, ax=ax, color='#ff00cc')

    ax.set_title(f"Quantum Item Selection (Target Weight: {target_capacity})", fontsize=14)
    ax.set_xlabel("Valid Label + Weight Combinations")
    ax.set_ylabel("Measurement Amplitude (Shots)")
    ax.set_facecolor('#111111')
    fig.patch.set_facecolor('#111111')
    ax.xaxis.label.set_color('lightgray')
    ax.yaxis.label.set_color('lightgray')
    ax.title.set_color('white')
    ax.tick_params(colors='lightgray', labelsize=8)

    plt.tight_layout()
    fig.savefig('output/qiskit_labeled_knapsack.png', dpi=150)
    print(f"Saved quantum output to output/qiskit_labeled_knapsack.png")


# -------------------------------------------------------------------------
# AUTO-GENERATOR FOR LABELS AND WEIGHTS
# -------------------------------------------------------------------------
def make_auto_inventory(n_items=80, min_weight=1, max_weight=100):
    # Instead of a small pool of Greek letters, generate numbered strings 
    # so we never run out of unique labels, no matter how high n_items goes.
    chosen_labels = [f"Item_{i+1}" for i in range(n_items)]
    
    # Generate dictionary mapping String Labels -> Integer Weights
    return {
        label: random.randint(min_weight, max_weight)
        for label in chosen_labels
    }

# Generate 8 random items
my_inventory = make_auto_inventory(n_items=11)

# Plant a hidden combination to guarantee a valid target sum
hidden_keys = random.sample(list(my_inventory.keys()), 3)
target_weight = sum(my_inventory[k] for k in hidden_keys)

print(f"[AUTO-GEN] Planted a valid combination: {hidden_keys} which sums to {target_weight}")

# Run the simulation
run_auto_quantum_knapsack(my_inventory, target_weight)