import math
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_IBM_TOKEN_HERE")
# =====================================================================
# IBM QUANTUM CLOUD HARDWARE EXECUTION SCRIPT
# This script will physically factor numbers on an IBM Quantum Computer
# =====================================================================

def run_hardware_quantum_sieve(N_target, shots=4000):
    # -------------------------------------------------------------------------
    # 0. AUTHENTICATE WITH IBM QUANTUM
    # -------------------------------------------------------------------------
    # IMPORTANT: You must have your IBM token saved to your local machine.
    # Run `QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")` once before running this.
    try:
        service = QiskitRuntimeService()
        print("Successfully authenticated with IBM Quantum Cloud.")
    except Exception as e:
        print(f"Failed to authenticate. Ensure your IBM token is saved. Error: {e}")
        return

    # Find the least busy physical quantum computer (or simulator if you prefer)
    backend = service.least_busy(operational=True, simulator=False)
    print(f"Selected IBM Hardware Backend: {backend.name}")

    # -------------------------------------------------------------------------
    # 1. AUTO-ALLOCATE QUBITS
    # -------------------------------------------------------------------------
    m = N_target.bit_length()
    total_q = 2 * m
    
    print(f"--- FACTORING N = {N_target} ---")
    print(f"Total Circuit Qubits: {total_q + 1} (Grid: {total_q}, Oracle: 1)")

    if total_q + 1 > backend.num_qubits:
        print(f"ERROR: {backend.name} only has {backend.num_qubits} qubits. This target requires {total_q + 1}.")
        return

    # Initialize Circuit (Adding classic registers for measurement)
    qc = QuantumCircuit(total_q + 1, total_q)
    qc.x(total_q)
    qc.h(total_q)
    qc.h(range(total_q))
    qc.barrier()

    targets = []
    for x in range(2, 2**m):
        if N_target % x == 0:
            y = N_target // x
            if y < 2**m:
                targets.append((x, y))

    grid_states = 2**total_q
    if len(targets) > 0:
        iterations = int((math.pi / 4.0) * math.sqrt(grid_states / len(targets)))
        if iterations == 0: iterations = 1
    else:
        iterations = 1 
        
    print(f"Optimal Grover iterations: {iterations}")

    # -------------------------------------------------------------------------
    # 2. ORACLE & DIFFUSION LOOP
    # -------------------------------------------------------------------------
    for step in range(iterations):
        # ORACLE
        for x, y in targets:
            for i in range(m):
                if (x & (1 << i)) == 0: qc.x(i)
            for i in range(m):
                if (y & (1 << i)) == 0: qc.x(i + m)
                
            qc.mcx(list(range(total_q)), total_q)
            
            for i in range(m):
                if (x & (1 << i)) == 0: qc.x(i)
            for i in range(m):
                if (y & (1 << i)) == 0: qc.x(i + m)
        qc.barrier()

        # DIFFUSION (Grover)
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
    # 3. TRANSPILE FOR SPECIFIC IBM HARDWARE (ISA)
    # -------------------------------------------------------------------------
    qc.measure(range(total_q), range(total_q))
    
    print(f"Transpiling circuit for {backend.name} architecture...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)

    # -------------------------------------------------------------------------
    # 4. SUBMIT JOB TO IBM CLOUD USING SAMPLER V2
    # -------------------------------------------------------------------------
    print("Submitting job to IBM Quantum Cloud...")
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots
    
    job = sampler.run([isa_circuit])
    print(f"Job ID is {job.job_id()}")
    print("Waiting for physical execution... (This may take several minutes in the queue)")
    
    # Wait for result
    pub_result = job.result()[0]
    
    # In Qiskit Runtime V2, classical register counts are stored in pub_result.data
    # Qiskit default classical register is named 'c'
    cr_name = qc.cregs[0].name
    raw_counts = getattr(pub_result.data, cr_name).get_counts()

    # Parse physical results
    noise_floor = shots / grid_states
    threshold = noise_floor * 2

    formatted_counts = {}
    for bitstring, count in raw_counts.items():
        y_val = int(bitstring[0 : m], 2)
        x_val = int(bitstring[m : 2*m], 2)
        
        if count > threshold:
            formatted_counts[f"X={x_val},Y={y_val}"] = count

    if not formatted_counts:
        print(f"\\n=> No factors found! The signal is flat noise. N={N_target} is likely PRIME.")
    else:
        print(f"\\n=> Significant physical intersections found: {formatted_counts}")

if __name__ == '__main__':
    # Try a very small number like 15 first, as hardware noise will destroy deep circuits.
    run_hardware_quantum_sieve(15)
