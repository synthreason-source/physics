"""
QHDC‑hyperobject prime‑solver: input one N, output only p and q.

Enter a composite integer N (>= 15):
N = [...]
p = X, q = Y
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import DiagonalGate, QFTGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from fractions import Fraction
from math import gcd, pi
import numpy as np

np.random.seed(42)

# --- 1. User input: one composite N ---------------------------

while True:
    try:
        N = int(input("Enter a composite integer N (>= 15): "))
        if N >= 15:
            break
    except ValueError:
        pass

print(f"N = {N}")

# Random base a coprime to N
# Use Python's random.randint (no int32 limit) to avoid high‑is‑out‑of‑bounds
import random
a = random.randint(2, N - 1)
while gcd(a, N) != 1:
    a = random.randint(2, N - 1)

# --- 2. QHDC‑hyperobject register (4‑qubit, 16‑D) --------------

D = 2**4
n_sys_qubits = 4
n_ancilla = 1

qr_sys = QuantumRegister(n_sys_qubits, name="sys")
qr_anc = QuantumRegister(n_ancilla, name="anc")
cr = ClassicalRegister(n_sys_qubits, name="readout")

# --- 3. Hyperobject circuit (no measurements, toy period r=4) -

qc_sv = QuantumCircuit(qr_sys, qr_anc)

# Toy modular‑exponentiation phase oracle (period r = 4)
r_toy = 4
phases_cos = []
for x in range(D):
    f_x = x % r_toy
    phase = 2.0 * pi * f_x / r_toy
    phases_cos.append(np.exp(1j * phase))
oracle_modexp = DiagonalGate(phases_cos)

# Epoch hypervectors (phase‑oracle DiagonalGates)
v1 = np.random.choice([-1.0, +1.0], size=D)
v2 = np.random.choice([-1.0, +1.0], size=D)

phases1 = [np.pi * vi / 2.0 for vi in v1]
phases2 = [np.pi * vi / 2.0 for vi in v2]

oracle1 = DiagonalGate([np.exp(1j * phi) for phi in phases1])
oracle2 = DiagonalGate([np.exp(1j * phi) for phi in phases2])

# Initialize + LCU‑bundling + phase‑binding
qc_sv.h(qr_sys)
qc_sv.h(qr_anc[0])
qc_sv.append(oracle1.control(1), [qr_anc[0]] + qr_sys[:])
qc_sv.append(oracle2.control(1), [qr_anc[0]] + qr_sys[:])
qc_sv.cp(pi / 3.0, qr_sys[0], qr_sys[2])
qc_sv.append(oracle_modexp, qr_sys)

# Inverse QFT (QFTGate, no 'inverse' kw in constructor)
qft = QFTGate(num_qubits=n_sys_qubits)
iqft = qft.inverse()
qc_sv.append(iqft, qr_sys)

# --- 4. Measurement circuit for shots -------------------------

qc = qc_sv.copy()
qc.add_register(cr)

qc.measure(qr_anc[0], cr[0])
qc.measure(qr_sys[1], cr[1])
qc.measure(qr_sys[2], cr[2])
qc.measure(qr_sys[3], cr[3])

# --- 5. Simulate and get period from histogram ----------------

sim = AerSimulator()
compiled = transpile(qc, sim, optimization_level=1)
job = sim.run(compiled, shots=1000)
counts = job.result().get_counts()

best_str = max(counts, key=counts.get)
meas_int = int(best_str[1:], 2)  # ignore ancilla bit
phi = meas_int / (2**n_sys_qubits)

try:
    r = Fraction(phi).limit_denominator(N).denominator
except Exception:
    r = 0

# --- 6. Compute potential factors (if r even) -----------------

f1 = 1
f2 = 1
if r > 0 and r % 2 == 0:
    power = r // 2
    try:
        plus = (a ** power + 1) % N
        minus = (a ** power - 1) % N
        f1 = gcd(plus, N)
        f2 = gcd(minus, N)
    except OverflowError:
        pass

# --- 7. Final output: only p and q ----------------------------

if 1 < f1 < N and 1 < f2 < N:
    p = min(f1, f2)
    q = max(f1, f2)
    print(f"p = {p}, q = {q}")
elif 1 < f1 < N:
    p = f1
    q = N // f1
    print(f"p = {p}, q = {q}")
elif 1 < f2 < N:
    p = f2
    q = N // f2
    print(f"p = {p}, q = {q}")
else:
    print("No factor found")
