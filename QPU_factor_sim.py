#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from math import gcd


def safe_imports():
    try:
        import numpy as np
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import CDKMRippleCarryAdder
        from qiskit.quantum_info import Statevector
        from qiskit_aer import AerSimulator
        return np, QuantumCircuit, transpile, CDKMRippleCarryAdder, Statevector, AerSimulator
    except Exception as exc:
        print("Qiskit import failed.")
        print("Install with:")
        print("  python -m pip install -U qiskit qiskit-aer numpy")
        print(f"Import error: {exc}")
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Print factor pairs using the same adder-style logic.")
    p.add_argument("--N", type=int, default=15, help="Integer to factor")
    p.add_argument("--bits", type=int, default=None, help="Bit width for candidate factors")
    p.add_argument("--shots", type=int, default=256, help="Measurement shots")
    return p.parse_args()


def required_bits(n: int) -> int:
    return max(1, n.bit_length())


def encode_integer(qc, reg, value: int):
    for i in range(len(reg)):
        if (value >> i) & 1:
            qc.x(reg[i])


def basis_probabilities(statevector, np):
    probs = np.abs(statevector.data) ** 2
    support = [(idx, p) for idx, p in enumerate(probs) if p > 1e-10]
    support.sort(key=lambda t: t[1], reverse=True)
    return support


def build_logic_circuit(p, q, bits, QuantumCircuit, CDKMRippleCarryAdder):
    adder = CDKMRippleCarryAdder(num_state_qubits=bits, kind='fixed')
    qc = QuantumCircuit(*adder.qregs)
    p_reg = adder.qregs[0]
    q_reg = adder.qregs[1]
    encode_integer(qc, p_reg, p)
    encode_integer(qc, q_reg, q)
    qc.compose(adder, inplace=True)
    return qc


def factor_pairs(n: int):
    pairs = []
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % p == 0:
            q = n // p
            pairs.append((p, q))
    return pairs


def analyse_pair(N, p, q, bits, np, QuantumCircuit, transpile, CDKMRippleCarryAdder, Statevector, AerSimulator, shots):
    qc = build_logic_circuit(p, q, bits, QuantumCircuit, CDKMRippleCarryAdder)
    sv = Statevector.from_instruction(qc)
    measured_qc = qc.copy()
    measured_qc.measure_all()
    backend = AerSimulator()
    compiled = transpile(measured_qc, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()
    support = basis_probabilities(sv, np)
    support_size = len(support)
    max_prob = support[0][1] if support else 0.0
    disruption = 1.0 - max_prob
    return {
        'pair': (p, q),
        'product': p * q,
        'exact': p * q == N,
        'gcds': (gcd(p, N), gcd(q, N)),
        'support_size': support_size,
        'max_prob': max_prob,
        'disruption': disruption,
        'top_counts': sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:4],
    }


def main():
    args = parse_args()
    if args.N < 2:
        print('Input error: N must be at least 2.')
        sys.exit(2)

    bits = args.bits if args.bits is not None else required_bits(args.N)
    np, QuantumCircuit, transpile, CDKMRippleCarryAdder, Statevector, AerSimulator = safe_imports()

    pairs = factor_pairs(args.N)
    print('Adder-style factor printer')
    print(f'Target N={args.N}, bit width={bits}')

    if not pairs:
        print('No non-trivial factor pairs found.')
        sys.exit(0)

    print('\nNon-trivial factor pairs:')
    for p, q in pairs:
        print(f'  {p} * {q} = {args.N}')

    print('\nArithmetic-state diagnostics for each pair:')
    for p, q in pairs:
        try:
            info = analyse_pair(
                args.N, p, q, bits,
                np, QuantumCircuit, transpile, CDKMRippleCarryAdder, Statevector, AerSimulator,
                args.shots
            )
            print(
                f"  pair={info['pair']} exact={info['exact']} gcds={info['gcds']} "
                f"support={info['support_size']} max_prob={info['max_prob']:.6f} disruption={info['disruption']:.6f}"
            )
        except Exception as exc:
            print(f'  pair=({p}, {q}) analysis failed: {exc}')

    print('\nPrime/non-trivial factors:')
    all_factors = sorted(set([f for pair in pairs for f in pair]))
    print('  ' + ', '.join(str(f) for f in all_factors))


if __name__ == '__main__':
    main()
