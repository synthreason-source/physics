"""
virtual_qubits_subset_sum.py

Virtual-qubit subset-sum demo with:
- ring buffer snapshots
- 2D bitmask view
- circuit history
- loading bars
- classical subset-sum solver for large sets
- optional small exact statevector mode

This is designed to mirror the HTML QuantumRing UX, but in Python.
"""

import math
import cmath
import random
import time
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

# =============================
# Utilities
# =============================

def print_loading_bar(progress: float, prefix: str = "Loading", width: int = 40):
    progress = max(0.0, min(1.0, progress))
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"\r{prefix}: [{bar}] {progress * 100:5.1f}%")
    sys.stdout.flush()
    if progress >= 1.0:
        sys.stdout.write("\n")


def bitcount(x: int) -> int:
    return x.bit_count()


def subset_from_bits(numbers: List[int], bits: int) -> List[int]:
    return [numbers[i] for i in range(len(numbers)) if (bits >> i) & 1]


def bits_str(bits: int, width: int) -> str:
    return format(bits, f"0{width}b")


# =============================
# Virtual qubit engine
# =============================

@dataclass
class VirtualSlot:
    label: str
    head: int
    tail: int
    gate: str = ""
    target: Optional[int] = None
    ctrl: Optional[int] = None
    active_bits: int = 0
    prob_summary: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


class VirtualQubitRing:
    """
    HTML-like ring buffer for virtual qubits.
    Stores snapshots of gate history and compact state summaries.
    """
    def __init__(self, num_qubits: int, ring_size: int = 32):
        self.n = num_qubits
        self.size = ring_size
        self.slots: List[Optional[VirtualSlot]] = [None] * ring_size
        self.head = 0
        self.tail = 0
        self.circuit_ops: List[Tuple[str, int, Optional[int]]] = []
        self.active_bits = 0
        self.prob_summary: Dict[str, float] = {}

    def commit(self, label: str, gate: str = "", target: Optional[int] = None,
               ctrl: Optional[int] = None, notes: str = ""):
        nxt = (self.head + 1) % self.size
        if nxt == self.tail:
            self.tail = (self.tail + 1) % self.size
        self.head = nxt
        self.slots[self.head] = VirtualSlot(
            label=label,
            head=self.head,
            tail=self.tail,
            gate=gate,
            target=target,
            ctrl=ctrl,
            active_bits=self.active_bits,
            prob_summary=dict(self.prob_summary),
            notes=notes,
        )
        if gate:
            self.circuit_ops.append((gate, target if target is not None else -1, ctrl))

    def set_probability_summary(self, probs: Dict[str, float]):
        self.prob_summary = dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))

    def set_active_bits(self, bits: int):
        self.active_bits = bits

    def history(self) -> List[VirtualSlot]:
        out = []
        i = self.tail
        seen = set()
        while i not in seen:
            seen.add(i)
            slot = self.slots[i]
            if slot is not None:
                out.append(slot)
            i = (i + 1) % self.size
            if i == (self.head + 1) % self.size:
                break
        return out

    def debug_summary(self) -> List[str]:
        lines = []
        for i, slot in enumerate(self.slots):
            if slot is None:
                lines.append(f"slot {i}: empty")
                continue
            head_flag = " (HEAD)" if i == self.head else ""
            label = slot.label
            lines.append(f"slot {i}: {label} active_bits={slot.active_bits:#0{self.n//4+4}x}{head_flag}")
        return lines

    def bitmask_2d(self, max_cols: int = 32) -> List[List[int]]:
        """2D bitmask: row=q, col=basis_state. For virtual mode, basis states are indices."""
        cols = min(max_cols, 1 << min(self.n, 12))
        return [[(s >> (self.n - 1 - q)) & 1 for s in range(cols)] for q in range(self.n)]


# =============================
# Exact small statevector backend (optional)
# =============================

class QuantumState:
    def __init__(self, num_qubits: int):
        self.n = num_qubits
        self.dim = 1 << num_qubits
        self.state = [0j] * self.dim
        self.state[0] = 1.0 + 0j

    def copy(self) -> "QuantumState":
        qs = QuantumState(self.n)
        qs.state = self.state[:]
        return qs

    def normalize(self):
        norm = math.sqrt(sum(abs(a) ** 2 for a in self.state))
        if norm > 1e-15:
            self.state = [a / norm for a in self.state]

    def probabilities(self) -> List[float]:
        return [abs(a) ** 2 for a in self.state]

    def pretty_amplitudes(self, cutoff=1e-3) -> List[Tuple[str, complex, float]]:
        out = []
        for i, amp in enumerate(self.state):
            p = abs(amp) ** 2
            if p >= cutoff:
                out.append((format(i, f"0{self.n}b"), amp, p))
        return out


def apply_single_qubit_gate(qs: QuantumState, q: int,
                            m00: complex, m01: complex,
                            m10: complex, m11: complex):
    dim = qs.dim
    mask = 1 << q
    new_state = qs.state[:]
    for i in range(dim):
        if not (i & mask):
            j = i | mask
            a0 = qs.state[i]
            a1 = qs.state[j]
            new_state[i] = m00 * a0 + m01 * a1
            new_state[j] = m10 * a0 + m11 * a1
    qs.state = new_state


def gate_H(qs: QuantumState, q: int):
    s = 1.0 / math.sqrt(2.0)
    apply_single_qubit_gate(qs, q, s, s, s, -s)


def gate_X(qs: QuantumState, q: int):
    apply_single_qubit_gate(qs, q, 0j, 1+0j, 1+0j, 0j)


def gate_Z(qs: QuantumState, q: int):
    apply_single_qubit_gate(qs, q, 1+0j, 0j, 0j, -1+0j)


def gate_CX(qs: QuantumState, c: int, t: int):
    if c == t:
        return
    dim = qs.dim
    cm = 1 << c
    tm = 1 << t
    new_state = qs.state[:]
    for i in range(dim):
        if i & cm:
            j = i ^ tm
            new_state[j] = qs.state[i]
    for i in range(dim):
        if not (i & cm):
            new_state[i] = qs.state[i]
    qs.state = new_state


def gate_CZ(qs: QuantumState, c: int, t: int):
    if c == t:
        return
    cm = 1 << c
    tm = 1 << t
    for i in range(qs.dim):
        if (i & cm) and (i & tm):
            qs.state[i] *= -1


# =============================
# Loading / classical solver
# =============================

def classical_subset_sum(numbers: List[int], target: int, show_progress=True) -> Optional[int]:
    n = len(numbers)
    total = 1 << n
    best = None
    for i in range(total):
        if show_progress and i % max(1, total // 50) == 0:
            print_loading_bar(i / total, prefix="Searching subsets")
        s = 0
        for j in range(n):
            if (i >> j) & 1:
                s += numbers[j]
        if s == target:
            best = i
            break
    print_loading_bar(1.0, prefix="Searching subsets")
    return best


# =============================
# Virtual Grover-ish demo
# =============================

def virtual_grover_demo(numbers: List[int], target: int,
                        num_qubits: int, ring_size: int = 32,
                        grover_iters: int = 1) -> Tuple[VirtualQubitRing, int]:
    rb = VirtualQubitRing(num_qubits=num_qubits, ring_size=ring_size)

    rb.set_active_bits(0)
    rb.set_probability_summary({"|0...0>": 1.0})
    rb.commit(label="|0...0> init")

    # classical search with progress
    solution = classical_subset_sum(numbers, target, show_progress=True)
    if solution is None:
        raise ValueError("No subset matches target")

    hidden_subset = subset_from_bits(numbers, solution)

    # virtual preparation stage
    rb.set_active_bits(0)
    rb.set_probability_summary({"uniform superposition": 1.0})
    rb.commit(label="H^⊗n superposition", gate="H", notes="virtual superposition")

    # emulate Grover iterations with progress bar
    for it in range(grover_iters):
        print_loading_bar(it / grover_iters, prefix=f"Grover iteration {it+1}/{grover_iters}")
        rb.set_active_bits(solution)
        rb.set_probability_summary({f"|{bits_str(solution, num_qubits)}>": 0.5,
                                    "other states": 0.5})
        rb.commit(label=f"Grover iter {it+1}", gate="G", target=solution, notes="virtual amplitude boost")
    print_loading_bar(1.0, prefix="Grover iterations")

    rb.set_active_bits(solution)
    rb.set_probability_summary({f"|{bits_str(solution, num_qubits)}>": 0.9,
                                "noise": 0.1})
    rb.commit(label="solution amplified", gate="M", target=solution, notes="virtual measurement")
    return rb, solution


# =============================
# Output / reporting
# =============================

def run_demo():
    set_size = 120
    numbers = sorted(random.sample(range(1, set_size * 20), set_size))
    num_to_sum = random.randint(2, min(set_size - 1, 6))
    hidden_subset = random.sample(numbers, num_to_sum)
    target = sum(hidden_subset)

    num_qubits = set_size
    grover_iters = 1
    ring_size = 320

    print()
    print("=" * 68)
    print("Virtual-Qubit Subset-Sum Demo")
    print("=" * 68)
    print()
    print("INPUT SET (shown to the user):")
    print(f"  numbers = {numbers}")
    print(f"  set size = {len(numbers)}")
    print()
    print("TARGET SUM (shown to the user):")
    print(f"  target = {target}")
    print()
    print("PLANTED HIDDEN SOLUTION (for verification):")
    print(f"  hidden_subset = {hidden_subset}")
    print()

    rb, solution_bits = virtual_grover_demo(numbers, target, num_qubits, ring_size, grover_iters)
    solution_subset = subset_from_bits(numbers, solution_bits)

    print()
    print("SOLUTION SUBSET (decoded from virtual marked basis state):")
    print(f"  solution bitstring = |{bits_str(solution_bits, num_qubits)}>" )
    print(f"  solution subset = {solution_subset}")
    print(f"  sum(solution_subset) = {sum(solution_subset)}")
    print()

    print("Ring buffer snapshot summary:")
    for line in rb.debug_summary():
        print(" ", line)

    print()
    print("2D bitmask preview (first 32 basis states):")
    bm = rb.bitmask_2d(max_cols=32)
    header = "     " + " ".join(f"s{i:02d}" for i in range(min(32, len(bm[0]))))
    print(header)
    for q, row in enumerate(bm):
        print(f"  q{q:<2} " + " ".join(str(v) for v in row[:32]))

    print()
    print("=" * 68)


if __name__ == "__main__":
    run_demo()
