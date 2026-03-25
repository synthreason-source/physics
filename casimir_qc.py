#!/usr/bin/env python3
"""
casimir_qc.py
─────────────────────────────────────────────────────────────────────
Casimir/Photonic-Bandgap Noise-Free Bell State Generator
Using Qiskit Aer thermal relaxation noise model

Physics:
  The vacuum mode density (LDOS) between two Casimir plates or inside
  a photonic bandgap crystal is suppressed by a factor η.
  This directly suppresses spontaneous emission (T1 decoherence):

      T1_eff = η × T1_free
      T2_eff = min(2·T1_eff, η × T2_free)

  A qubit at frequency ω inside the bandgap cannot emit a photon
  (no vacuum modes exist at ω) → coherence time → ∞ as η → ∞.

Requires:
  pip install qiskit==2.3.1 qiskit-aer==0.17.2 matplotlib numpy
─────────────────────────────────────────────────────────────────────
"""

import os, numpy as np, matplotlib.pyplot as plt
os.environ["QISKIT_SUPPRESS_1_0_IMPORT_ERROR"] = "1"

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit.quantum_info import state_fidelity, DensityMatrix, Statevector

# ── Baseline qubit parameters (superconducting, 2025 state-of-art) ──
T1_FREE  = 100e3   # ns  (100 µs)
T2_FREE  = 80e3    # ns  (80 µs)
TIME_H   = 50      # ns  Hadamard gate
TIME_CX  = 300     # ns  CNOT gate
TIME_M   = 1000    # ns  Measurement

IDEAL_BELL = Statevector([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])  # |Φ+⟩


# ─────────────────────────────────────────────────────────────
# Core Classes
# ─────────────────────────────────────────────────────────────

class CasimirNoiseModel:
    """
    Thermal relaxation noise model scaled by LDOS suppression factor η.

    Usage:
        nm = CasimirNoiseModel(eta=1000)     # 1000× bandgap suppression
        sim = AerSimulator(noise_model=nm.build())
    """
    def __init__(self, eta: float = 1.0):
        self.eta = eta
        self.T1  = eta * T1_FREE
        self.T2  = min(2.0 * self.T1, eta * T2_FREE)

    def build(self) -> NoiseModel:
        e_h  = thermal_relaxation_error(self.T1, self.T2, TIME_H)
        e_cx = (thermal_relaxation_error(self.T1, self.T2, TIME_CX)
                .expand(thermal_relaxation_error(self.T1, self.T2, TIME_CX)))
        e_m  = thermal_relaxation_error(self.T1, self.T2, TIME_M)
        nm = NoiseModel()
        for q in [0, 1]:
            nm.add_quantum_error(e_h, ["h"],       [q])
            nm.add_quantum_error(e_m, ["measure"], [q])
        nm.add_quantum_error(e_cx, ["cx"], [0, 1])
        return nm

    def __repr__(self):
        return (f"CasimirNoiseModel(η={self.eta:.1f}, "
                f"T1={self.T1/1e3:.1f}µs, T2={self.T2/1e3:.1f}µs)")


class BellStateGenerator:
    """
    Noise-free Bell pair generator using photonic bandgap suppression.

    Generates all four Bell states and measures fidelity.
    Usage:
        gen = BellStateGenerator(eta=5000)
        result = gen.run("phi_plus")
        print(result["fidelity"])
    """
    BELL_CONFIGS = {
        "phi_plus":  (False, False),   # (apply_x, apply_z)
        "phi_minus": (False, True),
        "psi_plus":  (True,  False),
        "psi_minus": (True,  True),
    }

    def __init__(self, eta: float = 1.0, shots: int = 4096):
        self.eta   = eta
        self.shots = shots
        nm         = CasimirNoiseModel(eta).build()
        self.sim_dm = AerSimulator(noise_model=nm, method="density_matrix")
        self.sim_sv = AerSimulator(noise_model=nm, method="statevector")

    def _circuit(self, state: str, measure: bool = True) -> QuantumCircuit:
        apply_x, apply_z = self.BELL_CONFIGS[state]
        qc = QuantumCircuit(2, 2 if measure else 0, name=f"Bell-{state}")
        qc.h(0)
        qc.cx(0, 1)
        if apply_x: qc.x(0)
        if apply_z: qc.z(0)
        if measure:
            qc.measure([0, 1], [0, 1])
        else:
            qc.save_density_matrix()
        return qc

    def run(self, state: str = "phi_plus") -> dict:
        # Density matrix for fidelity
        qc_dm = transpile(self._circuit(state, measure=False),
                          self.sim_dm, optimization_level=0)
        dm    = DensityMatrix(
            self.sim_dm.run(qc_dm, shots=1).result().data()["density_matrix"])

        # Measurement for counts
        qc_sv = transpile(self._circuit(state, measure=True),
                          self.sim_sv, optimization_level=0)
        counts = self.sim_sv.run(qc_sv, shots=self.shots).result().get_counts()

        ideal_sv = self._ideal_statevector(state)
        return {
            "state":    state,
            "eta":      self.eta,
            "T1_us":    self.eta * T1_FREE / 1e3,
            "fidelity": float(state_fidelity(dm, ideal_sv)),
            "counts":   counts,
        }

    def run_all(self) -> list:
        return [self.run(s) for s in self.BELL_CONFIGS]

    @staticmethod
    def _ideal_statevector(state: str) -> Statevector:
        s = 1/np.sqrt(2)
        svs = {
            "phi_plus":  Statevector([ s,  0,  0,  s]),
            "phi_minus": Statevector([ s,  0,  0, -s]),
            "psi_plus":  Statevector([ 0,  s,  s,  0]),
            "psi_minus": Statevector([ 0,  s, -s,  0]),
        }
        return svs[state]


class PhotonicBandgapSimulator:
    """
    Sweeps LDOS suppression η and measures Bell fidelity.
    Produces publication-ready plots.
    """
    def __init__(self, eta_min=1, eta_max=10000, n_points=50):
        self.etas = np.logspace(np.log10(eta_min), np.log10(eta_max), n_points)

    def sweep(self) -> dict:
        results = []
        for eta in self.etas:
            gen = BellStateGenerator(eta=eta)
            r   = gen.run("phi_plus")
            results.append(r)
            print(f"  η={eta:8.1f}  T1={r['T1_us']:8.1f}µs  F={r['fidelity']:.6f}")
        self.results = results
        return results

    def plot(self, save: str = "casimir_qiskit_sweep.png"):
        if not hasattr(self, "results"):
            self.sweep()
        etas = [r["eta"]      for r in self.results]
        fids = [r["fidelity"] for r in self.results]
        infid = [1 - f        for f in fids]

        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0d1117")
        for ax in [ax1, ax2]:
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="white", labelsize=11)
            ax.spines[["bottom","left","top","right"]].set_color("#333")
            ax.grid(True, color="#222", linewidth=0.8)

        ax1.plot(etas, fids, color="#58a6ff", linewidth=2.5)
        ax1.set_xscale("log")
        ax1.set_xlabel("LDOS Suppression η", fontsize=13, color="white")
        ax1.set_ylabel("Bell Fidelity F", fontsize=13, color="white")
        ax1.set_title("Bell Fidelity vs Casimir Vacuum Suppression", fontsize=13, color="white")
        ax1.set_ylim(0.98, 1.001)

        ax2.plot(etas, infid, color="#ff7b72", linewidth=2.5)
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.set_xlabel("LDOS Suppression η", fontsize=13, color="white")
        ax2.set_ylabel("Infidelity (1−F)", fontsize=13, color="white")
        ax2.set_title("Infidelity Decay (log-log)", fontsize=13, color="white")

        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print(f"Plot saved to {save}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Casimir Noise-Free Bell Generator — Qiskit Simulation")
    print("=" * 60)

    # Compare free space vs photonic bandgap
    for eta, label in [(1, "Free space"), (100, "Weak PBG"), (5000, "Deep PBG")]:
        gen = BellStateGenerator(eta=eta)
        results = gen.run_all()
        print(f"\n{label} (η={eta}, T1={eta*T1_FREE/1e3:.0f}µs):")
        for r in results:
            print(f"  {r['state']:12s}  F={r['fidelity']:.5f}  counts={r['counts']}")

    # Fidelity sweep
    print("\nRunning fidelity sweep...")
    sim = PhotonicBandgapSimulator(eta_min=1, eta_max=10000, n_points=130)
    sim.sweep()
    sim.plot("casimir_qiskit_sweep.png")
