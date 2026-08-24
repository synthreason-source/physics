#!/usr/bin/env python3
"""
Adiabatic Eigenvector State Transfer
------------------------------------

Three coupled sites:

    q0 <----> q1 <----> q2
    LEFT     MIDDLE     RIGHT

The couplings are changed counter-intuitively:

    J_right(t) turns on FIRST
    J_left(t)  turns on SECOND

This creates a dark eigenstate approximately

    |D(t)> = cos(theta)|q0> - sin(theta)|q2>

where

    tan(theta) = J_left / J_right

At the beginning:

    theta ~ 0

    |D> ~ |q0>

At the end:

    theta ~ pi/2

    |D> ~ -|q2>

Thus the eigenvector itself is continuously deformed
from the left site to the right site.

No measurement is performed during the transfer.

The code compares:

1. Actual Schrödinger evolution
2. Instantaneous eigenvector
3. Probability on each site
4. Fidelity with the target
5. Eigenvalue spectrum
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS
# ============================================================

HBAR = 1.0

T = 100.0

# Time resolution
N_STEPS = 5000

# Maximum coupling
J_MAX = 1.0

# Pulse width
SIGMA = 15.0

# Pulse centers
T_LEFT = 65.0
T_RIGHT = 35.0

times = np.linspace(0.0, T, N_STEPS)
dt = times[1] - times[0]


# ============================================================
# COUPLING FUNCTIONS
# ============================================================

def gaussian(t, center, sigma):
    return np.exp(
        -((t - center) ** 2) / (2.0 * sigma ** 2)
    )


def J_left(t):
    """
    Coupling q0 <-> q1

    Turns on later.
    """
    return J_MAX * gaussian(t, T_LEFT, SIGMA)


def J_right(t):
    """
    Coupling q1 <-> q2

    Turns on earlier.
    """
    return J_MAX * gaussian(t, T_RIGHT, SIGMA)


# ============================================================
# HAMILTONIAN
# ============================================================

def hamiltonian(t):
    """
    Single-excitation Hamiltonian.

    Basis:

        |100>
        |010>
        |001>

    represented as:

        |q0>
        |q1>
        |q2>
    """

    jl = J_left(t)
    jr = J_right(t)

    H = np.array(
        [
            [0.0, jl, 0.0],
            [jl, 0.0, jr],
            [0.0, jr, 0.0],
        ],
        dtype=complex,
    )

    return H


# ============================================================
# EIGENVECTOR TRACKING
# ============================================================

def instantaneous_eigensystem(t):
    H = hamiltonian(t)

    eigenvalues, eigenvectors = np.linalg.eigh(H)

    return eigenvalues, eigenvectors


# ============================================================
# IDENTIFY DARK STATE
# ============================================================

def dark_state(t):
    """
    Analytic zero-energy eigenstate:

        |D> =
        [ Jr, 0, -Jl ] / sqrt(Jr^2 + Jl^2)

    Depending on convention, an overall phase/sign is irrelevant.
    """

    jl = J_left(t)
    jr = J_right(t)

    norm = np.sqrt(jr * jr + jl * jl)

    if norm < 1e-14:
        return np.array([1.0, 0.0, 0.0], dtype=complex)

    return np.array(
        [
            jr / norm,
            0.0,
            -jl / norm,
        ],
        dtype=complex,
    )


# ============================================================
# UNITARY PROPAGATION
# ============================================================

def propagate_step(psi, H, dt):
    """
    Exact propagation for a time-independent Hamiltonian
    over one sufficiently small timestep.

        psi(t+dt) = exp(-i H dt) psi(t)
    """

    eigenvalues, eigenvectors = np.linalg.eigh(H)

    U = (
        eigenvectors
        @ np.diag(np.exp(-1j * eigenvalues * dt / HBAR))
        @ eigenvectors.conj().T
    )

    return U @ psi


# ============================================================
# INITIAL STATE
# ============================================================

# Initially localized entirely on q0.

psi = np.array(
    [
        1.0,
        0.0,
        0.0,
    ],
    dtype=complex,
)

# Normalize
psi /= np.linalg.norm(psi)


# ============================================================
# STORAGE
# ============================================================

probabilities = np.zeros((N_STEPS, 3))

dark_probabilities = np.zeros((N_STEPS, 3))

fidelity_dark = np.zeros(N_STEPS)

fidelity_target = np.zeros(N_STEPS)

eigenvalues_history = np.zeros((N_STEPS, 3))

coupling_left = np.zeros(N_STEPS)
coupling_right = np.zeros(N_STEPS)


# ============================================================
# SIMULATION
# ============================================================

for i, t in enumerate(times):

    H = hamiltonian(t)

    # Record probabilities
    probabilities[i] = np.abs(psi) ** 2

    # Record couplings
    coupling_left[i] = J_left(t)
    coupling_right[i] = J_right(t)

    # Instantaneous eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    eigenvalues_history[i] = eigenvalues

    # Analytic dark state
    dark = dark_state(t)

    dark_probabilities[i] = np.abs(dark) ** 2

    # Fidelity with instantaneous dark state
    overlap = np.vdot(dark, psi)

    fidelity_dark[i] = np.abs(overlap) ** 2

    # Fidelity with target |q2>
    target = np.array(
        [
            0.0,
            0.0,
            1.0,
        ],
        dtype=complex,
    )

    fidelity_target[i] = np.abs(
        np.vdot(target, psi)
    ) ** 2

    # Propagate to next timestep
    if i < N_STEPS - 1:

        # Midpoint Hamiltonian gives better accuracy
        t_mid = t + dt / 2.0

        H_mid = hamiltonian(t_mid)

        psi = propagate_step(
            psi,
            H_mid,
            dt,
        )

        # Numerical normalization
        psi /= np.linalg.norm(psi)


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("ADIABATIC EIGENVECTOR TRANSFER")
print("=" * 60)

print()
print("Initial state:")
print(psi)

print()
print("Final probabilities:")
print(f"q0 = {probabilities[-1, 0]:.12f}")
print(f"q1 = {probabilities[-1, 1]:.12f}")
print(f"q2 = {probabilities[-1, 2]:.12f}")

print()
print("Final target fidelity:")
print(f"{fidelity_target[-1]:.12f}")

print()
print("Final dark-state fidelity:")
print(f"{fidelity_dark[-1]:.12f}")

print()
print("Maximum middle-site population:")
print(f"{np.max(probabilities[:, 1]):.12f}")

print()
print("=" * 60)


# ============================================================
# PLOT 1 — PHYSICAL PROBABILITY
# ============================================================

plt.figure(figsize=(11, 6))

plt.plot(
    times,
    probabilities[:, 0],
    label="q0 probability"
)

plt.plot(
    times,
    probabilities[:, 1],
    label="q1 probability"
)

plt.plot(
    times,
    probabilities[:, 2],
    label="q2 probability"
)

plt.xlabel("Time")
plt.ylabel("Probability")

plt.title(
    "Adiabatic Quantum State Transfer"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()


# ============================================================
# PLOT 2 — INSTANTANEOUS DARK EIGENVECTOR
# ============================================================

plt.figure(figsize=(11, 6))

plt.plot(
    times,
    dark_probabilities[:, 0],
    label="Dark-state |q0|²"
)

plt.plot(
    times,
    dark_probabilities[:, 1],
    label="Dark-state |q1|²"
)

plt.plot(
    times,
    dark_probabilities[:, 2],
    label="Dark-state |q2|²"
)

plt.xlabel("Time")
plt.ylabel("Eigenvector weight")

plt.title(
    "Deformation of the Instantaneous Eigenvector"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()


# ============================================================
# PLOT 3 — HAMILTONIAN COUPLINGS
# ============================================================

plt.figure(figsize=(11, 6))

plt.plot(
    times,
    coupling_left,
    label="J_left : q0 ↔ q1"
)

plt.plot(
    times,
    coupling_right,
    label="J_right : q1 ↔ q2"
)

plt.xlabel("Time")
plt.ylabel("Coupling strength")

plt.title(
    "Counter-Intuitive Coupling Sequence"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()


# ============================================================
# PLOT 4 — EIGENVALUE SPECTRUM
# ============================================================

plt.figure(figsize=(11, 6))

plt.plot(
    times,
    eigenvalues_history[:, 0],
    label="E0"
)

plt.plot(
    times,
    eigenvalues_history[:, 1],
    label="E1 (dark state)"
)

plt.plot(
    times,
    eigenvalues_history[:, 2],
    label="E2"
)

plt.xlabel("Time")
plt.ylabel("Energy")

plt.title(
    "Instantaneous Hamiltonian Eigenvalues"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()


# ============================================================
# PLOT 5 — FIDELITY
# ============================================================

plt.figure(figsize=(11, 6))

plt.plot(
    times,
    fidelity_dark,
    label="Fidelity with instantaneous dark state"
)

plt.plot(
    times,
    fidelity_target,
    label="Fidelity with |q2>"
)

plt.xlabel("Time")
plt.ylabel("Fidelity")

plt.ylim(0.0, 1.05)

plt.title(
    "Adiabatic Following and Final Transfer"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()