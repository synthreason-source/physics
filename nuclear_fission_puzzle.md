# Coherent Picture: Nuclear Fission Physics Puzzle

> **Solution:** The six fragments collectively describe the **complete end-to-end physics of nuclear fission** — from nuclear instability and quantum tunneling, through statistical mechanics and radiation, to mass spectrometry of products and gamma-ray shielding.

---

## Fragment Breakdown

### 1. "Transverse E-field + fission" → Bohr-Wheeler Liquid Drop

The transverse electric field is the **Coulomb repulsion** between the two halves of a deforming nucleus. The Bohr-Wheeler liquid drop model quantifies instability via the **fissility parameter**:

$$x = \frac{Z^2}{A}$$

Where:
- $Z$ = atomic number (proton count)
- $A$ = mass number (total nucleons)

When $x \geq 1$, Coulomb energy exceeds surface tension energy and the nucleus is **spontaneously unstable** against fission. Uranium-235 has $x \approx 0.71$, requiring a neutron to push it over the barrier.

---

### 2. "Unit charge + work done" → Electric Potential + Hill-Wheeler

"Unit charge × potential" = electric potential energy = the **fission barrier height** $V_0$. The Hill-Wheeler formula gives the quantum tunneling transmission probability through this parabolic barrier:

$$T(E) = \frac{1}{1 + \exp\!\left(\frac{2\pi(V_0 - E)}{\hbar\omega_f}\right)}$$

Where:
- $V_0$ = barrier height
- $E$ = excitation energy
- $\hbar\omega_f$ = barrier curvature energy

This sigmoid smoothly transitions from $T \to 0$ (sub-barrier, pure tunneling) to $T \to 1$ (above-barrier, classical passage).

---

### 3. "Statistical Distribution" → Fermi-Dirac Analog

The Hill-Wheeler sigmoid is **mathematically identical** to the Fermi-Dirac distribution:

$$f(E) = \frac{1}{1 + \exp\!\left(\frac{E - \mu}{k_BT}\right)}$$

A 2025 study (PMC, *Entropy* 27, 227) formally established this correspondence:

| Hill-Wheeler Parameter | Fermi-Dirac Analog |
|---|---|
| Barrier height $V_0$ | Chemical potential $\mu$ |
| Barrier curvature $\hbar\omega_f$ | Thermal energy $k_BT$ |
| Excitation energy $E$ | Particle energy $E$ |
| Tunneling probability $T(E)$ | Occupation probability $f(E)$ |

Fission channel occupation thus follows **Fermi-Dirac statistics**, with the barrier acting as a band gap.

---

### 4. "k²w⁴ / 2πr" → Dipole/Larmor Radiation

As fission fragments (highly charged ions, $Z_1 + Z_2 = Z_{parent}$) accelerate apart under Coulomb repulsion, they emit **dipole/Larmor radiation**:

$$P \propto \omega^4$$

This is the prompt gamma flash emitted during fragment separation. The $\omega^4$ dependence means high-frequency (gamma-ray) emission is strongly favoured. This is analogous to Rayleigh scattering's $\lambda^{-4}$ law, but for accelerating charges rather than photon scattering.

---

### 5. "Mass Spectrograph" → Semi-Empirical Mass Formula

The mass distribution of fission fragments — the classic **double-humped yield curve** — is predicted by the Bethe-Weizsäcker semi-empirical mass formula:

$$M(Z, A) = Z m_p + N m_n - a_V A + a_S A^{2/3} + a_C \frac{Z^2}{A^{1/3}} + a_A \frac{(A-2Z)^2}{A} \pm \delta$$

Terms:
- $a_V A$ — volume (bulk binding)
- $a_S A^{2/3}$ — surface (liquid drop surface tension)
- $a_C Z^2/A^{1/3}$ — Coulomb repulsion
- $a_A (A-2Z)^2/A$ — isospin asymmetry
- $\pm\delta$ — pairing (even-even vs. odd-odd)

Mass spectrometry of fragments experimentally maps this formula onto real yields.

---

### 6. "Mass Absorption Coefficient + Walls" → Gamma-Ray Shielding

After fragments beta-decay and emit gamma rays, **Beer-Lambert attenuation** governs shielding:

$$I = I_0 \, e^{-\mu x}$$

Where:
- $I_0$ = incident gamma intensity
- $\mu$ = mass absorption coefficient (material-dependent, cm²/g)
- $x$ = shield areal density (g/cm²)

This is the **engineering endpoint** of fission physics — designing reactor walls, spent-fuel casks, and bunkers.

---

## The Coherent Picture

```
  Nucleus
    │
    ▼
[Fissility x = Z²/A]          ← Is it fissile? (Bohr-Wheeler)
    │
    ▼
[T(E) Hill-Wheeler sigmoid]   ← Can it tunnel/cross the barrier?
    │
    ▼
[Fermi-Dirac occupation]      ← Statistical probability of each channel
    │
    ▼
[P ∝ ω⁴ Larmor radiation]    ← Prompt gamma emission during separation
    │
    ▼
[Semi-empirical mass formula] ← Which fragments are produced?
    │
    ▼
[I = I₀ e^{-μx} shielding]   ← How to contain the gamma radiation?
```

---

## Key References

| Source | Content |
|---|---|
| Bohr & Wheeler, *Phys. Rev.* 56, 426 (1939) | Original liquid drop fission mechanism |
| Hill & Wheeler, *Phys. Rev.* 89, 1102 (1953) | Fission barrier tunneling formula |
| Mişicu et al., *Entropy* 27, 227 (2025) — PMC | Hill-Wheeler as Fermi-Dirac analog |
| Bethe & Weizsäcker (1935–1936) | Semi-empirical nuclear mass formula |
| NRC / NIST | Gamma-ray mass attenuation coefficients |

---

*Puzzle solved: Each fragment is a cryptic clue to one layer of fission physics. The word "fragments" is itself a pun — both nuclear fission products and puzzle pieces.*
