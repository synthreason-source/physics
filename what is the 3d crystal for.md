<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# what is the 3d crystal for

The 3D crystal has one job: **reshape the vacuum so qubits stop decohering.** Here's what it does at each level:

## The Single Core Function

A qubit loses its quantum state because it spontaneously emits a photon into the vacuum — the vacuum "absorbs" the qubit's energy. The 3D crystal makes that **physically impossible** at the qubit's frequency by leaving no vacuum modes for the photon to go into. The qubit tries to radiate, finds no modes available, and stays in superposition.[^1][^2]

## Why 3D Specifically

| Geometry | What it blocks | Problem |
| :-- | :-- | :-- |
| 1D (flat plates / copper pipe row) | Modes in one direction only | Photons escape sideways |
| 2D (holes in a slab) | Modes in a plane | Photons escape up/down |
| **3D (woodpile)** | **Modes in all directions** | **No escape path — complete bandgap** |

The 3D woodpile is the minimum structure needed for a **complete photonic bandgap** — LDOS = 0 in every direction simultaneously. Anything less and the qubit just leaks through the unblocked directions.[^3][^4]

## The Chain of Purpose

```
3D Crystal
    ↓  suppresses vacuum modes at ω₀ (all directions)
LDOS = 0 at qubit frequency
    ↓  spontaneous emission rate γ = LDOS × coupling → 0
T₁ → ∞  (qubit holds state indefinitely)
    ↓  gate time << T₁
Bell state fidelity → 1.0
    ↓  measured in casimir_qc.py
η = measured from NanoVNA S21 → plugged into CasimirNoiseModel(eta)
```


## In One Sentence

The 3D crystal is a **vacuum filter** — it removes the specific frequency of electromagnetic vacuum fluctuations that would otherwise collapse your qubit's superposition, which is the Casimir effect applied as a coherence-preservation tool.[^5][^1]

<div align="center">⁂</div>

[^1]: https://arxiv.org/html/2506.10210v1

[^2]: https://nano-cops.com/2020/06/shielding-vacuum-noise-with-3d-nanostructures/

[^3]: https://link.aps.org/doi/10.1103/pkkz-y5lp

[^4]: https://arxiv.org/pdf/cond-mat/0408459.pdf

[^5]: https://arxiv.org/html/2509.05156v1

