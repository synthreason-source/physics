# Photoemission-Driven Plasma Formation and Thermal Response of a Conductive Rod Irradiated by Hard X-Rays in a Hydrogen Environment: A Theoretical Analysis

**Type:** Conceptual / order-of-magnitude theoretical note
**Scope:** Qualitative and semi-quantitative treatment; not a validated simulation result

---

## Abstract

We present a theoretical analysis of the physical processes that occur when a conductive rod immersed in a low-pressure hydrogen environment is irradiated with hard X-rays. Because the photoelectric absorption cross-section scales approximately as *Z*⁴–*Z*⁵, hydrogen (*Z* = 1) is nearly transparent to hard X-rays, while a metallic rod of much higher atomic number absorbs the bulk of the incident flux. This absorption drives photoelectric emission and Auger electron cascades from the rod surface, which (a) electrically charge the rod (or, if grounded, drive a photocurrent), and (b) collisionally ionize the surrounding hydrogen gas, forming a thin, transient, strongly non-equilibrium plasma shell. We develop a two-temperature description of this plasma, distinguish short-pulse from long-pulse/continuous irradiation regimes, and derive the energy-balance conditions under which the rod undergoes conventional thermal melting versus ultrafast, non-thermal disintegration (Coulomb explosion). We conclude with a qualitative regime map organized by pulse duration and absorbed fluence.

---

## 1. Introduction

Hard X-ray interaction with matter is dominated by three competing processes: the photoelectric effect, Compton (incoherent) scattering, and, at very high energies, pair production. For light elements and X-ray energies in the tens-of-keV range relevant here, the photoelectric effect dominates absorption in high-*Z* materials, while low-*Z* gases such as hydrogen interact only weakly. This asymmetry is the organizing principle of the present analysis: a hydrogen environment is, to first order, a passive, largely transparent medium, while an embedded high-*Z* conductor becomes the dominant absorber and the source of essentially all secondary physics in the system.

The purpose of this paper is to walk through, in order, the chain of physical effects that follow from irradiating such a rod with hard X-rays: direct absorption and photoemission from the rod; charging and photocurrent behavior; secondary ionization of the surrounding gas and plasma-shell formation; the thermal (non-)equilibrium between electrons and heavy particles in that plasma; the transition from short-pulse to long-pulse behavior; and finally the conditions under which the rod itself is thermally or non-thermally damaged.

---

## 2. Physical System and Assumptions

We consider:

- A conductive rod of high atomic number (e.g., a transition or noble metal), either electrically isolated or grounded.
- The rod immersed in a volume of hydrogen gas (H₂), at low-to-moderate pressure, referred to loosely as a "ball" of gas surrounding the rod.
- A hard X-ray source (photon energies of order 10–100 keV), applied either as a short, intense pulse (e.g., an X-ray free-electron laser, XFEL, delivering fluence in femtoseconds) or as a longer/continuous exposure (e.g., a synchrotron beamline or X-ray tube operating over microseconds to seconds).

We assume vacuum-UV/X-ray-relevant photoemission physics applies at the rod surface, that the surrounding hydrogen is otherwise unperturbed prior to irradiation, and that geometric effects (beam collimation, rod aspect ratio) are secondary to the energy-balance arguments developed below.

---

## 3. X-Ray Interaction with the Conductive Rod

### 3.1 Absorption asymmetry

The photoelectric absorption coefficient per atom scales approximately as

  τ ∝ Z⁴–Z⁵ / E³ᐟ²

(the exact exponents depend on the energy regime relative to absorption edges). For hydrogen (*Z* = 1) versus a typical structural or noble metal (*Z* ≈ 13–79), this represents several orders of magnitude difference in absorption cross-section per atom. Combined with hydrogen's low density as a gas, the hydrogen "ball" is essentially transparent to a hard X-ray beam over laboratory-scale path lengths, while the rod intercepts and absorbs the majority of incident photons that strike it.

### 3.2 Photoelectric emission and secondary cascades

Each absorbed photon that undergoes photoelectric absorption in the rod ejects a photoelectron with kinetic energy

  Eₑ ≈ Eₚₕₒₜₒₙ − E_binding

where E_binding is the relevant electron shell binding energy. For hard X-rays this can leave the photoelectron with kinetic energy of several to tens of keV. The resulting inner-shell vacancy is filled either radiatively (characteristic fluorescence X-ray emission) or non-radiatively (Auger electron emission), producing an additional lower-energy (typically sub-keV to few-keV) electron. The net effect of sustained irradiation is a continuous flux of energetic electrons — plus some re-emitted, softer X-rays — leaving the rod's surface.

---

## 4. Rod Charging Dynamics

### 4.1 Isolated rod

If the rod is electrically isolated, each photoemission event removes one electron from the rod, producing a positive surface charge. As the rod charges, an increasingly strong retarding electrostatic potential develops, which suppresses escape of the lowest-energy photoelectrons first. The rod approaches a self-limiting equilibrium potential at which the retarding field balances the outgoing photoelectron flux — analogous to the well-studied phenomenon of spacecraft charging under solar X-ray/UV exposure.

### 4.2 Grounded rod

If the rod is grounded, charge is continuously replenished from the ground connection, and the system instead exhibits a steady (or pulsed) photocurrent proportional to the net electron emission rate. In this configuration the rod behaves analogously to a photoemissive X-ray detector.

---

## 5. Secondary Ionization and Plasma-Shell Formation

The photoelectrons and Auger electrons ejected from the rod do not travel far in the surrounding hydrogen before losing energy through inelastic collisions — principally electron-impact ionization and excitation of H₂. Each ionization event costs approximately 15–40 eV (comparable to the ionization potential of H₂, ~15.4 eV, plus inelastic losses), meaning a single ~10 keV primary electron can trigger on the order of hundreds of secondary ionizations before thermalizing.

This produces:

- **Free electrons**, spanning a wide energy range from the original fast primaries down to a much larger population of "sub-excitation" secondary electrons with only a few eV of kinetic energy each.
- **Positive ions** (predominantly H₂⁺, with some dissociated H⁺), created directly by electron-impact ionization.
- Some fraction of excited or dissociated neutral species (excited H₂, atomic H) from collisions that excite or dissociate without ionizing.

Because the ionizing electrons only travel a finite range before losing their energy, this ionization — and the resulting plasma — is spatially confined to a thin shell immediately surrounding the rod's surface, rather than filling the bulk gas volume.

---

## 6. Non-Equilibrium (Two-Temperature) Plasma Description

### 6.1 Why a single temperature is inadequate

The plasma shell formed under short-pulse irradiation is strongly non-thermal. Electrons and heavy particles (ions, neutrals) do not share a common temperature on short timescales, for two reasons:

1. **Large mass mismatch.** In an elastic electron–ion (or electron–neutral) collision, the maximum fractional kinetic energy transferred is of order 4mₑ/M (~1/500 for hydrogen), so electrons transfer only a small fraction of their energy per collision to the much heavier ions.
2. **Finite equilibration time.** Full electron–ion thermal equilibration requires many such collisions, occurring over a characteristic equilibration time τ_eq that depends on density and electron energy — typically nanoseconds to microseconds for the densities and energies considered here.

### 6.2 Effective electron and heavy-particle temperatures

We can define two approximate quantities:

- An **electron temperature** T_e, describing the (non-Maxwellian, but often approximated as such for the bulk secondary population) thermal spread of the free electrons — of order 1–10 eV (roughly 10⁴–10⁵ K) for the bulk of the secondary cascade, with a smaller high-energy tail extending to the keV range associated with the primary photo/Auger electrons.
- A **heavy-particle (ion/neutral) temperature** T_i ≈ T_gas, which remains close to the ambient gas temperature (~300 K) on short timescales, since energy transfer into the heavy species is slow relative to the pulse duration.

This T_e ≫ T_i condition is the hallmark of a strongly non-equilibrium plasma, structurally similar to those produced in low-temperature discharge plasmas or short-pulse laser-produced plasmas.

---

## 7. Long-Pulse and Continuous Irradiation: Approach to Local Thermal Equilibrium

### 7.1 Sustained ionization and expanding plasma volume

Under continuous or long-duration X-ray exposure, photoemission and secondary ionization are continuously replenished rather than occurring as a single transient burst. The ionized shell can grow in extent as electron diffusion and cascading secondary ionization proceed further from the rod surface, and the local ionization fraction increases over time.

### 7.2 Electron–heavy particle equilibration

If the irradiation duration exceeds the local electron–ion/neutral equilibration time τ_eq, repeated collisions transfer enough cumulative energy that T_i begins to rise toward T_e. In the limit of sufficiently long exposure and sufficiently high density (short mean free path, frequent collisions), the plasma can approach local thermodynamic equilibrium (LTE), in which a single temperature T meaningfully characterizes the shell.

### 7.3 Energy balance

The equilibrium (or quasi-steady-state) temperature reached is set by a balance between:

  P_in (absorbed X-ray power density, converted via photoemission/collisional ionization into plasma thermal energy)

and loss terms including:

  P_cond (thermal conduction to the cooler surrounding gas)
  P_rad (bremsstrahlung and recombination radiation)
  P_expansion (adiabatic cooling if the plasma is unconfined and expands)

Order-of-magnitude estimates for sustained X-ray-driven plasmas of this general type place the equilibrated bulk temperature in the range of roughly 1–100 eV (≈10⁴–10⁶ K), with the precise value highly sensitive to flux, gas density, and confinement geometry — this is fundamentally an energy-balance problem rather than a fixed material property.

### 7.4 Rod charging under continuous exposure

The rod's charging behavior also changes qualitatively: an isolated rod under continuous irradiation settles into a steady-state equilibrium potential (rather than a single transient charging event), while a grounded rod exhibits a steady photocurrent rather than a sharp pulse.

---

## 8. Thermal Response of the Rod: Melting and Ablation

### 8.1 Direct absorption dominates over plasma back-heating

Because the rod itself absorbs the large majority of incident X-ray photons directly (Section 3.1), the dominant heating pathway for the rod is direct photon absorption and subsequent electron–phonon energy transfer into the lattice — not conductive heating from the (relatively dilute, thin) surrounding plasma shell. The plasma's contribution to rod heating becomes significant only if the hydrogen is at substantially elevated pressure/density, where convective/conductive heat transfer from a denser plasma back onto the rod surface becomes non-negligible.

### 8.2 Thermal (long-pulse) regime

When the absorbed energy is deposited on a timescale longer than the characteristic electron–phonon coupling and thermal diffusion times (typically picoseconds to nanoseconds in metals), the rod's response is governed by a standard heat-conduction energy balance:

  ρ c_p (∂T/∂t) = Q(x, t) − ∇·**q**

where Q(x, t) is the local absorbed X-ray power density and **q** is the conductive heat flux. If the local absorbed power density exceeds what conduction can carry away, the local temperature rises past the melting point T_m, and the rod undergoes conventional melting — physically analogous to laser- or electron-beam-induced melting, with X-rays simply serving as the energy-deposition mechanism.

### 8.3 Ultrafast (non-thermal) regime

For sufficiently short and intense pulses — the regime realized by X-ray free-electron lasers — energy can be deposited faster than the lattice can respond thermally (femtosecond timescales, faster than electron–phonon coupling and much faster than lattice heat diffusion). In this regime, a large fraction of the rod's electrons can be photoionized away before the ion lattice has time to relax, leaving behind a lattice of unscreened positive ions that mutually repel. If the resulting electrostatic stress exceeds the material's cohesive strength, the rod undergoes a **Coulomb explosion** — rapid disintegration driven by electrostatic repulsion rather than by thermal melting. This effect is well documented in XFEL single-shot damage studies and represents a known practical limitation for X-ray imaging of radiation-sensitive samples.

---

## 9. Discussion: A Qualitative Regime Map

The system's outcome can be organized qualitatively along two axes — pulse duration and absorbed fluence (energy density) — yielding three broad regimes:

1. **Low fluence, any duration:** Photoemission and rod charging/photocurrent occur, with a thin, weakly ionized plasma shell forming in the surrounding hydrogen; the rod itself experiences negligible net heating.
2. **Moderate-to-high fluence, long/continuous duration:** The rod undergoes conventional thermal heating and, above threshold, classical melting; the surrounding plasma shell approaches quasi-steady-state and may partially thermalize toward a common electron–ion temperature.
3. **Very high fluence, ultrashort duration (XFEL-class):** Direct Coulomb explosion of the rod's near-surface region can occur before thermal melting has time to develop, bypassing the classical melting pathway entirely.

The surrounding hydrogen's own bulk plasma state remains, in all these regimes, a secondary and comparatively dilute phenomenon relative to the direct rod–X-ray interaction, owing to hydrogen's intrinsically weak hard-X-ray absorption cross-section.

---

## 10. Conclusion

Irradiating a conductive rod embedded in a hydrogen environment with hard X-rays produces a chain of effects dominated almost entirely by the rod: direct photoelectric absorption, photoelectron/Auger emission, rod charging or photocurrent generation, and — as a secondary consequence — a thin, strongly non-equilibrium plasma shell formed by secondary ionization of the surrounding gas. This plasma is characterized by a large electron–ion temperature disparity under short-pulse conditions, which relaxes toward a more thermalized state under sustained, long-duration irradiation, with the eventual equilibrium temperature set by an energy-balance calculation rather than a fixed value. Whether the rod itself melts, survives largely unaffected, or undergoes non-thermal Coulomb-explosion-driven disintegration depends primarily on the interplay between absorbed fluence and pulse duration relative to the material's thermal and electronic relaxation timescales — with the surrounding hydrogen plasma playing a distinctly secondary role in the rod's thermal fate except at elevated gas densities.

---

## References (General Background)

1. Jackson, J. D. *Classical Electrodynamics*, 3rd ed. — for photon–matter interaction fundamentals.
2. Chen, F. F. *Introduction to Plasma Physics and Controlled Fusion*, 3rd ed. — for two-temperature plasma and equilibration concepts.
3. NIST X-Ray Attenuation and Absorption Databases (XCOM / X-Ray Mass Attenuation Coefficients) — for photoelectric cross-section scaling and Z-dependence.
4. Krause, M. O. "Atomic Radiative and Radiationless Yields for K and L Shells," *Journal of Physical and Chemical Reference Data* — for fluorescence/Auger yield background.
5. General XFEL damage literature on non-thermal (Coulomb explosion) ablation in ultrafast X-ray free-electron laser experiments, as a well-established phenomenon in single-shot diffraction imaging studies.

*Note: References are provided as general background sources for the underlying established physics discussed above, not as citations for specific numerical results in this note, which are order-of-magnitude estimates.*
