# Coherence, Photon Energy, and “Geodesic Resistance” Near Strong Gravity

## A geometric thought experiment

**Abstract.** This note examines the hypothesis that if photonic coherence propagates causally at the speed of light, an analogous “atomic coherence” might encounter a gravitational resistance while approaching a compact object. In standard general relativity, the useful distinction is not between objects that do or do not resist a geodesic, but between (i) phase information and coherence, (ii) locally measured energy, and (iii) energy defined relative to a family of observers. A photon moving inward through a static gravitational field is blueshifted for static observers, while an outward-moving photon is redshifted. No universal energy “tax” is paid to the geodesic, and no new energy reservoir is required in the static-vacuum description. Atomic coherence can nevertheless acquire gravitationally dependent phase, and spatially distributed or internally entangled atoms can lose observable coherence through differential proper time, tidal effects, and ordinary environmental noise.

**Keywords:** quantum coherence, gravitational redshift, atomic clocks, Schwarzschild spacetime, geodesics, phase evolution, decoherence.

## 1. The key distinction

A coherent optical field is not a packet of coherence travelling as an independent substance. A field excitation, its phase correlations, and any entanglement propagate within the causal structure of spacetime. In vacuum, the wavefront follows null geodesics; the relevant causal speed is locally \(c\). Coherence can be transported, distorted, or degraded, but it is not itself a separately conserved energy-bearing medium.

The same logic applies to an atom. An atomic superposition such as

\[
|\psi\rangle = a|g\rangle+b e^{i\phi}|e\rangle
\]

has coherence represented by the off-diagonal density-matrix element \(\rho_{ge}\). Its phase evolves according to the atom’s proper time \(\tau\), approximately as

\[
\phi(\tau)=\phi_0-\frac{\Delta E}{\hbar}\tau,
\]

where \(\Delta E\) is the local internal energy splitting. Gravity changes the relationship between proper time and a chosen coordinate time. It therefore changes the phase accumulated by separated clocks or atoms, without requiring gravity to destroy coherence locally.

This is the first correction to the “geodesic resistance” intuition: curvature changes the phase geometry and the observer-dependent energy bookkeeping; it does not act like friction on a freely falling photon or atom.

## 2. Photon energy near a compact object

Consider the exterior Schwarzschild metric of a non-rotating, uncharged mass \(M\):

\[
ds^2=-f(r)c^2dt^2+f(r)^{-1}dr^2+r^2d\Omega^2,
\qquad
f(r)=1-\frac{r_s}{r},
\qquad
r_s=\frac{2GM}{c^2}.
\]

For a static observer at radius \(r\), the energy measured for a photon with conserved Killing energy \(E_\infty\) is

\[
E_{\rm loc}(r)=\frac{E_\infty}{\sqrt{f(r)}}.
\]

Therefore, if a photon travels inward from \(r_1\) to \(r_2<r_1\), static observers measure

\[
\frac{E_2}{E_1}=\sqrt{\frac{f(r_1)}{f(r_2)}}.
\]

The measured energy increment is consequently

\[
\Delta E=E_1\left[\sqrt{\frac{f(r_1)}{f(r_2)}}-1\right].
\]

For a photon of initial energy \(E_1=hf_1\), the frequency obeys the same ratio. The amount is not a universal number: it scales linearly with the initial photon energy and depends on both radii and the mass through \(r_s\).

In differential form,

\[
d\ln E_{\rm loc}=-\frac{r_s}{2r^2f(r)}\,dr.
\]

Since inward motion has \(dr<0\), the locally measured energy increases. This is gravitational blueshift. Conversely, a photon climbing outward is redshifted. For a static observer approaching the Schwarzschild horizon, the formula diverges because maintaining a fixed radius requires unbounded proper acceleration. This is not an infinite energy gain experienced by a freely falling observer; it is a limitation of the static-observer family at the horizon.

A useful operational statement is therefore:

> The photon does not encounter geodesic resistance. Static observers at smaller radius assign it a larger local energy, while the conserved quantity associated with the spacetime’s time-translation symmetry remains fixed.

The phrase “energy of the photon” is incomplete unless the observer or conserved reference is specified. In a time-dependent spacetime, such as one containing strong gravitational radiation, energy exchange can occur in a more literal sense. That is a different problem from propagation through a stationary Schwarzschild field.

## 3. Does coherence change when the photon is blueshifted?

The central optical effect is a rescaling of frequency, not automatic loss of phase coherence. In geometric optics, a mode can be described locally by a rapidly varying phase \(S\), with wave-vector

\[
k_\mu=\nabla_\mu S.
\]

The eikonal equation in vacuum is

\[
g^{\mu\nu}k_\mu k_\nu=0,
\]

and the wave-vector is parallel transported along the null ray to leading order. Curvature can focus the beam, alter its arrival time, shear a wavepacket, and produce frequency shifts. None of those effects alone is equivalent to irreversible decoherence.

Decoherence requires entanglement with uncontrolled degrees of freedom or averaging over unresolved variables. Examples include scattering, absorption, thermal emission, path-dependent frequency noise, gravitational lensing through unresolved geometries, and a source with finite bandwidth. A curved background can deform a quantum wavepacket even when it does not supply an environment that converts a pure state into a mixed state.

For a distributed photonic state, different portions of the state can acquire different gravitational phases. If those phases remain known and stable, they are unitary and can in principle be corrected. If they fluctuate or are ignored, the reduced state appears dephased. Thus “coherence affected by gravity” should be separated into:

- **Unitary phase evolution:** reversible gravitational redshift or propagation phase.
- **Mode deformation:** changes to bandwidth, pulse shape, focusing, or arrival-time structure.
- **Operational decoherence:** loss of interference visibility after averaging over unknown phases or tracing out correlated degrees of freedom.

## 4. Atomic coherence and gravitational time dilation

For an atom held static at radius \(r\), proper time satisfies

\[
d\tau=\sqrt{f(r)}\,dt.
\]

If the atom has transition frequency \(\omega_0=\Delta E/\hbar\) in its local frame, its phase relative to Schwarzschild coordinate time evolves as

\[
\frac{d\phi}{dt}=-\omega_0\sqrt{f(r)}.
\]

Two stationary atoms at radii \(r_A\) and \(r_B\) therefore accumulate a relative phase

\[
\Delta\phi(t)=\omega_0 t\left[\sqrt{f(r_B)}-\sqrt{f(r_A)}\right],
\]

up to the sign convention and any initial phase. In the weak-field limit, with Newtonian potential \(\Phi\),

\[
\frac{\Delta\omega}{\omega}\approx\frac{\Delta\Phi}{c^2}.
\]

This is atomic-clock gravitational redshift. It is not necessary to imagine an atom expending energy to overcome a geodesic. Rather, the atom’s internal oscillator samples a different proper-time rate.

A single atom in a spatial superposition can show the same physics. If its two wavepackets follow paths \(A\) and \(B\), then an internal superposition accumulates a path-dependent phase approximately

\[
\Delta\phi=\frac{\Delta E}{\hbar}\left(\tau_A-\tau_B\right).
\]

If the proper-time difference is stable, the effect is an interferometric phase shift. If the proper-time difference fluctuates with the environment or with uncontrolled motion, averaging can reduce the measured coherence.

## 5. Where curvature becomes genuinely important

Uniform acceleration and gravitational time dilation can often be described locally by the equivalence principle. Curvature appears when neighboring geodesics cannot be transformed away simultaneously. For an extended atomic wavepacket, the relevant scales are set by the Riemann tensor and the packet’s spatial separation.

A schematic phase-noise model is

\[
\rho_{ge}(t)=\rho_{ge}(0)\,e^{-i\langle\Delta\phi(t)\rangle}
\exp\left[-\frac{1}{2}\operatorname{Var}(\Delta\phi(t))\right].
\]

The first exponential is a coherent, reversible phase. The second represents visibility loss caused by uncertainty in the phase. Curvature contributes when the two branches of a superposition experience different proper times, tidal accelerations, or field couplings. The effect is amplified by:

- larger spatial separation of the atomic branches;
- larger internal frequency \(\Delta E/\hbar\);
- longer interrogation time;
- stronger tidal curvature rather than merely stronger coordinate acceleration;
- poorer control of motion, trapping, and electromagnetic or thermal environments.

Near a neutron star or black hole, the tidal tensor can become enormous, but the result is not a universal “coherence resistance coefficient.” It is a system-dependent phase and noise functional. The atom’s size, trajectory, state preparation, trap, and measurement protocol all matter.

## 6. A quantitative example

Take a photon sent inward from \(r_1=10r_s\) to \(r_2=2r_s\). Then

\[
\frac{E_2}{E_1}
=\sqrt{\frac{1-1/10}{1-1/2}}
=\sqrt{\frac{0.9}{0.5}}
\approx1.342.
\]

A 1 eV photon is therefore measured locally as approximately 1.342 eV by the static observer at \(2r_s\), an apparent increase of about 0.342 eV. The same photon climbing back from \(2r_s\) to \(10r_s\) is redshifted by the inverse factor.

For a small height difference \(\Delta h\) near Earth, the fractional frequency shift is approximately

\[
\frac{\Delta f}{f}\approx\frac{g\Delta h}{c^2}.
\]

At \(\Delta h=1\,\mathrm{m}\), this is about \(1.1\times10^{-16}\). For an optical transition near \(5\times10^{14}\,\mathrm{Hz}\), the frequency difference is roughly \(0.055\,\mathrm{Hz}\). The phase difference grows with interrogation time, which is why atomic clocks and interferometers can detect very small gravitational potential differences without requiring large energy transfers.

## 7. Testable formulation of the idea

The speculative idea becomes physically precise if “atomic coherence near gravity” is defined as a measurable visibility or phase observable. A useful experiment would prepare two identical atoms, or two branches of one atom, in coherent internal superpositions and place them on trajectories with different gravitational potentials. After recombination, measure

\[
V(T)=\frac{P_{+}(T)-P_{-}(T)}{P_{+}(T)+P_{-}(T)},
\]

where \(V\) is fringe visibility. A model should predict both

\[
\phi(T)=\frac{\Delta E}{\hbar}\Delta\tau(T)
\]

and a visibility envelope \(V(T)\) arising from specified phase noise, tidal coupling, and environmental channels. A pure gravitational phase shift is not evidence of decoherence; a reproducible reduction in visibility after controlling technical noise would be.

The strongest version of the hypothesis would require a new term in the evolution equation, for example a curvature-dependent Lindblad contribution. Such a term cannot be inserted merely because a geodesic is curved: it must identify the additional degrees of freedom that carry away which-path information and preserve complete positivity of the reduced quantum dynamics.

## 8. Conclusion

Approaching a high-gravity object does not impose a friction-like energy cost on a photon or a freely falling atom. In a stationary gravitational field, the locally measured photon energy increases inward according to the gravitational blueshift factor, while atomic coherence acquires phase according to proper time. The potentially observable “resistance” is better described as differential phase accumulation, tidal evolution, mode deformation, or decoherence from unresolved gravitationally induced correlations.

The productive research question is therefore not “how much energy must coherence give to the geodesic?” but:

> How does spacetime geometry map into the phase, mode structure, and reduced density matrix of a quantum system whose branches sample different proper times and tidal fields?

That formulation is compatible with general relativity, quantum interferometry, and experimental observables, while leaving room for genuinely new physics if measured coherence loss exceeds the unitary curved-spacetime prediction.

## References

1. Einstein Online, “Gravitational redshift,” https://www.einstein-online.info/en/explandict/gravitational-redshift/
2. L. C. B. Crispino et al., “The influence of the Earth’s curved spacetime on Gaussian quantum coherence,” arXiv:1910.02595, https://arxiv.org/abs/1910.02595
3. A. Roura, “Gravitational redshift in quantum-clock interferometry,” *Physical Review X* 10, 021014 (2020), https://link.aps.org/doi/10.1103/PhysRevX.10.021014
4. F. Di Pumpo et al., “Gravitational redshift tests with atomic clocks and atom interferometers,” *PRX Quantum* 2, 040333 (2021), https://link.aps.org/doi/10.1103/PRXQuantum.2.040333
5. “Decoherence due to spacetime curvature,” arXiv:2302.09038, https://arxiv.org/abs/2302.09038
6. “Atom-field dynamics in curved spacetime,” arXiv:2307.12222, https://arxiv.org/html/2307.12222v3
7. “Interplay between gravity and quantum coherence in a pulse of light propagating in curved spacetime,” arXiv:2106.12424, https://arxiv.org/pdf/2106.12424.pdf
